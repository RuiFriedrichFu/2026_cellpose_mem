from __future__ import annotations

import os
import glob
from typing import List, Optional, Literal

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from cellpose import plot, utils, models
from readlif.reader import LifFile
from skimage.restoration import rolling_ball


# -----------------------------
# Leica .lif reader
# -----------------------------
def get_lif(path: str, imgnum: int = 1, z: int = 0, t: int = 0, m: int = 0) -> np.ndarray:
    img_obj = LifFile(path).get_image(imgnum)
    frame = img_obj.get_frame(z, t, m)
    arr = np.array(frame).astype(np.float32)
    return arr


# -----------------------------
# Mask / region processing
# -----------------------------
def mask_single(masks: np.ndarray, num: int = 0) -> np.ndarray:
    target_label = num + 1
    result = (masks == target_label).astype(np.uint8)
    return result


def mask_dilated(mask: np.ndarray, n_iter: int = 3, kernel_size: int = 3) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if n_iter > 0:
        out = cv2.dilate(mask, kernel, iterations=n_iter)
    elif n_iter < 0:
        out = cv2.erode(mask, kernel, iterations=-n_iter)
    else:
        out = mask.copy()
    return out


def create_band(mask: np.ndarray, dilation: int = 2, erosion: int = 2) -> np.ndarray:
    dilated = mask_dilated(mask, n_iter=dilation)
    eroded = mask_dilated(mask, n_iter=-erosion)
    band = cv2.subtract(dilated, eroded)
    return (band != 0).astype(np.uint8)


def create_region_masks(mask: np.ndarray, dilation: int = 1, erosion: int = 1) -> dict[str, np.ndarray]:
    outer_total = mask_dilated(mask, n_iter=dilation)
    inner_total = mask_dilated(mask, n_iter=-erosion)
    membrane = cv2.subtract(outer_total, inner_total)
    return {
        "outer_total": (outer_total != 0).astype(np.uint8),
        "inner_total": (inner_total != 0).astype(np.uint8),
        "membrane": (membrane != 0).astype(np.uint8),
    }


def band_combined(masks: np.ndarray, dilation: int = 1, erosion: int = 1) -> np.ndarray:
    labels = np.unique(masks)
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.zeros_like(masks, dtype=np.uint8)

    combined = np.zeros_like(masks, dtype=np.uint8)
    for lab in labels:
        single = (masks == lab).astype(np.uint8)
        band = create_band(single, dilation=dilation, erosion=erosion)
        combined = np.maximum(combined, band)
    return (combined != 0).astype(np.uint8)


# -----------------------------
# Backward-compatible visualization helpers
# -----------------------------
def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    if image.dtype != np.uint8:
        img8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = image
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)], dtype=np.float32)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img8, lut)


def merge_plt(
    img1: np.ndarray,
    img2: np.ndarray,
    filename: str = "merged.tif",
    gamma1: float = 0.5,
    gamma2: float = 1.0,
) -> np.ndarray:
    img1n = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2n = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    g1 = gamma_correction(img1n, gamma=gamma1)
    g2 = gamma_correction(img2n, gamma=gamma2)

    h, w = img1n.shape[:2]
    merged = np.zeros((h, w, 3), dtype=np.uint8)
    merged[..., 1] = g1
    merged[..., 2] = g1
    merged[..., 0] = g2
    merged[..., 2] = cv2.add(merged[..., 2], g2)

    cv2.imwrite(filename, merged)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
    plt.title("Merged Image (Cyan + Magenta)")
    plt.axis("off")
    plt.show()

    return merged


# -----------------------------
# Statistics / export
# -----------------------------
def get_area(region_mask: np.ndarray) -> int:
    return int(np.count_nonzero(region_mask))


def get_intensity_readout(img: np.ndarray, region_mask: np.ndarray, average: bool = True) -> float:
    area = get_area(region_mask)
    if area == 0:
        return 0.0
    total_intensity = float(np.sum(img * region_mask))
    return total_intensity / area if average else total_intensity


def get_density(img: np.ndarray, band: np.ndarray) -> float:
    return get_intensity_readout(img, band, average=True)


def summarize_region_metrics(img: np.ndarray, region_mask: np.ndarray) -> dict[str, float]:
    area = get_area(region_mask)
    total = get_intensity_readout(img, region_mask, average=False)
    average = total / area if area > 0 else 0.0
    return {
        "total": float(total),
        "area": int(area),
        "average": float(average),
    }


def lst_to_excel(lst: List[float], filename: str, column_name: str = "New_Column") -> None:
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    max_len = max(len(lst), len(df))
    if len(df) < max_len:
        extra = pd.DataFrame(index=range(len(df), max_len))
        df = pd.concat([df, extra], ignore_index=True)
    elif len(lst) < max_len:
        lst = lst + [np.nan] * (max_len - len(lst))

    df[column_name] = lst
    df.to_excel(filename, index=False)
    print(f"Wrote column '{column_name}' to Excel file: {filename}")


# -----------------------------
# Legacy batch pipeline (kept for compatibility)
# -----------------------------
def batch_analysis(
    img_lst: List[str],
    imgnum_lst: Optional[List[int]] = None,
    z_lst: Optional[List[int]] = None,
    t_lst: Optional[List[int]] = None,
    output_name: str = "processed.xlsx",
    model_type: str = "cyto3",
    diameter: Optional[float] = 200.0,
    band_dilation: int = 1,
    band_erosion: int = 7,
) -> None:
    if imgnum_lst is None:
        imgnum_lst = [1] * len(img_lst)
    if z_lst is None:
        z_lst = [0] * len(img_lst)
    if t_lst is None:
        t_lst = [0] * len(img_lst)

    assert len(img_lst) == len(imgnum_lst) == len(z_lst) == len(t_lst), (
        "img_lst, imgnum_lst, z_lst, and t_lst must have the same length"
    )

    model = models.Cellpose(gpu=False, model_type=model_type)
    channels = [0, 0]

    for i, lif_path in enumerate(img_lst):
        img = get_lif(lif_path, imgnum=imgnum_lst[i], z=z_lst[i], t=t_lst[i])
        masks, flows, styles, diams = model.eval([img], diameter=diameter, channels=[channels])
        masks = masks[0] if isinstance(masks, list) else masks
        masks = utils.remove_edge_masks(masks=masks)

        labels = np.unique(masks)
        labels = labels[labels != 0]

        density_lst: List[float] = []
        for lab in labels:
            single = (masks == lab).astype(np.uint8)
            band = create_band(single, dilation=band_dilation, erosion=band_erosion)
            density_lst.append(float(get_density(img, band)))

        lst_to_excel(density_lst, output_name, column_name=str(i + 1))

    print("Image Processing Done")


# -----------------------------
# General utilities
# -----------------------------
def ratio_cal(img1: np.ndarray, img2: np.ndarray, bit: int = 16, scale: bool = False) -> np.ndarray:
    ratio = np.divide(img1, img2, out=np.zeros_like(img1, dtype=float), where=(img2 != 0))

    if bit not in (8, 16, 32):
        raise ValueError("Only support bit depth 8/16/32")

    if bit == 8:
        target_dtype = np.uint8
        max_value = 255
    elif bit == 16:
        target_dtype = np.uint16
        max_value = 65535
    else:
        target_dtype = np.float32
        max_value = 1.0

    if scale:
        if bit == 32:
            out = (ratio / (ratio.max() if ratio.max() > 0 else 1.0)).astype(target_dtype)
        else:
            tmp = np.clip(ratio, 0, ratio.max() if ratio.max() > 0 else 1.0)
            out = ((tmp / (tmp.max() if tmp.max() > 0 else 1.0)) * max_value).astype(target_dtype)
    else:
        if bit == 32:
            out = np.clip(ratio, 0, ratio.max() if ratio.max() > 0 else 1.0).astype(target_dtype)
        else:
            out = np.clip(ratio, 0, max_value).astype(target_dtype)

    return out


def find_file(folder_path: str, pattern: str) -> List[str]:
    return glob.glob(os.path.join(folder_path, pattern))


# -----------------------------
# TIFF axis normalization
# -----------------------------
def _normalize_to_TZYCX(arr: np.ndarray, axes: Optional[str]) -> tuple[np.ndarray, str]:
    axes = (axes or "").upper().replace(" ", "")
    axes = axes.replace("S", "C")
    if not axes:
        axes = {2: "YX", 3: "PYX", 4: "PCYX", 5: "TZYXC"}.get(arr.ndim, "YX")

    axes = "".join(ch for ch in axes if ch in "TZYXC" or ch == "P")

    if "P" in axes:
        if "T" not in axes:
            axes = axes.replace("P", "T")
        elif "Z" not in axes:
            axes = axes.replace("P", "Z")
        else:
            axes = axes.replace("P", "T")

    target = list("TZYXC")
    out = arr
    current = list(axes)
    k = 0
    for ax in target:
        if ax in current:
            i = current.index(ax)
            if i != k:
                out = np.moveaxis(out, i, k)
                moved = current.pop(i)
                current.insert(k, moved)
            k += 1
        else:
            out = np.expand_dims(out, axis=k)
            current.insert(k, ax)
            k += 1

    return out, "TZYXC"


# -----------------------------
# Ratio helpers
# -----------------------------
def compute_ratio(
    region_metrics: dict[str, dict[str, float]],
    ratio_region_1: str,
    ratio_region_2: str,
    ratio_mode: str,
) -> tuple[float, str, str, str]:
    if ratio_mode == "off":
        return np.nan, "off", ratio_region_1, ratio_region_2

    metric_key = "average" if ratio_mode == "average" else "total"
    numerator = float(region_metrics[ratio_region_1][metric_key])
    denominator = float(region_metrics[ratio_region_2][metric_key])

    if denominator == 0:
        return np.nan, metric_key, ratio_region_1, ratio_region_2

    return float(numerator / denominator), metric_key, ratio_region_1, ratio_region_2


def _save_combined_band_overlay(seg_img: np.ndarray, combined_band: np.ndarray, out_png: str, dpi: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(seg_img, cmap="gray")
    masked = np.ma.masked_where(combined_band == 0, combined_band)
    ax.imshow(masked, cmap="viridis_r", alpha=0.5)
    ax.set_axis_off()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main analysis pipeline
# -----------------------------
def analyze_timelapse_zstack_to_excel(
    path: str,
    seg_channel: int = 0,
    measure_channel: int = 0,
    model_type: str = "cyto3",
    diameter: float | None = 50,
    band_dilation: int = 1,
    band_erosion: int = 1,
    output_name: str = "output",
    rb_radius: Optional[int] = None,
    save_seg: Literal["none", "repr", "all"] = "repr",
    save_band: Literal["none", "repr", "all"] = "repr",
    stride_t: int = 1,
    stride_z: int = 1,
    use_time: bool = True,
    use_z: bool = True,
    ratio_mode: Literal["off", "average", "total"] = "off",
    ratio_region_1: Literal["membrane", "outer_total", "inner_total"] = "membrane",
    ratio_region_2: Literal["membrane", "outer_total", "inner_total"] = "outer_total",
    **kwargs,
) -> pd.DataFrame:
    with tiff.TiffFile(path) as tif:
        series = tif.series[0]
        raw_axes = getattr(series, "axes", None)
        arr = series.asarray()

    arr_norm, _ = _normalize_to_TZYCX(arr, raw_axes)
    T, Z, Y, X, C = arr_norm.shape

    if not use_time:
        arr_norm = arr_norm[:1, ...]
        T = 1
    if not use_z:
        arr_norm = arr_norm[:, :1, ...]
        Z = 1

    def select_channel(frame: np.ndarray, ch: int) -> np.ndarray:
        if C == 1:
            if ch != 0:
                raise ValueError(f"Image has only 1 channel, but got ch={ch}")
            return frame[..., 0]
        if ch < 0 or ch >= C:
            raise ValueError(f"Specified channel={ch} exceeds available channels {C}")
        return frame[..., ch]

    model = models.Cellpose(gpu=False, model_type=model_type)
    channels = [0, 0]

    per_cell_rows = []
    best = {
        "signal": -np.inf,
        "t": None,
        "z": None,
        "img": None,
        "band": None,
        "combined_band": None,
        "masks": None,
        "flows": None,
    }

    for t_idx in range(T):
        for z_idx in range(Z):
            seg_img = select_channel(arr_norm[t_idx, z_idx], seg_channel).astype(np.float32)
            measure_img = select_channel(arr_norm[t_idx, z_idx], measure_channel).astype(np.float32)

            if rb_radius and rb_radius > 0:
                seg_img_raw = seg_img.copy()
                bg = rolling_ball(seg_img, radius=rb_radius)
                seg_img_rb = np.clip(seg_img - bg, 0, None)

                # debug save
                tiff.imwrite(f"{output_name}_debug_raw_t{t_idx:03d}_z{z_idx:03d}.tif", seg_img_raw.astype(np.float32))
                tiff.imwrite(f"{output_name}_debug_bg_t{t_idx:03d}_z{z_idx:03d}.tif", bg.astype(np.float32))
                tiff.imwrite(f"{output_name}_debug_rb_t{t_idx:03d}_z{z_idx:03d}.tif", seg_img_rb.astype(np.float32))

                seg_img = seg_img_rb

            masks, flows, _, _ = model.eval(seg_img, diameter=diameter, channels=channels)
            masks = utils.remove_edge_masks(masks=masks)

            labels = np.unique(masks)
            labels = labels[labels != 0]

            if labels.size == 0:
                per_cell_rows.append({
                    "t": t_idx,
                    "z": z_idx,
                    "cell_id": None,
                    "membrane_total": np.nan,
                    "membrane_area": np.nan,
                    "membrane_average": np.nan,
                    "outer_total_total": np.nan,
                    "outer_total_area": np.nan,
                    "outer_total_average": np.nan,
                    "inner_total_total": np.nan,
                    "inner_total_area": np.nan,
                    "inner_total_average": np.nan,
                    "ratio": np.nan,
                    "ratio_mode": ratio_mode,
                    "ratio_region_1": ratio_region_1,
                    "ratio_region_2": ratio_region_2,
                    "ratio_metric": "off" if ratio_mode == "off" else ("average" if ratio_mode == "average" else "total"),
                })
                continue

            signal_list = []
            band_list = []
            for lab in labels:
                single = (masks == lab).astype(np.uint8)
                regions = create_region_masks(single, dilation=band_dilation, erosion=band_erosion)

                region_metrics = {
                    "membrane": summarize_region_metrics(measure_img, regions["membrane"]),
                    "outer_total": summarize_region_metrics(measure_img, regions["outer_total"]),
                    "inner_total": summarize_region_metrics(measure_img, regions["inner_total"]),
                }

                ratio_value, ratio_metric, out_r1, out_r2 = compute_ratio(
                    region_metrics=region_metrics,
                    ratio_region_1=ratio_region_1,
                    ratio_region_2=ratio_region_2,
                    ratio_mode=ratio_mode,
                )

                per_cell_rows.append({
                    "t": t_idx,
                    "z": z_idx,
                    "cell_id": int(lab),
                    "membrane_total": region_metrics["membrane"]["total"],
                    "membrane_area": region_metrics["membrane"]["area"],
                    "membrane_average": region_metrics["membrane"]["average"],
                    "outer_total_total": region_metrics["outer_total"]["total"],
                    "outer_total_area": region_metrics["outer_total"]["area"],
                    "outer_total_average": region_metrics["outer_total"]["average"],
                    "inner_total_total": region_metrics["inner_total"]["total"],
                    "inner_total_area": region_metrics["inner_total"]["area"],
                    "inner_total_average": region_metrics["inner_total"]["average"],
                    "ratio": ratio_value,
                    "ratio_mode": ratio_mode,
                    "ratio_region_1": out_r1,
                    "ratio_region_2": out_r2,
                    "ratio_metric": ratio_metric,
                })

                signal_list.append(region_metrics["membrane"]["average"])
                band_list.append(regions["membrane"])

            combined_band = band_combined(masks, dilation=band_dilation, erosion=band_erosion)

            kmax = int(np.argmax(signal_list))
            if signal_list[kmax] > best["signal"]:
                best.update({
                    "signal": signal_list[kmax],
                    "t": t_idx,
                    "z": z_idx,
                    "img": seg_img,
                    "band": band_list[kmax],
                    "combined_band": combined_band,
                    "masks": masks,
                    "flows": flows,
                })

            if (t_idx % stride_t == 0) and (z_idx % stride_z == 0):
                if save_seg == "all":
                    fig = plt.figure(figsize=(8, 8))
                    plot.show_segmentation(fig, seg_img, masks, flows[0], channels=channels)
                    fig.savefig(f"{output_name}_seg_t{t_idx:03d}_z{z_idx:03d}.png", dpi=200, bbox_inches="tight")
                    plt.close(fig)
                if save_band == "all":
                    _save_combined_band_overlay(
                        seg_img,
                        combined_band,
                        f"{output_name}_band_t{t_idx:03d}_z{z_idx:03d}.png",
                        dpi=200,
                    )

    per_cell_df = pd.DataFrame(per_cell_rows)

    if per_cell_df.empty:
        per_cell_df = pd.DataFrame(columns=[
            "t", "z", "cell_id",
            "membrane_total", "membrane_area", "membrane_average",
            "outer_total_total", "outer_total_area", "outer_total_average",
            "inner_total_total", "inner_total_area", "inner_total_average",
            "ratio", "ratio_mode", "ratio_region_1", "ratio_region_2", "ratio_metric",
        ])

    summary_df = (
        per_cell_df.groupby(["t", "z"], dropna=False)
        .agg(
            n_cells=("cell_id", lambda s: int(s.notna().sum())),
            membrane_total_mean=("membrane_total", "mean"),
            membrane_area_mean=("membrane_area", "mean"),
            membrane_average_mean=("membrane_average", "mean"),
            outer_total_total_mean=("outer_total_total", "mean"),
            outer_total_area_mean=("outer_total_area", "mean"),
            outer_total_average_mean=("outer_total_average", "mean"),
            inner_total_total_mean=("inner_total_total", "mean"),
            inner_total_area_mean=("inner_total_area", "mean"),
            inner_total_average_mean=("inner_total_average", "mean"),
            ratio_mean=("ratio", "mean"),
            ratio_median=("ratio", "median"),
            ratio_std=("ratio", "std"),
        )
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product([range(T), range(Z)], names=["t", "z"])
    full_df = full_index.to_frame(index=False)
    summary_df = full_df.merge(summary_df, on=["t", "z"], how="left")
    summary_df["n_cells"] = summary_df["n_cells"].fillna(0).astype(int)
    summary_df = summary_df.sort_values(["t", "z"], ignore_index=True)

    excel_out = f"{output_name}.xlsx"
    try:
        with pd.ExcelWriter(excel_out, engine="xlsxwriter") as w:
            per_cell_df.to_excel(w, sheet_name="per_cell", index=False)
            summary_df.to_excel(w, sheet_name="summary", index=False)
    except Exception:
        with pd.ExcelWriter(excel_out, engine="openpyxl") as w:
            per_cell_df.to_excel(w, sheet_name="per_cell", index=False)
            summary_df.to_excel(w, sheet_name="summary", index=False)

    if best["img"] is not None:
        t_idx, z_idx = best["t"], best["z"]
        if save_seg in ("repr", "all"):
            fig = plt.figure(figsize=(8, 8))
            plot.show_segmentation(fig, best["img"], best["masks"], best["flows"][0], channels=[0, 0])
            fig.savefig(f"{output_name}_seg_repr_t{t_idx:03d}_z{z_idx:03d}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        if save_band in ("repr", "all"):
            _save_combined_band_overlay(
                best["img"],
                best["combined_band"],
                f"{output_name}_band_repr_t{t_idx:03d}_z{z_idx:03d}.png",
                dpi=300,
            )

    print(f"[OK] Wrote Excel to: {excel_out}")
    return per_cell_df
