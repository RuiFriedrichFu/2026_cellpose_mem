
from __future__ import annotations

import sys
import time
import logging
from datetime import datetime
from typing import Optional

import typer
from pathlib import Path

from .core import analyze_timelapse_zstack_to_excel

app = typer.Typer(help="Cellpose-based utilities")


def setup_logger(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("cellpose_tools")
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logging.captureWarnings(True)
    return logger


class _PrintTee:
    def __init__(self, logger: logging.Logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, msg: str):
        msg = msg.rstrip()
        if msg:
            self.logger.log(self.level, msg)

    def flush(self):
        pass


@app.command()
def one(
    path: Path = typer.Option(..., "--path", "-p", help="Image path (TIFF)"),
    seg_channel: int = typer.Option(0, "--seg-channel", help="Channel used for segmentation"),
    measure_channel: int = typer.Option(0, "--measure-channel", help="Channel used for intensity measurement"),
    model: str = typer.Option("cyto3", "--model", help="Cellpose model type"),
    diameter: float = typer.Option(50, "--diameter", help="Estimated cell diameter (px)"),
    band_dilation: int = typer.Option(1, "--band-dilation", "--band_dilation", help="Outer dilation (px)"),
    band_erosion: int = typer.Option(1, "--band-erosion", "--band_erosion", help="Inner erosion (px)"),
    output_name: Path = typer.Option("output", "--output-name", "--output_name", "-o",
                                     help="Output basename (no extension)"),
    rb_radius: Optional[int] = typer.Option(
        None, "--rb-radius", "--rb_radius",
        help="Rolling-ball radius (px); omit or 0/None to disable"
    ),
    use_time: bool = typer.Option(True, "--use-time/--no-use-time",
                                  help="Keep/analyze the time (T) axis (default: on)"),
    use_z: bool = typer.Option(True, "--use-z/--no-use-z",
                               help="Keep/analyze the z-stack (Z) axis (default: on)"),
    save_seg: str = typer.Option("repr", "--save-seg",
                                 help='Save segmentation overlays: "none", "repr", or "all" (default: "repr")'),
    save_band: str = typer.Option("repr", "--save-band",
                                  help='Save membrane-band overlays: "none", "repr", or "all" (default: "repr")'),
    stride_t: int = typer.Option(1, "--stride-t", help="Save every N-th time frame (default: 1)"),
    stride_z: int = typer.Option(1, "--stride-z", help="Save every N-th z slice (default: 1)"),
    ratio_mode: str = typer.Option(
        "off", "--ratio-mode",
        help='Ratio mode: "off", "average", or "total"'
    ),
    ratio_region_1: str = typer.Option(
        "membrane", "--ratio-region-1",
        help='Ratio numerator region: "membrane", "outer_total", or "inner_total"'
    ),
    ratio_region_2: str = typer.Option(
        "outer_total", "--ratio-region-2",
        help='Ratio denominator region: "membrane", "outer_total", or "inner_total"'
    ),
    job_name: Optional[str] = typer.Option(
        None, "--job-name", "-j",
        help="Human-readable task name (defaults to output_name)"
    ),
    log: bool = typer.Option(True, "--log/--no-log", help="Write a log file (default: on)"),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", "-L",
        help="Log file path; defaults to <output_name>.log when --log is on"
    ),
):
    """
    Process a TIFF (optionally timelapse + z-stack), optionally apply rolling-ball
    background subtraction, run Cellpose, and write results to Excel.
    """
    if not path.exists():
        typer.secho(f"Error: File does not exist -> {path}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    valid_ratio_modes = {"off", "average", "total"}
    valid_ratio_regions = {"membrane", "outer_total", "inner_total"}

    if ratio_mode not in valid_ratio_modes:
        typer.secho(f"Error: --ratio-mode must be one of {sorted(valid_ratio_modes)}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if ratio_region_1 not in valid_ratio_regions:
        typer.secho(f"Error: --ratio-region-1 must be one of {sorted(valid_ratio_regions)}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if ratio_region_2 not in valid_ratio_regions:
        typer.secho(f"Error: --ratio-region-2 must be one of {sorted(valid_ratio_regions)}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    job_name = job_name or output_name.stem

    resolved_log_file: Optional[Path] = None
    if log:
        resolved_log_file = log_file if log_file else output_name.with_suffix(".log")

    logger = setup_logger(resolved_log_file)

    sys_stdout_orig, sys_stderr_orig = sys.stdout, sys.stderr
    sys.stdout = _PrintTee(logger, logging.INFO)
    sys.stderr = _PrintTee(logger, logging.ERROR)

    start_ts = time.time()
    start_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        logger.info("===== Job started =====")
        logger.info(f"Job name     : {job_name}")
        logger.info(f"Start time   : {start_iso}")
        logger.info(f"Input image  : {path}")
        logger.info(f"Output base  : {output_name}")
        logger.info(
            "Params       : "
            f"seg_channel={seg_channel}, measure_channel={measure_channel}, model={model}, diameter={diameter}, "
            f"band_dilation={band_dilation}, band_erosion={band_erosion}, rb_radius={rb_radius}, "
            f"use_time={use_time}, use_z={use_z}, save_seg={save_seg}, save_band={save_band}, "
            f"ratio_mode={ratio_mode}, ratio_region_1={ratio_region_1}, ratio_region_2={ratio_region_2}"
        )
        if resolved_log_file:
            logger.info(f"Log file     : {resolved_log_file}")

        per_cell_df = analyze_timelapse_zstack_to_excel(
            path=str(path),
            seg_channel=seg_channel,
            measure_channel=measure_channel,
            model_type=model,
            diameter=diameter,
            band_dilation=band_dilation,
            band_erosion=band_erosion,
            output_name=str(output_name),
            rb_radius=rb_radius,
            save_seg=save_seg,
            save_band=save_band,
            stride_t=stride_t,
            stride_z=stride_z,
            use_time=use_time,
            use_z=use_z,
            ratio_mode=ratio_mode,
            ratio_region_1=ratio_region_1,
            ratio_region_2=ratio_region_2,
        )

        excel_file = f"{output_name}.xlsx"
        n_objects = int(per_cell_df["cell_id"].notna().sum())

        logger.info(f"Outputs      : {excel_file} (+ PNG overlays if enabled)")
        logger.info(f"Objects      : {n_objects} instances written")
        typer.secho(f"Done. Wrote: {excel_file}  |  Cells: {n_objects}", fg=typer.colors.GREEN)

    except Exception:
        logger.exception("Job failed with an unhandled exception")
        raise
    finally:
        elapsed = time.time() - start_ts
        logger.info(f"Elapsed time : {elapsed:.2f} s")
        logger.info("===== Job finished =====")
        sys.stdout, sys.stderr = sys_stdout_orig, sys_stderr_orig


def main():
    app()
