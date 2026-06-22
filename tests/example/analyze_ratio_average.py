import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction without statsmodels."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def load_ratio_from_excel(xlsx_path: Path, sheet_name: str = "per_cell", ratio_mode_filter: str = "any") -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if "ratio" not in df.columns:
        raise ValueError(f"{xlsx_path} has no 'ratio' column in sheet '{sheet_name}'")

    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    df = df[np.isfinite(df["ratio"])].copy()

    if ratio_mode_filter != "any":
        mode_col = None
        if "ratio_mode" in df.columns:
            mode_col = "ratio_mode"
        elif "ratio_metric" in df.columns:
            mode_col = "ratio_metric"

        if mode_col is None:
            raise ValueError(f"{xlsx_path} has no ratio_mode/ratio_metric column, cannot filter '{ratio_mode_filter}'")

        df = df[df[mode_col].astype(str).str.lower() == ratio_mode_filter.lower()].copy()

    if df.empty:
        raise ValueError(f"{xlsx_path} has no usable ratio values after filtering")

    df["source_file"] = xlsx_path.name
    df["source_path"] = str(xlsx_path)
    return df


def infer_group_name(xlsx_path: Path, group_mode: str = "parent") -> str:
    if group_mode == "parent":
        return xlsx_path.parent.name
    elif group_mode == "prefix":
        return xlsx_path.stem.split("_")[0]
    else:
        raise ValueError("group_mode must be 'parent' or 'prefix'")


def collect_data(input_dir: Path, pattern: str = "*.xlsx", group_mode: str = "parent", ratio_mode_filter: str = "any") -> pd.DataFrame:
    rows = []
    files = sorted(input_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {input_dir} matching {pattern}")

    for f in files:
        try:
            df = load_ratio_from_excel(f, ratio_mode_filter=ratio_mode_filter)
            group = infer_group_name(f, group_mode=group_mode)
            df["group"] = group
            rows.append(df[["group", "ratio", "source_file", "source_path"]])
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")

    if not rows:
        raise ValueError("No valid ratio data found in any Excel files.")

    return pd.concat(rows, ignore_index=True)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("group")["ratio"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["n"])
    return out


def run_global_test(df: pd.DataFrame):
    groups = [g["ratio"].values for _, g in df.groupby("group")]
    if len(groups) < 2:
        return None, None
    stat, p = stats.kruskal(*groups)
    return stat, p


def run_pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    group_names = sorted(df["group"].unique())
    results = []

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = group_names[i]
            g2 = group_names[j]
            x = df.loc[df["group"] == g1, "ratio"].values
            y = df.loc[df["group"] == g2, "ratio"].values

            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            results.append({
                "group1": g1,
                "group2": g2,
                "n1": len(x),
                "n2": len(y),
                "median1": np.median(x),
                "median2": np.median(y),
                "mean1": np.mean(x),
                "mean2": np.mean(y),
                "mannwhitney_u": stat,
                "p_raw": p,
            })

    out = pd.DataFrame(results)
    if not out.empty:
        out["p_fdr_bh"] = bh_fdr(out["p_raw"].values)
        out["significant_fdr_0.05"] = out["p_fdr_bh"] < 0.05
    return out


def plot_box(df: pd.DataFrame, out_png: Path):
    groups = sorted(df["group"].unique())
    data = [df.loc[df["group"] == g, "ratio"].values for g in groups]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=groups, showfliers=False)

    rng = np.random.default_rng(42)
    for i, g in enumerate(groups, start=1):
        y = df.loc[df["group"] == g, "ratio"].values
        x = rng.normal(i, 0.06, size=len(y))
        plt.scatter(x, y, alpha=0.6, s=18)

    plt.ylabel("Ratio")
    plt.xlabel("Group")
    plt.title("Ratio comparison across groups")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_violin(df: pd.DataFrame, out_png: Path):
    groups = sorted(df["group"].unique())
    data = [df.loc[df["group"] == g, "ratio"].values for g in groups]

    plt.figure(figsize=(8, 5))
    plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    rng = np.random.default_rng(42)
    for i, g in enumerate(groups, start=1):
        y = df.loc[df["group"] == g, "ratio"].values
        x = rng.normal(i, 0.06, size=len(y))
        plt.scatter(x, y, alpha=0.5, s=15)

    plt.xticks(range(1, len(groups) + 1), groups)
    plt.ylabel("Ratio")
    plt.xlabel("Group")
    plt.title("Ratio distribution across groups")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze ratio from exported cellpose Excel tables")
    parser.add_argument("--input", required=True, help="Root folder containing Excel files")
    parser.add_argument("--pattern", default="*.xlsx", help="File glob pattern")
    parser.add_argument("--group-mode", choices=["parent", "prefix"], default="parent")
    parser.add_argument("--ratio-mode", choices=["any", "average", "total"], default="any",
                        help="Filter ratio mode; default is any non-null ratio")
    parser.add_argument("--output", default="ratio_analysis_output", help="Output folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_data(
        input_dir,
        pattern=args.pattern,
        group_mode=args.group_mode,
        ratio_mode_filter=args.ratio_mode,
    )
    df.to_csv(out_dir / "all_ratio_values.csv", index=False)

    summary = compute_summary_stats(df)
    summary.to_csv(out_dir / "summary_stats.csv", index=False)

    global_stat, global_p = run_global_test(df)
    pairwise = run_pairwise_tests(df)
    pairwise.to_csv(out_dir / "pairwise_stats.csv", index=False)

    plot_box(df, out_dir / "ratio_boxplot.png")
    plot_violin(df, out_dir / "ratio_violin.png")

    print("\n=== Summary statistics ===")
    print(summary.to_string(index=False))

    if global_p is not None:
        print("\n=== Global Kruskal-Wallis test ===")
        print(f"H = {global_stat:.4f}, p = {global_p:.6g}")

    if not pairwise.empty:
        print("\n=== Pairwise Mann-Whitney U tests ===")
        print(pairwise.to_string(index=False))

    print(f"\n[OK] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()