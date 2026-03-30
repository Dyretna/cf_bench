from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def save_heatmap(
    df_matrix: pd.DataFrame, out_path: Path, title: str, diverging: bool = False
) -> None:
    """
    Render and save a heatmap from a numeric DataFrame using matplotlib only.
    Supports annotation and diverging colormap for delta matrices.

    Parameters
    ----------
    df_matrix : pd.DataFrame
        The matrix to visualize. Non-numeric values are coerced to NaN.
    out_path : Path
        File path where the heatmap image will be saved.
    title : str
        Title displayed on the heatmap figure.
    diverging : bool
        If True, use a diverging colormap centered at zero (for deltas).
    """
    numeric = df_matrix.apply(pd.to_numeric, errors="coerce")

    # Choose colormap
    if diverging:
        cmap = plt.cm.coolwarm
        vmin = -max(abs(numeric.min().min()), abs(numeric.max().max()))
        vmax = -vmin
    else:
        cmap = plt.cm.viridis
        vmin = numeric.min().min()
        vmax = numeric.max().max()

    fig, ax = plt.subplots(
        figsize=(max(10, 0.6 * numeric.shape[1]), max(4, 0.6 * numeric.shape[0]))
    )

    im = ax.imshow(numeric.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)

    ax.set_yticks(range(numeric.shape[0]))
    ax.set_yticklabels(df_matrix.index.tolist())

    ax.set_xticks(range(numeric.shape[1]))
    ax.set_xticklabels(df_matrix.columns.tolist(), rotation=90)

    # Annotate each cell with its value
    for i in range(numeric.shape[0]):
        for j in range(numeric.shape[1]):
            val = numeric.iat[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(val) > (vmax * 0.5) else "black",
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_cf_heatmaps(
    query_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    prefix: str,
    normalize: bool = False,
) -> Tuple[Path, Path]:
    """
    Build and save two heatmaps: one for raw CF values and one for deltas.
    Optionally normalizes each feature column to [0, 1] for visualization.
    """
    matrix = pd.concat(
        [
            query_df[feature_cols].assign(_label="Original"),
            cf_df[feature_cols].assign(_label=[f"CF{i}" for i in range(len(cf_df))]),
        ],
        ignore_index=True,
    ).set_index("_label")

    orig_vals = pd.to_numeric(query_df[feature_cols].iloc[0], errors="coerce")
    deltas = (
        cf_df[feature_cols].apply(pd.to_numeric, errors="coerce").sub(orig_vals, axis=1)
    )
    deltas.index = [f"CF{i}" for i in range(len(cf_df))]

    if normalize:
        # matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        deltas = (deltas - deltas.min()) / (deltas.max() - deltas.min())

    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = out_dir / f"{prefix}_heatmap.png"
    delta_path = out_dir / f"{prefix}_delta_heatmap.png"

    # original
    save_heatmap(matrix, heatmap_path, f"{prefix} - values", diverging=False)
    # deltas
    save_heatmap(deltas, delta_path, f"{prefix} - deltas (normalized)", diverging=True)

    return heatmap_path, delta_path
