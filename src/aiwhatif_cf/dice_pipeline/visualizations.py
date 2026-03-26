import matplotlib.pyplot as plt
import pandas as pd


def create_heatmap(df, title=None):
    numeric = df.apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(
        figsize=(max(10, 0.6 * numeric.shape[1]), max(4, 0.6 * numeric.shape[0]))
    )
    im = ax.imshow(numeric.values, aspect="auto")

    if title:
        ax.set_title(title)

    ax.set_yticks(range(numeric.shape[0]))
    ax.set_yticklabels(df.index.tolist())
    ax.set_xticks(range(numeric.shape[1]))
    ax.set_xticklabels(df.columns.tolist(), rotation=90)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig
