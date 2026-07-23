"""
figures.py — Phase H: paper figures + exports, from REAL results only.

Fig. 1  Retrieval quality vs faithfulness scatter (diagonal = RFG 0)
Fig. 2  RFG heatmap, model × dataset
Fig. 3  nRFG by training paradigm (box plot)
Fig. 4  Robustness matrix: Spearman ρ between the 9 RFG variants (§5.3)

The notebook's figure cells consumed an undefined SIMULATED_RESULTS
(audit B2); these functions require the assembled DataFrame and fail
loudly if the pipeline hasn't produced one.
"""
import matplotlib

matplotlib.use('Agg')  # headless server
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import load_checkpoint

PARADIGM_COLORS = {
    'contrastive': '#E74C3C',
    'multilingual': '#8E44AD',
    'instruction-tuned': '#27AE60',
    'distilled': '#2980B9',
}
DATASET_MARKERS = {'NQ': 'o', 'HotpotQA': 's', 'QASPER': '^'}
SHORT_NAMES = {'all-mpnet-base-v2': 'SBERT',
               'text-embedding-3-small': 'OpenAI-3-small'}


def _gpt_rows(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df['generator'] == 'gpt4o']
    if sub.empty:
        raise RuntimeError('No gpt4o rows in results — run the pipeline first.')
    return sub


def plot_rfg_scatter(df: pd.DataFrame, save_path: str):
    """Fig. 1 — each point one model × dataset; below diagonal = high RFG."""
    sub = _gpt_rows(df)
    fig, ax = plt.subplots(figsize=(9, 7))
    lo = min(sub['NDCG@5'].min(), sub['faithfulness'].min()) - 0.05
    hi = max(sub['NDCG@5'].max(), sub['faithfulness'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, linewidth=1.2,
            label='RFG = 0')
    for _, row in sub.iterrows():
        ax.scatter(row['NDCG@5'], row['faithfulness'],
                   color=PARADIGM_COLORS.get(row['paradigm'], '#555'),
                   marker=DATASET_MARKERS.get(row['dataset'], 'o'),
                   s=120, zorder=3, edgecolors='white', linewidth=0.8,
                   alpha=0.9)
        ax.annotate(SHORT_NAMES.get(row['model'], row['model']),
                    (row['NDCG@5'], row['faithfulness']),
                    textcoords='offset points', xytext=(6, 3),
                    fontsize=7.5, alpha=0.85)
    handles = [mpatches.Patch(color=c, label=p.title())
               for p, c in PARADIGM_COLORS.items()
               if p in set(sub['paradigm'])]
    handles += [mlines.Line2D([0], [0], marker=m, color='gray',
                              linestyle='None', markersize=8, label=d)
                for d, m in DATASET_MARKERS.items()
                if d in set(sub['dataset'])]
    ax.legend(handles=handles, title='Paradigm / Dataset', fontsize=9,
              loc='upper left', framealpha=0.9)
    ax.set_xlabel('Retrieval Quality (NDCG@5)', fontsize=12)
    ax.set_ylabel('Faithfulness', fontsize=12)
    ax.set_title('Retrieval Quality vs Faithfulness across Embedding Models',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {save_path}')


def plot_rfg_heatmap(df: pd.DataFrame, save_path: str, metric: str = 'nRFG'):
    """Fig. 2 — model × dataset heatmap (default: primary metric nRFG)."""
    sub = _gpt_rows(df)
    pivot = sub.pivot_table(index='model', columns='dataset', values=metric)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)),
                  [SHORT_NAMES.get(m, m) for m in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=9)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f'{metric} by Model and Dataset (lower = better)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {save_path}')


def plot_paradigm_rfg(df: pd.DataFrame, save_path: str, metric: str = 'nRFG'):
    """Fig. 3 — H1 test: metric distribution by training paradigm."""
    sub = _gpt_rows(df)
    paradigms = [p for p in PARADIGM_COLORS if p in set(sub['paradigm'])]
    data = [sub[sub['paradigm'] == p][metric].dropna().values
            for p in paradigms]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=[p.title() for p in paradigms],
                    patch_artist=True)
    for patch, p in zip(bp['boxes'], paradigms):
        patch.set_facecolor(PARADIGM_COLORS[p])
        patch.set_alpha(0.6)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} by Training Paradigm', fontsize=12,
                 fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {save_path}')


def plot_robustness_matrix(save_path: str):
    """Fig. 4 — §5.3: Spearman ρ between all RFG variant rankings."""
    stored = load_checkpoint('robustness_matrix')
    if not stored:
        print('robustness_matrix checkpoint missing — run results.robustness_analysis')
        return
    labels, corr = stored
    names = [f'{r}/{f}' for r, f in labels]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha='right',
                  fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(corr[i, j]):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if corr[i, j] < 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='Spearman ρ')
    ax.set_title('RFG Ranking Stability across Metric Choices (§5.3)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {save_path}')


def export_all(df: pd.DataFrame):
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    plot_rfg_scatter(df, str(out / 'fig1_rfg_scatter.pdf'))
    plot_rfg_heatmap(df, str(out / 'fig2_rfg_heatmap.pdf'))
    plot_paradigm_rfg(df, str(out / 'fig3_paradigm_rfg.pdf'))
    plot_robustness_matrix(str(out / 'fig4_robustness_matrix.pdf'))
    df.to_csv(out / 'full_results.csv', index=False)
    print(f'exports written to {out}')
