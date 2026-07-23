"""
metrics.py — RFG, nRFG, and robustness analysis.
Implements Eq. (2) RFG, Eq. (3) nRFG, and the 9-variant robustness matrix.
"""
import numpy as np
from itertools import product
from scipy.stats import spearmanr


def rfg(retrieval_quality: float, faithfulness: float) -> float:
    """
    Retrieval-Faithfulness Gap (Eq. 2).
    RFG(E) = RetrievalQuality(E) - Faithfulness(E)
    Both terms assumed normalized to [0, 1].
    """
    return round(max(0.0, retrieval_quality) - max(0.0, faithfulness), 4)


def nrfg(retrieval_quality: float, faithfulness: float) -> float:
    """
    Normalized RFG (Eq. 3) — addresses both-high vs both-low insensitivity.
    nRFG(E) = (RetrievalQuality - Faithfulness) / RetrievalQuality
    Distinguishes (0.9, 0.85) -> 0.056 from (0.3, 0.25) -> 0.167.
    """
    rq = max(1e-9, retrieval_quality)  # avoid div-by-zero
    return round((rq - max(0.0, faithfulness)) / rq, 4)


def compute_rfg_variants(per_model_scores: dict,
                         retrieval_metrics: list,
                         faithfulness_metrics: list) -> dict:
    """
    Compute all RFG variants for the robustness analysis (Berend Point 3).

    per_model_scores: {
        model_name: {
            'ndcg@5': v, 'recall@5': v, 'mrr@5': v,
            'alignscore': v, 'nli': v, 'mean': v
        }, ...
    }

    Returns: {
        (ret_metric, faith_metric): {model_name: rfg_value, ...}, ...
    }
    """
    variants = {}
    for ret_m, faith_m in product(retrieval_metrics, faithfulness_metrics):
        ranking = {}
        for model, scores in per_model_scores.items():
            if ret_m in scores and faith_m in scores:
                ranking[model] = rfg(scores[ret_m], scores[faith_m])
        variants[(ret_m, faith_m)] = ranking
    return variants


def robustness_correlation_matrix(variants: dict):
    """
    Compute the Spearman rank correlation between all pairs of RFG variants.
    A stable metric -> high off-diagonal correlations.

    Returns: (labels, correlation_matrix as np.ndarray)
    """
    labels = list(variants.keys())
    n = len(labels)

    # Fixed model ordering so ranks are comparable across variants
    all_models = sorted({m for v in variants.values() for m in v})

    # Build a rank vector per variant
    rank_vectors = []
    for lab in labels:
        scores = variants[lab]
        vec = [scores.get(m, np.nan) for m in all_models]
        rank_vectors.append(vec)
    rank_vectors = np.array(rank_vectors)

    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xi, xj = rank_vectors[i], rank_vectors[j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() >= 3:
                rho, _ = spearmanr(xi[mask], xj[mask])
                corr[i, j] = rho
    return labels, corr


def summarize_robustness(labels, corr) -> str:
    """Human-readable summary of the robustness matrix."""
    n = len(labels)
    off_diag = [corr[i, j] for i in range(n) for j in range(n)
                if i != j and not np.isnan(corr[i, j])]
    if not off_diag:
        return "Insufficient data for robustness summary."
    mean_rho = np.mean(off_diag)
    min_rho = np.min(off_diag)
    verdict = ("ROBUST — rankings stable across measurement choices"
               if mean_rho > 0.7 else
               "MODERATE — some sensitivity to measurement choice"
               if mean_rho > 0.4 else
               "VOLATILE — metric depends heavily on measurement choice")
    return (f"RFG robustness across {n} variants:\n"
            f"  Mean pairwise Spearman rho: {mean_rho:.3f}\n"
            f"  Min pairwise Spearman rho:  {min_rho:.3f}\n"
            f"  Verdict: {verdict}")


def label_str(label_tuple) -> str:
    """Format a (ret, faith) label tuple for display."""
    return f"{label_tuple[0]}/{label_tuple[1]}"


if __name__ == '__main__':
    # Self-test with Berend's example
    print("=== Berend's insensitivity example ===")
    print(f"Both-high (0.90, 0.85): RFG={rfg(0.90,0.85)}  nRFG={nrfg(0.90,0.85)}")
    print(f"Both-low  (0.30, 0.25): RFG={rfg(0.30,0.25)}  nRFG={nrfg(0.30,0.25)}")
    print("  -> RFG identical, nRFG distinguishes them. ✓\n")

    # Robustness self-test with dummy data
    print("=== Robustness matrix self-test ===")
    dummy = {
        'modelA': {'ndcg@5':0.9,'recall@5':0.88,'mrr@5':0.91,'alignscore':0.7,'nli':0.72,'mean':0.71},
        'modelB': {'ndcg@5':0.85,'recall@5':0.83,'mrr@5':0.86,'alignscore':0.8,'nli':0.79,'mean':0.795},
        'modelC': {'ndcg@5':0.7,'recall@5':0.72,'mrr@5':0.69,'alignscore':0.65,'nli':0.63,'mean':0.64},
    }
    variants = compute_rfg_variants(dummy, ['ndcg@5','recall@5','mrr@5'],
                                    ['alignscore','nli','mean'])
    print(f"Computed {len(variants)} RFG variants")
    labels, corr = robustness_correlation_matrix(variants)
    print(summarize_robustness(labels, corr))
