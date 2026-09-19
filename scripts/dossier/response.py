"""Conditional, descriptive RNA response statistics with explicit control strata.

Cells are resampling units, never automatically biological replicates. Methods
were registered in Issue #4 before inspecting these new response results.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def grouped_moments(matrix, batches, batch_names):
    groups = []
    for batch in batch_names:
        ids = np.flatnonzero(batches == batch)
        x = matrix[ids]
        groups.append({"batch": batch, "n": len(ids), "indices": ids,
                       "mean": x.mean(axis=0, dtype=np.float64) if len(ids) else np.zeros(matrix.shape[1]),
                       "variance": x.var(axis=0, ddof=1, dtype=np.float64) if len(ids) > 1 else np.full(matrix.shape[1], np.nan),
                       "detected": (x > 0).sum(axis=0)})
    return groups


def matched_effect(target_groups, control_groups):
    eligible = [(t, c) for t, c in zip(target_groups, control_groups) if t["n"] and c["n"]]
    if not eligible:
        return None, {"status": "not_estimable", "reason": "no_same_batch_control", "target_cells_used": 0}
    n = sum(t["n"] for t, _ in eligible)
    effect = sum((t["mean"] - c["mean"]) * (t["n"] / n) for t, c in eligible)
    return effect, {"status": "completed", "target_cells_used": n,
                    "target_cells_unmatched": sum(t["n"] for t in target_groups) - n,
                    "control_cells": sum(c["n"] for _, c in eligible), "batches": len(eligible)}


def conditional_de(target_groups, control_groups, min_cells=20, min_detection=10):
    pairs = [(t, c) for t, c in zip(target_groups, control_groups) if t["n"] >= 2 and c["n"] >= 2]
    n = sum(t["n"] for t, _ in pairs)
    meta = {"unit": "observed_cells_conditional_on_sample", "independent_biological_replicates": None,
            "target_cells_used": n, "target_cells_not_used": sum(t["n"] for t in target_groups) - n,
            "control_cells_used": sum(c["n"] for _, c in pairs), "batches_used": len(pairs)}
    if n < min_cells:
        return None, dict(meta, status="not_estimable", reason="fewer_than_20_target_cells_in_variance_estimable_batches")
    effect = sum((t["mean"] - c["mean"]) * (t["n"] / n) for t, c in pairs)
    parts = [(t["variance"] / t["n"] * (t["n"] / n) ** 2, t["n"] - 1) for t, c in pairs]
    parts += [(c["variance"] / c["n"] * (t["n"] / n) ** 2, c["n"] - 1) for t, c in pairs]
    variance = np.maximum(sum(v for v, _ in parts), 0)
    denominator = sum(v * v / df for v, df in parts)
    df = np.divide(variance * variance, denominator, out=np.full_like(variance, np.inf), where=denominator > 0)
    se = np.sqrt(variance)
    test_stat = np.divide(effect, se, out=np.zeros_like(effect), where=se > 0)
    p = 2 * stats.t.sf(np.abs(test_stat), df)
    p[(se == 0) & (effect != 0)] = 0
    p[(se == 0) & (effect == 0)] = 1
    detected_t = sum(t["detected"] for t, _ in pairs)
    detected_c = sum(c["detected"] for _, c in pairs)
    tested = np.maximum(detected_t, detected_c) >= min_detection
    q = np.full_like(p, np.nan)
    q_by = np.full_like(p, np.nan)
    if tested.any():
        q[tested] = stats.false_discovery_control(p[tested], method="bh")
        q_by[tested] = stats.false_discovery_control(p[tested], method="by")
    half_width = stats.t.ppf(0.975, df) * se
    return {"effect_de_subset": effect, "se": se, "df": df, "p": np.where(tested, p, np.nan),
            "q_bh": q, "q_by": q_by, "ci95_low": effect - half_width, "ci95_high": effect + half_width,
            "tested": tested}, dict(meta, status="completed", tested_genes=int(tested.sum()),
                                     interpretation="Approximate cell-sampling association; no between-experiment inference; BH requires dependence assumptions; BY is sensitivity only")


def disjoint_control_indices(target_batches, control_batches, batch_names, rng, control_pools=None):
    """Two non-overlapping NTC samples matched to target counts within each batch."""
    a, b = [], []
    for batch in batch_names:
        n = int((target_batches == batch).sum())
        pool = control_pools[batch] if control_pools is not None else np.flatnonzero(control_batches == batch)
        k = min(n, len(pool) // 2)
        if k:
            chosen = rng.choice(pool, 2 * k, replace=False)
            a.extend(chosen[:k])
            b.extend(chosen[k:])
    return np.asarray(a, dtype=int), np.asarray(b, dtype=int)


def control_half_means(control, control_batches, batch_names, reps, seed):
    rng = np.random.default_rng(seed)
    means = np.empty((reps, 2, len(batch_names), control.shape[1]), dtype=np.float32)
    for r in range(reps):
        for b, name in enumerate(batch_names):
            ids = rng.permutation(np.flatnonzero(control_batches == name))
            for h, part in enumerate(np.array_split(ids, 2)):
                means[r, h, b] = control[part].mean(axis=0, dtype=np.float64) if len(part) else np.nan
    return means


def resampling(target, target_batches, control, control_batches, batch_names,
               half_controls, downstream, seed, reps=20, control_pools=None):
    rng = np.random.default_rng(seed)
    rows = []
    if len(target) < 4:
        return [], {"status": "not_estimable", "reason": "fewer_than_four_target_cells"}
    if control_pools is None:
        control_pools = {batch: np.flatnonzero(control_batches == batch) for batch in batch_names}
    for r in range(reps):
        halves = np.array_split(rng.permutation(len(target)), 2)
        effects = []
        for h, ids in enumerate(halves):
            weights = np.array([(target_batches[ids] == batch).sum() for batch in batch_names], dtype=float)
            weights /= weights.sum()
            used = weights > 0
            if not np.isfinite(half_controls[r, h, used]).all():
                return rows, {"status": "not_estimable", "reason": "empty_control_half_in_matched_batch"}
            baseline = weights[used] @ half_controls[r, h, used].astype(float)
            effects.append(target[ids].mean(axis=0, dtype=np.float64) - baseline)
        a, b = disjoint_control_indices(target_batches, control_batches, batch_names, rng, control_pools)
        null = control[a].mean(axis=0, dtype=np.float64) - control[b].mean(axis=0, dtype=np.float64) if len(a) else None
        rows.append({"replicate": r, "half_1_cells": len(halves[0]), "half_2_cells": len(halves[1]),
                     "half_effect_correlation": correlation(effects[0][downstream], effects[1][downstream]),
                     "half_1_rms": float(np.sqrt(np.mean(effects[0][downstream] ** 2))),
                     "half_2_rms": float(np.sqrt(np.mean(effects[1][downstream] ** 2))),
                     "null_arm_cells": len(a), "null_target_cells_not_matched": len(target) - len(a),
                     "null_rms": float(np.sqrt(np.mean(null[downstream] ** 2))) if null is not None else None,
                     "null_reference_intersection": len(np.intersect1d(a, b)),
                     "interpretation": "Within-sample split stability and NTC variation, not biological replication"})
    return rows, {"status": "completed", "repetitions": reps, "unit": "cells", "confidence_interval": False}


def consistency(target, target_batches, guide_ids, batch_names, controls, downstream, min_cells=10):
    batch_effects, guide_effects, small_batches, small_guides = [], [], [], []
    for b, batch in enumerate(batch_names):
        ids = np.flatnonzero(target_batches == batch)
        if len(ids) >= min_cells and controls[b]["n"]:
            batch_effects.append((target[ids].mean(axis=0, dtype=np.float64) - controls[b]["mean"])[downstream])
        elif len(ids):
            small_batches.append(str(batch))
    for guide in np.unique(guide_ids):
        ids = np.flatnonzero(guide_ids == guide)
        if len(ids) < min_cells:
            small_guides.append(str(guide))
            continue
        groups = grouped_moments(target[ids], target_batches[ids], batch_names)
        effect, meta = matched_effect(groups, controls)
        if effect is not None:
            guide_effects.append(effect[downstream])
    def summary(vectors, small, reason):
        if len(vectors) < 2:
            return {"status": "not_estimable", "reason": reason, "usable_groups": len(vectors), "small_groups": small}
        values = [correlation(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i)]
        valid = [v for v in values if v is not None]
        return {"status": "completed" if valid else "not_estimable", "reason": None if valid else "constant_effect_vectors",
                "usable_groups": len(vectors), "small_groups": small, "pairs": len(valid),
                "median_correlation": float(np.median(valid)) if valid else None,
                "minimum_correlation": float(np.min(valid)) if valid else None,
                "independent_biological_replicates": False}
    return {"batch": summary(batch_effects, small_batches, "fewer_than_two_batches_with_10_target_cells"),
            "guide": summary(guide_effects, small_guides, "fewer_than_two_constructs_with_10_target_cells")}
