"""Conservative, uncalibrated marker inference; never source truth labels."""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Analyst-defined broad hypotheses, not a cell ontology or an exhaustive atlas.
PROFILES = {
    "pluripotent_like": ("pluripotent", ["Embryonic stem cells", "Pluripotent stem cells"]),
    "hematopoietic_progenitor_like": ("hematopoietic", ["Hematopoietic stem cells"]),
    "erythroid_like": ("hematopoietic", ["Erythroblasts", "Erythroid-like and erythroid precursor cells", "Reticulocytes"]),
    "megakaryocyte_like": ("hematopoietic", ["Megakaryocytes", "Platelets"]),
    "B_like": ("immune", ["B cells", "B cells memory", "B cells naive", "Plasma cells"]),
    "T_like": ("immune", ["T cells", "T helper cells", "T cytotoxic cells", "T regulatory cells", "Thymocytes"]),
    "NK_like": ("immune", ["NK cells"]),
    "myeloid_like": ("immune", ["Monocytes", "Macrophages", "Dendritic cells", "Neutrophils"]),
    "epithelial_like": ("epithelial", ["Epithelial cells", "Basal cells", "Luminal epithelial cells"]),
    "hepatocyte_like": ("epithelial", ["Hepatocytes", "Hepatoblasts"]),
    "neural_progenitor_like": ("neural", ["Neural stem/precursor cells", "Neuroblasts"]),
    "neuron_like": ("neural", ["Neurons", "Immature neurons"]),
    "glial_like": ("neural", ["Astrocytes", "Oligodendrocytes", "Schwann cells"]),
    "stromal_like": ("mesenchymal", ["Fibroblasts", "Stromal cells", "Mesenchymal stem cells"]),
    "endothelial_like": ("vascular", ["Endothelial cells"]),
    "muscle_like": ("muscle", ["Myocytes", "Myoblasts", "Cardiomyocytes", "Smooth muscle cells"]),
    "germline_like": ("germline", ["Germ cells", "Spermatocytes", "Spermatozoa"]),
}
RULES = {"minimum_markers": 10, "minimum_coverage": .3, "minimum_detected": 3,
         "rank_score": .05, "expression_score": .1, "rank_margin": .03,
         "expression_margin": .1, "seed": 20260919, "state_bins": 24, "state_controls_per_gene": 10}


def validate_records(frame, input_hash, n):
    from rna import value_hash
    if len(frame) != n or frame.record_id.duplicated().any():
        raise ValueError("annotation_record_count_or_identity_mismatch")
    if not np.array_equal(frame.row_index.to_numpy(), np.arange(n)) or not (frame.input_sha256 == input_hash).all():
        raise ValueError("annotation_source_row_identity_mismatch")
    if frame.record_id.tolist() != [value_hash([input_hash, i]) for i in range(n)]:
        raise ValueError("annotation_record_hash_mismatch")


def marker_model(genes, profiles):
    """Only unique, already resolved input symbols enter this analysis view."""
    counts = Counter(x for x in genes if x is not None)
    lookup = {g: i for i, g in enumerate(genes) if g is not None and counts[g] == 1}
    frequency = Counter(g for p in profiles.values() for g in set(p["genes"]))
    weights = np.zeros((len(genes), len(profiles)), dtype=np.float32)
    membership = np.zeros_like(weights)
    coverage = []
    for j, (name, profile) in enumerate(profiles.items()):
        reference = set(profile["genes"])
        observed = sorted(reference & lookup.keys())
        fraction = len(observed) / len(reference) if reference else 0.
        eligible = len(observed) >= RULES["minimum_markers"] and fraction >= RULES["minimum_coverage"]
        for gene in observed:
            weights[lookup[gene], j] = 1 / frequency[gene]
            membership[lookup[gene], j] = 1
        if weights[:, j].sum():
            weights[:, j] /= weights[:, j].sum()
        coverage.append({"profile": name, "lineage": profile["lineage"], "reference_markers": len(reference),
                         "measured_markers": len(observed), "coverage": fraction, "eligible": eligible,
                         "status": "completed" if eligible else "not_estimable",
                         "reason": None if eligible else "insufficient_unique_measured_markers",
                         "measured_symbols": observed, "unmeasured_or_ambiguous": sorted(reference - lookup.keys())})
    return {"names": list(profiles), "lineages": [p["lineage"] for p in profiles.values()],
            "weights": weights, "membership": membership, "eligible": np.array([c["eligible"] for c in coverage]),
            "coverage": coverage, "lookup": lookup, "valid_axis": np.array([g is not None and counts[g] == 1 for g in genes])}


def decide(rank, expression, detected, model):
    n, k = rank.shape
    if not k or not model["eligible"].any():
        return pd.DataFrame({"inferred_lineage": ["unknown"] * n, "inferred_type": ["unknown"] * n,
                             "inferred_subtype": ["unknown"] * n, "inference_status": ["not_estimable"] * n,
                             "inference_reason": ["no_applicable_reference"] * n,
                             "method_conflict": [False] * n, "mixed_marker_signal": [False] * n,
                             "confidence_calibration": ["uncalibrated"] * n,
                             "probability_correct": [np.nan] * n})
    rs, es = rank.copy(), expression.copy()
    rs[:, ~model["eligible"]], es[:, ~model["eligible"]] = -np.inf, -np.inf
    ir, ie = rs.argmax(axis=1), es.argmax(axis=1)
    ix = np.arange(n)
    rtop, etop = rs[ix, ir], es[ix, ie]
    rmargin = rtop - np.partition(rs, -2, axis=1)[:, -2] if k > 1 else np.full(n, np.inf)
    emargin = etop - np.partition(es, -2, axis=1)[:, -2] if k > 1 else np.full(n, np.inf)
    support = (rtop >= RULES["rank_score"]) & (etop >= RULES["expression_score"]) & (detected[ix, ir] >= 3) & (detected[ix, ie] >= 3)
    separated = (rmargin >= RULES["rank_margin"]) & (emargin >= RULES["expression_margin"])
    same = ir == ie
    lineages = np.asarray(model["lineages"])
    coarse = support & (lineages[ir] == lineages[ie])
    assigned = support & separated & same
    reasons = np.where(~support, "weak_marker_support", np.where(assigned, "two_scores_support_broad_type",
        np.where(~same, "method_conflict", "mixed_or_small_margin")))
    frame = pd.DataFrame({"rank_candidate": np.asarray(model["names"])[ir], "expression_candidate": np.asarray(model["names"])[ie],
        "rank_score": rtop, "expression_score": etop, "rank_margin": rmargin, "expression_margin": emargin,
        "rank_candidate_detected_markers": detected[ix, ir].astype(int), "expression_candidate_detected_markers": detected[ix, ie].astype(int),
        "inferred_lineage": np.where(coarse, lineages[ir], "unknown"),
        "inferred_type": np.where(assigned, np.asarray(model["names"])[ir], "unknown"),
        "inferred_subtype": "unknown", "inference_status": "completed", "inference_reason": reasons,
        "method_conflict": ~same, "mixed_marker_signal": support & ~separated,
        "confidence_calibration": "uncalibrated", "probability_correct": np.nan})
    return frame.replace([np.inf, -np.inf], np.nan)


def annotate(log, targets, model):
    valid = model["valid_axis"]
    if not valid.any():
        return decide(np.zeros((len(log), len(model["names"]))), np.zeros((len(log), len(model["names"]))),
                      np.zeros((len(log), len(model["names"]))), model), {}
    rank = np.zeros_like(log, dtype=np.float32)
    rank[:, valid] = rankdata(log[:, valid], axis=1, method="average") / valid.sum() - .5
    mean, std = log[:, valid].mean(axis=1), log[:, valid].std(axis=1)
    standardized = np.zeros_like(log, dtype=np.float32)
    standardized[:, valid] = np.divide(log[:, valid] - mean[:, None], std[:, None],
                                      out=np.zeros_like(log[:, valid]), where=std[:, None] > 0)
    rs, es = rank @ model["weights"], standardized @ model["weights"]
    detected = (log > 0).astype(np.float32) @ model["membership"]
    result = decide(rs, es, detected, model)
    # Remove only each record's named target, then renormalize the fixed marker weights.
    rs2, es2, detected2 = rs.copy(), es.copy(), detected.copy()
    for target in set(targets):
        if target not in model["lookup"]:
            continue
        g = model["lookup"][target]
        rows = np.flatnonzero(np.asarray(targets) == target)
        w = model["weights"][g]
        rs2[rows] = (rs[rows] - rank[rows, g, None] * w) / np.maximum(1 - w, 1e-8)
        es2[rows] = (es[rows] - standardized[rows, g, None] * w) / np.maximum(1 - w, 1e-8)
        detected2[rows] -= (log[rows, g, None] > 0) * model["membership"][g]
    sensitivity = decide(rs2, es2, detected2, model)
    result["target_excluded_type"] = sensitivity.inferred_type
    result["target_marker_label_changed"] = result.inferred_type != sensitivity.inferred_type
    result["target_is_reference_marker"] = [t in model["lookup"] and bool(model["membership"][model["lookup"][t]].any()) for t in targets]
    scores = {"rank__" + name: rs[:, j] for j, name in enumerate(model["names"])}
    scores.update({"expression__" + name: es[:, j] for j, name in enumerate(model["names"])})
    return result, scores


def state_model(genes, sets, baseline_mean):
    counts = Counter(g for g in genes if g is not None)
    lookup = {g: i for i, g in enumerate(genes) if g is not None and counts[g] == 1}
    valid = np.array(sorted(lookup.values()), dtype=int)
    order = valid[np.argsort(baseline_mean[valid], kind="stable")]
    bins = np.full(len(genes), -1)
    bins[order] = np.arange(len(order)) * RULES["state_bins"] // max(len(order), 1)
    rng = np.random.default_rng(RULES["seed"])
    weights = np.zeros((len(genes), len(sets)), dtype=np.float32)
    coverage = []
    for j, (name, reference) in enumerate(sets.items()):
        symbols = set(reference["genes"])
        ids = np.array([lookup[g] for g in sorted(symbols & lookup.keys())], dtype=int)
        eligible = len(ids) >= 5 and len(ids) / max(1, len(symbols)) >= .3
        background = []
        if eligible:
            for i in ids:
                pool = np.setdiff1d(valid[bins[valid] == bins[i]], ids)
                if len(pool):
                    background.extend(rng.choice(pool, min(len(pool), 10), replace=False))
            eligible = bool(background)
        if eligible:
            weights[ids, j] = 1 / len(ids)
            for i in background:
                weights[i, j] -= 1 / len(background)
        coverage.append({"state": name, "reference_markers": len(symbols), "measured_markers": len(ids),
                         "coverage": len(ids) / max(1, len(symbols)), "status": "completed" if eligible else "not_estimable",
                         "reference": reference["reference"], "background_source_rows": sorted(set(map(int, background)))})
    return weights, coverage
