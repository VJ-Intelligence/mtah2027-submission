"""Thin wrapper over the OFFICIAL metric code, so every number we report is the official one."""
import numpy as np

from .paths import official_on_path

official_on_path()
import possession_metrics as pm  # noqa: E402


def codes(labels) -> np.ndarray:
    return pm.label_codes(np.asarray(labels))


def score(gt_labels, pred_labels, runs=None) -> dict:
    """spellF1 (ranking metric) + macroF1 (tie-break) with the official functions.

    runs: list of [start, stop) spans to grade (e.g. one per half). Default: wherever gt != X.
    """
    gt, pred = codes(gt_labels), codes(pred_labels)
    if runs is None:
        runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spells = pm.spell_scores(gt, filled, runs)
    mask = np.zeros(len(gt), bool)
    for a, b in runs:
        mask[a:b] = True
    cm3 = pm.scored_confusion(pm.confusion(gt[mask], pred[mask]))
    return {
        "spell_f1": spells["mean_f1"],
        "macro_f1": pm.macro_f1(cm3),
        "n_gt_spells": len(pm.possession_spells(gt, runs)),
        "n_pred_spells": len(pm.possession_spells(filled, runs)),
    }
