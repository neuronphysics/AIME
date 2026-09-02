"""Segmentation metrics shared by all runners.

Hamming distance follows the NPBayesHMM / bnpy protocol: build the
true-vs-predicted contingency table over the *union* of label sets, find the
optimal one-to-one matching with the Hungarian algorithm, and report the
fraction of misassigned timesteps.  ``mapping`` sends predicted labels to the
matched true label and is reused by the plotting code so every model's ribbon
is drawn in ground-truth colours.

Raw model objectives (SHS variational bound, rSLDS joint log-likelihood) are
recorded per model for convergence figures but are NOT comparable across model
classes -- different latent spaces, different likelihoods.  Label-based
metrics are the shared currency; that is why they are the headline numbers.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _contingency(z_true, z_pred):
    z_true = np.asarray(z_true).ravel().astype(int)
    z_pred = np.asarray(z_pred).ravel().astype(int)
    n = int(max(z_true.max(), z_pred.max())) + 1
    C = np.zeros((n, n), dtype=np.int64)
    np.add.at(C, (z_true, z_pred), 1)
    return C


def hamming(z_true, z_pred):
    """(distance, mapping): optimal one-to-one label matching (Hungarian)."""
    C = _contingency(z_true, z_pred)
    r, c = linear_sum_assignment(-C)
    dist = 1.0 - C[r, c].sum() / np.asarray(z_true).size
    return float(dist), {int(cc): int(rr) for rr, cc in zip(r, c)}


def many_to_one(z_true, z_pred):
    """Each predicted state maps to its majority true label. Fair when the two
    models are not matched in state count."""
    C = _contingency(z_true, z_pred)
    return float(C.max(axis=0).sum() / np.asarray(z_true).size)


def all_metrics(z_true, z_pred):
    z_true = np.asarray(z_true).ravel().astype(int)
    z_pred = np.asarray(z_pred).ravel().astype(int)
    ham, mapping = hamming(z_true, z_pred)
    out = dict(hamming=ham, m2o=many_to_one(z_true, z_pred),
               K_used=int(np.unique(z_pred).size))
    try:
        from sklearn.metrics import (normalized_mutual_info_score,
                                     adjusted_rand_score)
        out["nmi"] = float(normalized_mutual_info_score(z_true, z_pred))
        out["ari"] = float(adjusted_rand_score(z_true, z_pred))
    except ImportError:            # legacy env without sklearn: skip
        out["nmi"] = out["ari"] = float("nan")
    return out, mapping
