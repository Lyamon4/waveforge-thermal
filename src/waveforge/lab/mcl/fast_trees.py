"""Exact continuous HGBR tree traversal, without per-request estimator overhead."""

import numpy as np

try:
    from numba import njit
except ImportError:

    def njit(**kwargs):
        return lambda f: f


def pack_predictor(model):
    if model.n_trees_per_iteration_ != 1 or model.is_categorical_ is not None:
        raise ValueError("only continuous single-output regressors are supported")
    trees = [iteration[0].nodes for iteration in model._predictors]
    roots = np.cumsum(np.r_[0, [len(t) for t in trees[:-1]]]).astype(np.int64)
    nodes = np.concatenate(trees)
    offsets = np.repeat(roots, [len(t) for t in trees])
    return dict(
        value=nodes["value"].copy(),
        threshold=nodes["num_threshold"].copy(),
        feature=nodes["feature_idx"].copy(),
        left=nodes["left"].astype(np.int64) + offsets,
        right=nodes["right"].astype(np.int64) + offsets,
        missing=nodes["missing_go_to_left"].copy(),
        leaf=nodes["is_leaf"].copy(),
        roots=roots,
        baseline=float(model._baseline_prediction[0, 0]),
        features=model.n_features_in_,
    )


@njit(cache=True, nogil=True)
def _predict(value, threshold, feature, left, right, missing, leaf, roots, baseline, x):
    result = np.full(len(x), baseline)
    for row in range(len(x)):
        for root in roots:
            node = root
            while not leaf[node]:
                v = x[row, feature[node]]
                go_left = bool(missing[node]) if np.isnan(v) else v <= threshold[node]
                node = left[node] if go_left else right[node]
            result[row] += value[node]
    return result


def predict_packed(packed, x):
    x = np.ascontiguousarray(x[:, : packed["features"]], dtype=np.float64)
    if x.shape[1] != packed["features"]:
        raise ValueError("wrong tree input dimensions")
    return _predict(
        *(
            packed[k]
            for k in (
                "value",
                "threshold",
                "feature",
                "left",
                "right",
                "missing",
                "leaf",
                "roots",
                "baseline",
            )
        ),
        x,
    )
