import numpy as np


def group_means(cams, labels=None):
    cams = np.asarray(cams, dtype=np.float32)
    if cams.ndim != 3:
        raise ValueError(f"Expected [N,H,W] CAMs, got {cams.shape}")
    groups = {"overall": np.arange(len(cams))}
    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        if len(labels) != len(cams):
            raise ValueError("CAM and label counts differ")
        groups.update({f"label_{label}": np.flatnonzero(labels == label) for label in (0, 1)})
    result = {}
    for name, indices in groups.items():
        if len(indices) == 0:
            continue
        result[name] = {
            "mean": cams[indices].mean(axis=0),
            "n": int(len(indices)),
            "zero_map_count": int((cams[indices].max(axis=(1, 2)) == 0).sum()),
        }
    return result
