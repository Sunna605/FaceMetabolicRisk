import torch
from torch import nn


POSITIVE_CLASS_WEIGHTS = {
    "MetS": 2.315862,
    "obesity": 6.680511,
    "COE": 1.734926,
    "dyslipidemia": 1.931707,
    "hypertension": 1.479629,
    "hyperglycemia": 8.193117,
}


def weighted_cross_entropy(task, device="cpu"):
    """Return the study's binary class-weighted cross-entropy criterion."""
    if task not in POSITIVE_CLASS_WEIGHTS:
        raise KeyError(f"Unknown task: {task}")
    weights = torch.tensor([1.0, POSITIVE_CLASS_WEIGHTS[task]], device=device)
    return nn.CrossEntropyLoss(weight=weights)
