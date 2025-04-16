import torch
from typing import Dict


def calc_F1(TP: float, TN: float, FP: float, FN: float):
    """
    compute F1 score for a batch of TP, TN, FP, FN
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    return F1

def calc_AUC(TP: float, TN: float, FP: float, FN: float):
    """
    compute AUC score for a batch of TP, TN, FP, FN
    """
    pass

def calc_TP_TN_FP_FN(predict: torch.Tensor,
                    mask: torch.Tensor,) -> Dict[str, float]:
    """
    calculate a bathsize TP, TN, FP, FN
    Args:
        predict (torch.Tensor): predict mask, shape [b, h, w]
        mask (torch.Tensor): ground truth, shape [b, h, w]
    Returns:
        metrics (Dict) :  TP, TN, FP, FN metrics
    """
    assert predict.shape == mask.shape, \
        "The shape of prediction is different from the mask"
    assert torch.all(torch.isin(predict, torch.tensor([0, 1]))) ,\
        "The value of predict should be 0 or 1"
    assert torch.all(torch.isin(mask, torch.tensor([0, 1]))) ,\
        "The value of ground truth should be 0 or 1"

    return {
        "TP": torch.sum(predict * mask, dim=(1, 2)).sum().item(),
        "TN": torch.sum((1 - predict) * (1 - mask), dim=(1, 2)).sum().item(),
        "FP": torch.sum(predict * (1 - mask), dim=(1, 2)).sum().item(),
        "FN": torch.sum((1 - predict) * mask, dim=(1, 2)).sum().item()
    }


if __name__ == "__main__":

    predict = torch.tensor([
        [[1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]]
    ], dtype=torch.float32)

    mask = torch.tensor([
        [[1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]]
    ], dtype=torch.float32)
    metrics = calc_TP_TN_FP_FN(predict, mask)
    print("metrics1 : ",metrics)

    predict = torch.tensor([
        [[1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]],
        [[1, 0, 1],
        [1, 1, 1],
        [1, 1, 0]]
    ], dtype=torch.float32)

    mask = torch.tensor([
        [[1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]],
        [[1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]]
    ], dtype=torch.float32)
    metrics = calc_TP_TN_FP_FN(predict, mask)
    print("metrics2 : ",metrics)