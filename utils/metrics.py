import torch
from typing import Dict, Any
from sklearn.metrics import roc_auc_score as auc


def calc_F1(TP: float,
            TN: float,
            FP: float,
            FN: float) -> float:
    """
    compute F1 score for a epoch of TP, TN, FP, FN
    """

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    return F1

def calc_AUC(mask_array, predict_array):
    """
    compute AUC score
    """
    return auc(mask_array, predict_array)

def cal_confusion_matrix(gt_mask: torch.Tensor,
                         predict_mask: torch.Tensor) -> Dict[str, Any]:
    """
    calculate a bathsize TP, TN, FP, FN
    Args:
        predict (torch.Tensor): predict mask, shape [b, h, w]
        mask (torch.Tensor): ground truth, shape [b, h, w]
    Returns:
        metrics (Dict) :  TP, TN, FP, FN metrics
    """
    assert predict_mask.shape == gt_mask.shape, \
        "The shape of prediction is different from the mask"
    assert torch.all(torch.isin(predict_mask, torch.tensor([0, 1]))) ,\
        "The value of predict should be 0 or 1"
    assert torch.all(torch.isin(gt_mask, torch.tensor([0, 1]))) ,\
        "The value of ground truth should be 0 or 1"
    assert len(predict_mask.shape) == 3, \
        "the shape of prediction should be [b, h, w]"
    assert len(gt_mask.shape) == 3, \
        "the shape of mask should be [b, h, w]"

    return {
        "TP": torch.sum(predict_mask * gt_mask, dim=(1, 2)),
        "TN": torch.sum((1 - predict_mask) * (1 - gt_mask), dim=(1, 2)),
        "FP": torch.sum(predict_mask * (1 - gt_mask), dim=(1, 2)),
        "FN": torch.sum((1 - predict_mask) * gt_mask, dim=(1, 2))
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
    metrics = cal_confusion_matrix(predict, mask)
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
    metrics = cal_confusion_matrix(predict, mask)
    print("metrics2 : ",metrics)
    print("F1 score:", calc_F1(metrics["TP"].sum(),
                               metrics["TN"].sum(),
                               metrics["FP"].sum(),
                               metrics["FN"].sum()))

    mask_array = [1, 1, 1, 1, 0, 0, 0, 0]
    predict_array = [1, 0, 1, 1, 0, 1, 1, 0]
    print("auc value", calc_AUC(mask_array, predict_array))
