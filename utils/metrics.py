import torch
from typing import Dict
from sklearn.metrics import roc_auc_score as auc


def calc_F1(TP: float,
            TN: float,
            FP: float,
            FN: float) -> float:
    """
    compute F1 score
    """

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    return F1

def calc_AUC(mask_array: torch.Tensor,
             predict_array: torch.Tensor,
             reversed: bool = False):
    """
    compute AUC score
    Args:
        mask_array (torch.Tensor): ground truth, shape [h, w]
        predict_array (torch.Tensor): probability array for prediction, shape [h, w]
        reversed (bool) :
            if True, calculate AUC using inverted predicted values
    Return:
        auc metrics
    """
    assert len(mask_array.shape) == 2, \
        "the shape of mask_array should be [h, w]"
    assert len(predict_array.shape) == 2, \
        "the shape of predict_array should be [h, w]"
    assert torch.all((predict_array >= 0) & (predict_array <= 1)), \
        "the value of predict_array should be [0, 1]"

    if reversed:
        predict_array = 1 - predict_array

    m = mask_array.flatten()
    p = predict_array.flatten()
    return auc(m, p)

def cal_confusion_matrix(gt_mask: torch.Tensor,
                         predict_mask: torch.Tensor) -> Dict[str, float]:
    """
    calculate  TP, TN, FP, FN for a image
    Args:
        gt_mask (torch.Tensor): ground truth
        predict_mask (torch.Tensor): predict mask
    Returns:
        metrics (Dict) :  TP, TN, FP, FN metrics
    """
    assert predict_mask.shape == gt_mask.shape, \
        "the shape of prediction is different from the mask"
    assert torch.all(torch.isin(predict_mask, torch.tensor([0, 1]))) ,\
        "the value of predict should be 0 or 1"
    assert torch.all(torch.isin(gt_mask, torch.tensor([0, 1]))) ,\
        "the value of ground truth should be 0 or 1"
    assert len(gt_mask.shape) == 2, \
        "the shape of mask should be [h, w]"
    assert len(predict_mask.shape) == 2, \
        "the shape of prediction should be [h, w]"


    return {
        "TP": torch.sum(predict_mask * gt_mask, dim=(0, 1)).item(),
        "TN": torch.sum((1 - predict_mask) * (1 - gt_mask), dim=(0, 1)).item(),
        "FP": torch.sum(predict_mask * (1 - gt_mask), dim=(0, 1)).item(),
        "FN": torch.sum((1 - predict_mask) * gt_mask, dim=(0, 1)).item()
    }



if __name__ == "__main__":

    predict = torch.tensor(
        [[1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]]
    , dtype=torch.float32)

    mask = torch.tensor(
        [[1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]]
    , dtype=torch.float32)
    confusion_matrix = cal_confusion_matrix(mask, predict)
    print("metrics : ",confusion_matrix)
    print("F1 score:", calc_F1(confusion_matrix["TP"],
                               confusion_matrix["TN"],
                               confusion_matrix["FP"],
                               confusion_matrix["FN"]))

    mask_array = torch.Tensor([[1, 1, 1, 0, 0],
                               [1, 0, 0, 1, 1]])
    predict_array = torch.Tensor([[0.9, 0.8, 0.6, 0.1, 0.4],
                                  [0.7, 0.8, 0.7, 0.6, 0.9]])
    auc1 = calc_AUC(mask_array, predict_array)
    print("auc1 value", auc1, type(auc1))

    mask_array = torch.Tensor([[1, 1, 1, 1],
                               [0, 0, 0, 0]])
    predict_array = torch.Tensor([[1, 0, 1, 1],
                                  [0, 1, 1, 0]])
    print("auc value2", calc_AUC(mask_array, predict_array))
