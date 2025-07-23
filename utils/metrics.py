import torch
from typing import List
from sklearn.metrics import roc_auc_score


class PixelF1():
    def __init__(self):
        pass

    def confuse_matrix(self, gt_prob_batch: torch.Tensor, predict_prob_batch: torch.Tensor):
        """
        Calculate the confusion matrix for pixel-wise F1 score for a batch of images.
        Args:
            gt_prob_batch (torch.Tensor): Ground truth probability batch, shape [N, C, H, W]
            predict_prob_batch (torch.Tensor): Predicted probability batch, shape [N, C, H, W]

        the gt_prob_batch and predict_prob_batch should be binary masks
        Returns:
            Dict: TP, TN, FP, FN counts for each image in the batch
            TP (True Positive): Pixels correctly predicted as positive
            TN (True Negative): Pixels correctly predicted as negative
            FP (False Positive): Pixels incorrectly predicted as positive
            FN (False Negative): Pixels incorrectly predicted as negative
        """
        return {
            "TP": torch.sum(predict_prob_batch * gt_prob_batch, dim=(1, 2, 3)),
            "TN": torch.sum((1 - predict_prob_batch) * (1 - gt_prob_batch), dim=(1, 2, 3)),
            "FP": torch.sum(predict_prob_batch * (1 - gt_prob_batch), dim=(1, 2, 3)),
            "FN": torch.sum((1 - predict_prob_batch) * gt_prob_batch, dim=(1, 2, 3))
        }

    def calc_F1(self,
                TP: float,
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

class PixelAUC():
    def __init__(self):
        pass

    def calc_single_AUC_regular(self, gt_prob_img: torch.Tensor, predict_prob_img: torch.Tensor) -> float:
        """
        Calculate AUC score for a single image.
        from https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/evaluation/AUC.py
        """
        y_true = gt_prob_img.flatten()
        y_scores = predict_prob_img.flatten()

        # Check if the mask has only one class
        if len(y_true.unique()) < 2:
            print(f"Warning: Ground truth for image has only one class, AUC cannot be computed.")
            return 0.0

        # Sort scores and corresponding true labels
        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        # Calculate the number of positive and negative samples
        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        # Calculate cumulative true positives and false positives
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
        tpr = tps / n_pos
        fpr = fps / n_neg

        # Calculate AUC using the trapezoidal rule
        auc = torch.trapz(tpr, fpr)

        return auc.item()

    def calc_single_AUC_regular2(self,
                                gt_prob_img: torch.Tensor,
                                predict_prob_img: torch.Tensor) -> float:
        """
        Tie‑aware AUC 计算（与 sklearn 结果一致）
        Args:
            gt_prob_img       : torch.Tensor  [C,H,W]  (0/1)
            predict_prob_img  : torch.Tensor  [C,H,W]  (任意实数分数)

        Returns
        -------
            float : AUC 值；若只有单一类别则返回 nan
        """
        # 展平
        y_true  = gt_prob_img.reshape(-1).float()
        y_score = predict_prob_img.reshape(-1).float()

        # 若仅有一个类别，AUC 无定义
        n_pos = int(torch.sum(y_true).item())
        n_neg = y_true.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        # ---------- Mann–Whitney U / Wilcoxon 秩和 ----------
        # 1) 对分数升序排序
        order = torch.argsort(y_score, descending=False)
        y_true_sorted  = y_true[order]
        scores_sorted  = y_score[order]

        # 2) 计算平均秩（处理 ties）
        #    找到分数块边界
        new_block = torch.ones_like(scores_sorted, dtype=torch.bool)
        new_block[1:] = scores_sorted[1:] != scores_sorted[:-1]
        block_id   = torch.cumsum(new_block, 0) - 1  # 0‑based
        block_cnt  = torch.bincount(block_id)        # 每块大小

        #    每块起始秩（从 1 开始）
        first_rank = torch.zeros_like(block_cnt, dtype=torch.float32)
        first_rank[1:] = torch.cumsum(block_cnt[:-1], 0)
        avg_rank = first_rank + (block_cnt.float() - 1) / 2.0 + 1.0

        # 3) 为每个样本分配平均秩
        ranks = avg_rank[block_id]

        # 4) 计算正样本秩和 → AUC
        R_plus = torch.sum(ranks * y_true_sorted).item()
        auc = (R_plus - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        return float(auc)

    def calc_AUC_regular(self, gt_prob_batch: torch.Tensor, predict_prob_batch: torch.Tensor) -> List:
        """
        Calculate AUC score for a batch of images.
        Args:
            gt_prob_batch (torch.Tensor): Ground truth probability batch, shape [N, C, H, W]
            predict_prob_batch (torch.Tensor): Predicted probability batch, shape [N, C, H, W]
        Returns:
            List: AUC score for every image in the batch
        """

        AUC_List = []
        for i in range(gt_prob_batch.shape[0]):
            gt_prob = gt_prob_batch[i]
            predict_prob = predict_prob_batch[i]
            auc_score = self.calc_single_AUC_regular2(gt_prob, predict_prob)
            AUC_List.append(auc_score)
        return AUC_List


    def calc_AUC_sklearn(self, gt_prob_batch: torch.Tensor, predict_prob_batch: torch.Tensor) -> List:
        """
        Calculate AUC score for a batch of images.
        Args:
            gt_prob_batch (torch.Tensor): Ground truth probability batch, shape [N, C, H, W]
            predict_prob_batch (torch.Tensor): Predicted probability batch, shape [N, C, H, W]
        Returns:
            float: AUC score for every image in the batch
        """
        assert gt_prob_batch.shape == predict_prob_batch.shape, \
            "Ground truth and predicted probability batches must have the same shape"
        AUC_List = []
        for i in range(gt_prob_batch.shape[0]):
            gt_prob = gt_prob_batch[i].view(-1).cpu().numpy()
            predict_prob = predict_prob_batch[i].view(-1).cpu().numpy()
            if len(set(gt_prob)) > 1:
                auc_score = roc_auc_score(gt_prob, predict_prob)
                AUC_List.append(auc_score)
            else:
                print(f"Warning: Ground truth for image {i} has only one class, AUC cannot be computed.")
                AUC_List.append(0.0)
        return AUC_List

class DatasetAUC():
    def __init__(self, device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
       self.predict_label = torch.tensor([], device=device)
       self.label = torch.tensor([], device=device)

    def reset(self, device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        """Resets the stored predictions and labels for a new epoch."""
        self.predict_label = torch.tensor([], device=device)
        self.label = torch.tensor([], device=device)

    def add_batch(self, predict_prob_batch: torch.Tensor, gt_prob_batch: torch.Tensor):
        """
        Add a batch of predictions and ground truth to the dataset.
        Args:
            predict_prob_batch (torch.Tensor): Predicted probability batch, shape [N, C, H, W]
            gt_prob_batch (torch.Tensor): Ground truth probability batch, shape [N, C, H, W]
        """

        self.predict_label = torch.cat((self.predict_label, predict_prob_batch.view(-1)), dim=0)
        self.label = torch.cat((self.label, gt_prob_batch.view(-1)), dim=0)

    def calc_AUC_sklearn(self):
        """
        Calculate the AUC score for the dataset.
        Returns:
            float: AUC score for the dataset
        """
        assert self.label.numel() > 0 and self.predict_label.numel() > 0, \
            "No data available to calculate AUC. Please add batches first."
        return roc_auc_score(self.label.cpu().numpy(), self.predict_label.cpu().numpy())

    def calc_single_AUC_regular(self) -> float:
        """
        Calculate AUC score for a single image.
        from https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/evaluation/AUC.py
        """
        y_true = self.label
        y_scores = self.predict_label

        # Check if the mask has only one class
        if len(y_true.unique()) < 2:
            print(f"Warning: Ground truth for image has only one class, AUC cannot be computed.")
            return 0.0

        # Sort scores and corresponding true labels
        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        # Calculate the number of positive and negative samples
        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        # Calculate cumulative true positives and false positives
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
        tpr = tps / n_pos
        fpr = fps / n_neg

        # Calculate AUC using the trapezoidal rule
        auc = torch.trapz(tpr, fpr)

        return auc.item()

    def calc_single_AUC_regular2(self) -> float:
        """
        Tie‑aware AUC 计算（与 sklearn 结果一致）
        Args:
            gt_prob_img       : torch.Tensor  [C,H,W]  (0/1)
            predict_prob_img  : torch.Tensor  [C,H,W]  (任意实数分数)

        Returns
        -------
            float : AUC 值；若只有单一类别则返回 nan
        """
        # 展平
        y_true  = self.label
        y_score = self.predict_label

        # 若仅有一个类别，AUC 无定义
        n_pos = int(torch.sum(y_true).item())
        n_neg = y_true.numel() - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        # ---------- Mann–Whitney U / Wilcoxon 秩和 ----------
        # 1) 对分数升序排序
        order = torch.argsort(y_score, descending=False)
        y_true_sorted  = y_true[order]
        scores_sorted  = y_score[order]

        # 2) 计算平均秩（处理 ties）
        #    找到分数块边界
        new_block = torch.ones_like(scores_sorted, dtype=torch.bool)
        new_block[1:] = scores_sorted[1:] != scores_sorted[:-1]
        block_id   = torch.cumsum(new_block, 0) - 1  # 0‑based
        block_cnt  = torch.bincount(block_id)        # 每块大小

        #    每块起始秩（从 1 开始）
        first_rank = torch.zeros_like(block_cnt, dtype=torch.float32)
        first_rank[1:] = torch.cumsum(block_cnt[:-1], 0)
        avg_rank = first_rank + (block_cnt.float() - 1) / 2.0 + 1.0

        # 3) 为每个样本分配平均秩
        ranks = avg_rank[block_id]

        # 4) 计算正样本秩和 → AUC
        R_plus = torch.sum(ranks * y_true_sorted).item()
        auc = (R_plus - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        return float(auc)



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predict_prob_batch = torch.tensor([
        # --- Image 1 ---
        [[[1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]]]
        # --- Image 2 ---
        , [[[0, 1, 0],
        [1, 0, 1],
        [0, 0, 1]]]
        ]
    , dtype=torch.float32).to(device)

    gt_prob_batch = torch.tensor([
        # --- Image 1 ---
        [[[1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]]],
        # --- Image 2 ---
        [[[0, 1, 1],
         [1, 0, 1],
        [1, 0, 1]]]
        ]
    , dtype=torch.float32).to(device)

    print(f"predict_prob_batch shape: {predict_prob_batch.shape}")
    print(f"gt_prob_batch shape: {gt_prob_batch.shape}")
    pixel_f1 = PixelF1()
    cm = pixel_f1.confuse_matrix(gt_prob_batch, predict_prob_batch)
    print(f"Confusion Matrix: {cm}")
    # calculate F1 score based on all images in the batch
    TP = cm["TP"].sum().item()
    TN = cm["TN"].sum().item()
    FP = cm["FP"].sum().item()
    FN = cm["FN"].sum().item()
    f1score = pixel_f1.calc_F1(TP, TN, FP, FN)
    print(f"F1 Score: {f1score:.4f}")
    # ==========================================================
    print("=================================================")
    pixel_auc = PixelAUC()
    print(f"AUC Scores by sklearn: {pixel_auc.calc_AUC_sklearn(gt_prob_batch, predict_prob_batch)}")
    print(f"AUC Scores by regular method: {pixel_auc.calc_AUC_regular(gt_prob_batch, predict_prob_batch)}")
    print("=================================================")
    dataset_auc = DatasetAUC(device="cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_auc.add_batch(predict_prob_batch, gt_prob_batch)
    auc_score_sklearn = dataset_auc.calc_AUC_sklearn()
    auc_score_regular = dataset_auc.calc_single_AUC_regular()
    auc_score_regular2 = dataset_auc.calc_single_AUC_regular2()
    print(f"AUC Score for the dataset: {auc_score_sklearn:.4f}")
    print(f"AUC Score for the dataset (regular method): {auc_score_regular:.4f}")
    print(f"AUC Score for the dataset (regular method 2): {auc_score_regular2:.4f}")