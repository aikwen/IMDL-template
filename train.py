import config as Cfg
from pathlib import Path
from typing import Optional
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
import torch
from rich.text import Text
from utils import logs

cfg = Cfg.new_config()
PROJECT_DIR = Path(__file__).parent
CHECKPOINTS_DIR = PROJECT_DIR / cfg.get("train", "check_points", "dir")
LOGS_DIR = PROJECT_DIR / cfg.get("train", "logs", "dir")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class STATUS:
    Model: Optional[Module] = None
    Optimizer: Optional[Optimizer] = None
    Scheduler: Optional[LRScheduler] = None
    DataLoader_Train: Optional[DataLoader] = None
    Epoch: Optional[int] = 0

def init_dir():
    """
    初始化目录:权重目录 和 日志目录
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def init_models():
    """
    初始化模型
    Status.Model, Status.Optimizer, Status.Scheduler
    """
    from models import model
    # 初始化模型
    resuming_set = cfg.get("train", "check_points", "resuming_checkpoint")
    STATUS.Model = model.Model().to(DEVICE)
    log_resuming_set = ""
    if isinstance(resuming_set, str):
        log_resuming_set = resuming_set
        resuming_checkpoint_path = PROJECT_DIR / resuming_set
        assert resuming_checkpoint_path.exists(), f"{resuming_checkpoint_path} 权重不存在"
        resuming_checkpoint = torch.load(resuming_checkpoint_path)
        STATUS.Model.load_state_dict(resuming_checkpoint["model_state"], strict=True)
    else:
        log_resuming_set = "无"
        pass
    # 初始化优化器
    lr = cfg.get("train", "base", "adamw","lr")
    wd = cfg.get("train", "base", "adamw","weight_decay")
    STATUS.Optimizer = Cfg.new_optimizer(STATUS.Model, lr, wd)
    # 初始化schedule
    power = cfg.get("train", "base", "polynomialLR","power")
    total_iters = len(STATUS.DataLoader_Train) * cfg.get("train", "base", "epochs")
    STATUS.Scheduler = Cfg.new_schedule(STATUS.Optimizer, total_iters, power)

    with logs.Logs(LOGS_DIR / cfg.get("train", "logs", "name")) as logger:
        base_rows = [
            ["epochs", f"{cfg.get('train', 'base', 'epochs')}"],
            ["batchsize", f"{cfg.get('train', 'base', 'batchsize')}"],
            ["adamw-lr", f"{cfg.get('train', 'base', 'adamw', 'lr')}"],
            ["adamw-weight_decay", f"{cfg.get('train', 'base', 'adamw', 'weight_decay')}"],
            ["polynomialLR-power", f"{cfg.get('train', 'base', 'polynomialLR', 'power')}"],
        ]

        logger.print(logs.NewTableLog(
            logs.base_headers, base_rows, logs.base_title
        ))

        logger.print(Text(f"resuming_checkpoint:{log_resuming_set}", "bold"))
        logger.print(Text(f'checkpoint autosave:{cfg.get("train", "check_points", "auto_save")}', 'bold'))
        logger.print(Text(f'checkpoint saving dir:{cfg.get("train", "check_points", "dir")}', 'bold'))
        logger.print(Text(f'\n'))

def init_dataset():
    """
    初始化数据集
    """
    # 获取数据集相关参数
    datasets_path = cfg.get("datasets")
    preprocess_args = cfg.get("preprocess")
    # 训练数据集相关参数
    dataset_name_train = cfg.get("train", "training_set")
    preprogress_order_train = cfg.get("train", "train_preprocess")
    batchsize = cfg.get("train", "base", "batchsize")
    num_workers = cfg.get("train", "base", "dataLoader_workers")
    # 遍历数据集
    datasets_train = []
    # 数据集日志
    logs_train_data = []
    for dataset_name in dataset_name_train:
        dataset = Cfg.new_dataset(PROJECT_DIR / datasets_path[dataset_name],
                        preprocess_args,
                        preprogress_order_train)
        datasets_train.append(dataset)
        logs_train_data.append([dataset_name, f"{len(dataset)}"])
    
    STATUS.DataLoader_Train = Cfg.new_dataloader(
        datasets_train,
        batchsize,
        num_workers
    )

    # 数据集处理日志
    logs_train_process = []
    for idx, p in enumerate(preprogress_order_train, start=1):
        param = ""
        for k, v in preprocess_args[p].items():
            if k != "p":
                param = f"{v}"
        logs_train_process.append(
            [f"{idx}", p, f"{preprocess_args[p]['p']}", param])

    # 写入日志
    with logs.Logs(LOGS_DIR / cfg.get("train", "logs", "name")) as logger:
        # 写入数据集信息
        logger.print(logs.NewTableLog(
            logs.t_datasets_headers,
            logs_train_data,
            logs.t_datasets_title
        ))

        logger.print(logs.NewTableLog(
            logs.pre_headrs,
            logs_train_process,
            logs.pre_title
        ))

def num_width(n:int)-> int:
    width = 0
    while n > 0:
        width = width + 1
        n //=10
    return width

def save_checkpoints():
    """
    保存权重
    """
    n = cfg.get("train", "base", "epochs")
    path = CHECKPOINTS_DIR / f"checkpoints_epoch_{STATUS.Epoch:0{num_width(n)}}.pth"
    checkpoint = {
        "model_state": STATUS.Model.state_dict(),
        "optimizer_state": STATUS.Optimizer.state_dict(),
        "scheduler_state": STATUS.Scheduler.state_dict(),
        "epoch": STATUS.Epoch
    }
    torch.save(checkpoint, path)

def train_epoch(logger: logs.Logs):
    """
    训练一个轮次
    """
    for batch_idx, (tp, gt, tp_name, gt_name) in enumerate(STATUS.DataLoader_Train):
        tp = tp.to(DEVICE)
        gt = gt.unsqueeze(1).to(DEVICE)
        # 清空梯度
        STATUS.Optimizer.zero_grad()
        # 计算损失
        # 导入损失
        import models.loss as Loss
        outputs = STATUS.Model(tp)
        l = Loss.Loss(gt, outputs)
        #  反向传播
        l.backward()
        STATUS.Optimizer.step()
        # 学习率调整
        STATUS.Scheduler.step()
        ################################################################
        # 每 10 轮打印一次损失和学习率
        if batch_idx % 10 == 0:
            cur_l = f"{l.item():.5f}"
            cur_lr = f"{STATUS.Optimizer.param_groups[0]['lr']:.8f}"
            logger.print(Text.assemble(
                    (f"{STATUS.Epoch:>7}/{batch_idx:<8}", "bold"),
                    (f"{cur_l:^10}", "bold"),
                    (f"{cur_lr:^15}", "bold"),
                ))

def train():
    """
    训练代码
    """
    epochs = cfg.get("train", "base", "epochs")
    start_epoch = STATUS.Epoch
    ##########################################################################
    # 打印训练日志
    with logs.Logs(LOGS_DIR / cfg.get("train", "logs", "name")) as logger:
        logger.print(Text.assemble(
            (f"{'epoch/batch':^15}", "bold"),
            (f"{'loss':^10}", "bold"),
            (f"{'lr':^15}", "bold")
        ))

        for epoch in range(start_epoch, epochs):
            # 设置模型为训练模式
            STATUS.Model.train()
            # 设置当前 epoch
            STATUS.Epoch = epoch
            # 开始一次训练
            train_epoch(logger)
            # 训练完一个epoch保存一下权重
            if cfg.get("train", "check_points", "auto_save"):
                save_checkpoints()

if __name__ == "__main__":
    init_dir()
    init_dataset()
    init_models()
    train()