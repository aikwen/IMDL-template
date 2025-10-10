import yaml
from pathlib import Path
import pprint
from utils import dataset, img_ops
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch import optim


CONFFILE = Path(__file__).parent.joinpath("config.yaml")

class Config():
    def __init__(self, data) -> None:
        self.data = data

    def get(self, *args):
        temp = self.data
        for arg in args:
            temp = temp.get(arg, None)
            assert temp is not None, f"{args} 中的 {arg} 不存在"
        return temp

def new_config(path:Path=CONFFILE) -> Config:
    with open(path, 'r', encoding='utf-8') as file:
        return Config(yaml.safe_load(file))

def new_dataset(path, process_param, order:list)->Dataset:
    p = []
    for o in order:
        p.append(img_ops.get_aug(o, process_param[o]))
    aug_list = img_ops.aug_compose(p)
    return dataset.ImageDataset(path, aug_list)

def new_dataloader(datasets, batchsize, num_workers):
    combined_dataset = ConcatDataset(datasets)
    return DataLoader(combined_dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)

def new_optimizer(m:nn.Module, lr, weight_decay):
    adamw = optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    return adamw

def new_schedule(optimizer, total_iters, power):
    schedule = optim \
                .lr_scheduler \
                .PolynomialLR(optimizer=optimizer,
                              total_iters=total_iters,
                              power=power)
    return schedule

if __name__ == "__main__":
    pprint.pprint(new_config().data)