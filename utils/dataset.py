from PIL import Image
from torch.utils.data import Dataset
from typing import Union, Dict, List, Optional
import albumentations as albu
import pathlib
import json
import numpy as np
import torch

def rgba2rgb(rgba:np.ndarray, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


class ImageDataset(Dataset):
    """
    A dataset class for loading images and their corresponding masks from a directory structure.
    """
    def __init__(self, path:Union[str, pathlib.Path],
                 transform:Optional[albu.Compose]=None):
        """
        Args:
            path (Union[str, pathlib.Path]):  包含 gt文件夹, pt文件夹 and json 文件的文件夹路径
            postprocess (Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]):
                A function to post-process the images and masks after loading.
        """
        self.__check(path)
        self.path = pathlib.Path(path)
        # 数据集名称
        self.dataset_name = self.path.name
        # 数据集 json 文件
        self.dataset_json = self.path.joinpath(f"{self.dataset_name}.json")
        # 数据集列表
        self.dataset_list:List[Dict] = []
        self.dataset_num:int = 0

        self.transform = transform
        # 初始化数据集列表
        self.__dataList()

    def __check(self, path):
        assert isinstance(path, (str, pathlib.Path)), f"{path} 必须是 string 或者 pathlib.Path 类型"
        p = pathlib.Path(path)
        assert p.is_dir(), f"{path} 文件夹不存在"
        assert p.joinpath("gt").is_dir(), f"{path} 的 gt 文件夹不存在"
        assert p.joinpath("tp").is_dir(), f"{path} 的 tp 文件夹不存在"
        assert p.joinpath(f"{p.name}.json").is_file(), f"{path} 的 json 文件不存在"
        del p

    def __dataList(self):
        # 加载数据集 json 文件
        try:
            with open(self.dataset_json, 'r', encoding='utf-8') as f:
                self.dataset_list = json.load(f)
        except Exception as e:
            print(f"数据集 {self.dataset_name} json 文件加载错误 : {e}")

        self.dataset_num = len(self.dataset_list)

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 对应的下标
        Returns:
            Tuple[Tensor, Tensor, tp_name, gt_name]:
            tp tensor, gt tensor, tp 图像名， gt 图像名
        """
        # 加载对应 idx 的 tp、gt 图像路径
        tp_path = self.path.joinpath("tp", self.dataset_list[idx]["tp"])
        tp_name, gt_name = tp_path.name, ""

        if self.dataset_list[idx]["gt"] == "":
            gt_path = None
        else:
            gt_path = self.path.joinpath("gt", self.dataset_list[idx]["gt"])
            gt_name = gt_path.name

        # 加载图像
        try:
            tp_img = Image.open(tp_path)
            # 判断 gt 图像是否存在，如果不存在那么就生成一张全黑色图像
            if gt_path is not None:
                gt_img = Image.open(gt_path).convert("L")  # Convert mask to grayscale
            else:
                # tp_img.size return (width, height)
                w, h = tp_img.size
                gt_img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
        except FileNotFoundError as e:
            print(f"File not found: {tp_path} or {gt_path}")
            raise e
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            raise e

        # 获取图像矩阵
        gt_array = np.array(gt_img)
        if tp_img.mode == 'RGBA':
            tp_img.load()  # 确保图像被加载
            tp_array = rgba2rgb(np.array(tp_img))
        else:
            tp_img = tp_img.convert("RGB")
            tp_array = np.array(tp_img)

        # 对 tp， gt 矩阵进行一些增强
        if self.transform:
            transformed = self.transform(image=tp_array, mask=gt_array)
            tp_array = transformed['image']
            gt_array = transformed['mask']

        # 将 gt 矩阵变成概率矩阵
        if gt_array.max() > 1.0:
            gt_array = gt_array.astype(np.float32) / 255.0
            gt_array[gt_array > 0.5] = 1.0
            gt_array[gt_array <= 0.5] = 0.0

        # to tensor
        tp_tensor = torch.from_numpy(tp_array).to(torch.float32).permute(2, 0, 1) # Convert to CxHxW
        gt_tensor = torch.from_numpy(gt_array).to(torch.float32)
        return tp_tensor, gt_tensor, tp_name, gt_name

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    dataset = ImageDataset(r"./data/simple")
    print(f"Dataset {dataset.dataset_name} loaded with {len(dataset)} items.")
    img, mask, img_name, mask_name = dataset[0]
    print(img_name, mask_name)
    print(f"img shape:{img.shape}, img dtype:{img.dtype}")
    img_display = img.permute(1, 2, 0).numpy().astype(np.uint8)
    mask_display = mask.numpy()
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(img_display)
    ax[0].set_title(img_name)
    ax[1].imshow(mask_display, cmap='gray')
    ax[1].set_title(mask_name)
    plt.show()
