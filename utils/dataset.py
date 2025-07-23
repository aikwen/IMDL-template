from PIL import Image
from torch.utils.data import Dataset
from typing import Union, Dict, Optional, Callable, Tuple
import pathlib
import json
import numpy as np

class ImageDataset(Dataset):
    """
    A dataset class for loading images and their corresponding masks from a directory structure.
    """
    def __init__(self, path:Union[str, pathlib.Path],
                 pre_check:bool=True,
                 postprocess: Optional[Callable[[Image.Image, Image.Image],
                                               Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Args:
            path (Union[str, pathlib.Path]): Path to the directory containing the dir gt, pt and json
        """
        assert isinstance(path, (str, pathlib.Path)), "Path must be a string or pathlib.Path object"
        assert pathlib.Path(path).is_dir(), "Provided path must be a directory"

        self.path = pathlib.Path(path)
        self.dataset_name = self.path.name
        self.dataset_json = self.path.joinpath(f"{self.dataset_name}.json")

        assert self.dataset_json.exists(), "JSON file for dataset metadata does not exist"

        self.data_list:list[Dict] = []

        try:
            with open(self.dataset_json, 'r', encoding='utf-8') as f:
                self.data_list = json.load(f)
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")

        # check the data validity
        if pre_check:
            self.preCheck()

        self.postprocess = postprocess

    def preCheck(self):
        """
        Pre-check the dataset for missing files and metadata
        """
        for item in self.data_list:
            if not item.get("tp") or not item.get("gt"):
                raise ValueError(f"Missing 'tp' or 'gt' key in dataset item: {item}")
            if not self.path.joinpath("tp", item["tp"]).exists():
                raise FileNotFoundError(f"Image file {item['tp']} does not exist in {self.path.joinpath('tp')}")
            if not self.path.joinpath("gt", item["gt"]).exists():
                raise FileNotFoundError(f"Mask file {item['gt']} does not exist in {self.path.joinpath('gt')}")
        print(f"load dataset {self.dataset_name} successfully.")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            1. No postprocess
            Tuple[Image.Image, Image.Image]: A tuple containing the image and its corresponding mask
            2. Postprocess
            Tuple[np.ndarray, np.ndarray]: A tuple containing the processed image and mask as numpy arrays
        """
        # Load the image and mask
        img_path = self.path.joinpath("tp", self.data_list[idx]["tp"])
        mask_path = self.path.joinpath("gt", self.data_list[idx]["gt"])

        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
        except FileNotFoundError as e:
            print(f"File not found: {img_path} or {mask_path}")
            raise e
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            raise e

        if self.postprocess:
            img, mask = self.postprocess(img, mask)

        return img, mask

if __name__ == "__main__":
    # Example usage
    dataset = ImageDataset(r"./data/simple")
    print(f"Dataset {dataset.dataset_name} loaded with {len(dataset)} items.")
    img, mask = dataset[0]
    if isinstance(img, Image.Image):
        print(f"img size:{img.size}, img channel:{img.getbands()}")
        img.show()
    if isinstance(mask, Image.Image):
        print(f"img size:{mask.size}, img channel:{mask.getbands()}")
        mask.show()