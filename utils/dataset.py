from PIL import Image
from torch.utils.data import Dataset
from typing import Union, Dict, Optional, Callable, Tuple
import pathlib
import json
import numpy as np
import random
import tqdm

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
                 pre_check:bool=True,
                 split_ratio:float = 1.0,
                 seed:Optional[int] = None,
                 postprocess: Optional[Callable[[np.ndarray, np.ndarray],
                                               Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Args:
            path (Union[str, pathlib.Path]): Path to the directory containing the dir gt, pt and json
            pre_check (bool): Whether to perform pre-checks on the dataset
            split_ratio (float): Ratio of the dataset to use, default is 1 (use
                the entire dataset)
            postprocess (Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]):
                A function to post-process the images and masks after loading.
        """
        assert isinstance(split_ratio, float) and 0 < split_ratio <= 1.0, "split_ratio must be a float between 0 and 1"

        self.split_ratio = split_ratio
        self.seed = seed

        assert isinstance(path, (str, pathlib.Path)), "Path must be a string or pathlib.Path object"
        assert pathlib.Path(path).is_dir(), "Provided path must be a directory"

        self.path = pathlib.Path(path)
        self.dataset_name = self.path.name
        self.dataset_json = self.path.joinpath(f"{self.dataset_name}.json")

        assert self.dataset_json.exists(), "JSON file for dataset metadata does not exist"

        self.postprocess = postprocess

        # Initialize the dataset list and number of items
        self.dataset_list:list[Dict] = []
        self.dataset_num:int = 0
        self.init_dataset()

        # check the data validity
        if pre_check:
            self.preCheck()

    def preCheck(self):
        """
        pre-check the dataset for validity:
        1. Check if the 'tp' and 'gt' keys exist in each item
        2. Check if the image and mask files exist in the specified directories
        """
        for item in tqdm.tqdm(self.dataset_list, desc="Checking dataset validity", ascii=True):
            if "tp" not in item or "gt" not in item:
                raise ValueError(f"Missing 'tp' or 'gt' key in dataset item: {item}")
            if not self.path.joinpath("tp", item["tp"]).exists():
                raise FileNotFoundError(f"Image file {item['tp']} does not exist in {self.path.joinpath('tp')}")
            if not self.path.joinpath("gt", item["gt"]).exists() and item["gt"] != "":
                raise FileNotFoundError(f"Mask file {item['gt']} does not exist in {self.path.joinpath('gt')}")
        print(f"load dataset {self.dataset_name} successfully.")

    def init_dataset(self):
        """
        1. Initialize the dataset by loading the metadata from the JSON file and checking the files
        2. shuffle the dataset if seed is provided
        3. split the dataset if split_ratio is less than 1.0
        """

        # Load the dataset metadata from JSON file
        try:
            with open(self.dataset_json, 'r', encoding='utf-8') as f:
                self.dataset_list = json.load(f)
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")

        # shuffle the dataset
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(self.dataset_list)

        # split the dataset if split_ratio is less than 1.0
        if self.split_ratio < 1.0:
            split_index = int(len(self.dataset_list) * self.split_ratio)
            self.dataset_list = self.dataset_list[:split_index]
        self.dataset_num = len(self.dataset_list)

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed image and mask
        """
        # Load the image and mask
        img_path = self.path.joinpath("tp", self.dataset_list[idx]["tp"])
        if self.dataset_list[idx]["gt"] == "":
            mask_path = None
        else:
            mask_path = self.path.joinpath("gt", self.dataset_list[idx]["gt"])

        try:
            img = Image.open(img_path)
            if mask_path is not None:
                mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
            else:
                mask_array = np.zeros(img.size, dtype=np.uint8)
                mask = Image.fromarray(mask_array)
        except FileNotFoundError as e:
            print(f"File not found: {img_path} or {mask_path}")
            raise e
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            raise e

        # Convert images to numpy arrays
        mask_array = np.array(mask)
        if img.mode == 'RGBA':
            img.load()  # Ensure the image is loaded
            img_array = rgba2rgb(np.array(img))
        else:
            img = img.convert("RGB")
            img_array = np.array(img)
        if self.postprocess:
            img_array, mask_array = self.postprocess(img_array, mask_array)
        return img_array, mask_array

if __name__ == "__main__":
    # Example usage
    dataset = ImageDataset(r"./data/simple")
    print(f"Dataset {dataset.dataset_name} loaded with {len(dataset)} items.")
    img, mask = dataset[0]
    print(f"img shape:{img.shape}, img dtype:{img.dtype}")
    img = Image.fromarray(img)
    img.show()
    print(f"mask shape:{mask.shape}, mask dtype:{mask.dtype}")
    mask = Image.fromarray(mask)
    mask.show()