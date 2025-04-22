from PIL import Image
import numpy as np
from pathlib import Path
from typing import Literal
import albumentations as albu

def pil_loader(path: str, mode: Literal["RGB", "L"]) -> Image.Image:
    """PIL image loader
    Args:
        path (str): image path
    Returns:
        Image.Image: PIL image (after np.array(x) becomes [0,255] int8)
    """
    assert Path(path).exists(), \
        "pil load image error, the path is not exits()"
    assert mode in {"RGB", "L"}, \
        "pil load mode error"
    img = Image.open(path)
    return img.convert(mode=mode)

def data_augmentation(aug_p=1):
    """
    Randomly applying an image augmentation method to images
    based on probability aug_p;

    augmentation : target
    jpeg_compression or gaussian_blur :only image
    vertical_flip or horizontal_flip  :image and mask both
    """
    jpeg_compression = albu.ImageCompression(quality_range=(70, 100), p=1)
    gaussian_blur = albu.GaussianBlur(blur_limit=(3, 7), p=1)
    vertical_flip = albu.VerticalFlip(p=1)
    horizontal_flip = albu.HorizontalFlip(p=1)
    return albu.OneOf([jpeg_compression, gaussian_blur,
                       vertical_flip, horizontal_flip], p=aug_p)

def padzero(scale_size=512,
            output_height=512,
            output_width=512)-> albu.Compose:
    """
    Adjust the scale of the image
    to make the maximum edge equal to the x_size,
    while maintaining the aspect ratio of the original image.
    Then fill the size of the image to the output_height, output_width
    """
    return albu.Compose([
        albu.LongestMaxSize(max_size=scale_size),
        albu.PadIfNeeded(min_height=output_height,
                         min_width=output_width,
                         border_mode=0,
                         position= 'top_left')
        ])

if __name__ == "__main__":
    """
    test
    """
    tp = Path(__file__).parent.parent.joinpath("data", "simple","tp", "Sp_D_CND_A_pla0005_pla0023_0281.jpg")
    gt = Path(__file__).parent.parent.joinpath("data", "simple","gt", "Sp_D_CND_A_pla0005_pla0023_0281_gt.png")
    tp_array = np.array(pil_loader(str(tp), "RGB"))
    gt_array = np.array(pil_loader(str(gt), "L"))
    print("tp shape:",tp_array.shape, "gt_shape",gt_array.shape)
    img = data_augmentation()(image = tp_array, mask=gt_array)
    res_tp = Image.fromarray(img['image'])
    res_gt = Image.fromarray(img['mask'])
    print("mask:", img['mask'].shape,(img['mask'] == gt_array).all())
    print("image:", img['image'].shape, (img['image'] == tp_array).all())
    img_pad = padzero()(image=img['image'], mask=img['mask'])
    res_tp = Image.fromarray(img_pad['image'])
    res_gt = Image.fromarray(img_pad['mask'])
    res_tp.save("res_tp.jpg")
    res_gt.save("res_gt.png")

