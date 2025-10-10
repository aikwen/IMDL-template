import albumentations as albu
from typing import List, Optional


def get_aug(aug_type: str, aug_param)->albu.BasicTransform:
    p = aug_param['p']
    if p < 0:
        p = 0
    elif p > 1:
        p = 1

    if aug_type == "rotate":
        # 随机旋转
        return albu.RandomRotate90(p=p)
    elif aug_type == "v_flip":
        # 垂直翻转
        return albu.VerticalFlip(p=p)
    elif aug_type == "h_flip":
        # 垂直翻转
        return albu.HorizontalFlip(p=p)
    elif aug_type == "resize":
        # 改变尺寸
        # args0 和 args1 分别表示height， width
        size = aug_param['size']
        assert isinstance(size[0], int) and isinstance(size[1], int)
        return albu.Resize(height=size[0], width=size[1], p=p)
    elif aug_type == "jpeg":
        # jpeg 压缩
        # 随机在 (args0, args1) 区间之间选择一个质量因子进行压缩
        quality_range = aug_param['quality_range']
        assert isinstance(quality_range[0], int) and isinstance(quality_range[1], int)
        return albu.ImageCompression(quality_range=(quality_range[0],
                                                    quality_range[1]), p=p)
    elif aug_type == "gblur":
        # 高斯模糊
        # 随机在 (args0, args1) 区间之间选择一个 kernal size
        kernel_range = aug_param['kernel_range']
        assert isinstance(kernel_range[0], int) and isinstance(kernel_range[1], int)
        return albu.GaussianBlur(blur_limit=(kernel_range[0], kernel_range[1]), p=p)
    elif aug_type == "gnoise":
        # 高斯噪声
        std_range = aug_param['std_range']
        return albu.GaussNoise(std_range=(std_range[0], std_range[1]), p=p)
    elif aug_type == "scale":
        # 按比例缩放
        # 随机在 (1+args0, 1+args1) 之间旋转一个值进行缩放
        scale_limit = aug_param['scale_limit']
        return albu.RandomScale(scale_limit=(scale_limit[0], scale_limit[1]), p=p)
    else:
        return albu.NoOp(p=1)

def aug_compose(aug_list:List[albu.BasicTransform]) -> Optional[albu.Compose]:
    if len(aug_list) <= 0:
        return None
    
    return albu.Compose([item for item in aug_list])