from PIL import Image
import numpy as np
import albumentations as albu
from typing import Tuple, Union,List , TypeAlias

Limits: TypeAlias = Union[tuple[int, int], tuple[float, float], None]

class AUGTYPE:
    def __init__(self, type: str, p: float, r: Limits=None):
        self.type = type
        self.p = p
        self.r = r
        self.__type = (
            "rotate",
            "v_flip",
            "h_flip",
            "resize",
            "jpeg",
            "gblur",
            "gnoise",
            "scale")
        self.__check(type, p, r)

    def __check(self, t:str, p:float, r:Limits=None):
        assert isinstance(t, str)
        assert t in self.__type, f"类型必须是{self.__check}中的一种"
        assert isinstance(p, float)
        assert p>=0 and p<=1, "概率 p 必须大于等于 0 小于等于 1"
        assert r is None or (isinstance(r, tuple) and len(r) == 2)
        if t in ("resize","jpeg","gblur","gnoise","scale"):
            assert r is not None and isinstance(r, tuple) and len(r) == 2
            if t in ("resize","jpeg", "gblur"):
                assert isinstance(r[0], int) and isinstance(r[1], int)


def get_aug(aug_type: AUGTYPE):
    p = aug_type.p
    assert aug_type.r is not None
    args0, args1 = aug_type.r
    if aug_type.type   == "rotate":
        # 随机旋转
        return albu.RandomRotate90(p = p)
    elif aug_type.type == "v_flip":
        # 垂直翻转
        return albu.VerticalFlip(p = p)
    elif aug_type.type == "h_flip":
        # 垂直翻转
        return albu.HorizontalFlip(p = p)
    elif aug_type.type == "resize":
        # 改变尺寸
        # args0 和 args1 分别表示height， width
        assert isinstance(args0, int) and isinstance(args1, int)
        return albu.Resize(height=args0, width=args1, p=p)
    elif aug_type.type == "jpeg":
        # jpeg 压缩
        # 随机在 (args0, args1) 区间之间选择一个质量因子进行压缩
        assert isinstance(args0, int) and isinstance(args1, int)
        return albu.ImageCompression(quality_range=(args0, args1), p=p)
    elif aug_type.type == "gblur":
        # 高斯模糊
        # 随机在 (args0, args1) 区间之间选择一个 kernal size
        assert isinstance(args0, int) and isinstance(args1, int)
        return albu.GaussianBlur(blur_limit=(args0, args1), p=p)
    elif aug_type.type == "gnoise":
        # 高斯噪声
        #
        return albu.GaussNoise(std_range=(args0, args1), p=p)
    elif aug_type.type == "scale":
        # 按比例缩放
        # 随机在 (1+args0, 1+args1) 之间旋转一个值进行缩放
        return albu.RandomScale(scale_limit=(args0, args1), p=p)
    else:
        return albu.NoOp(p=1)

def aug_compose(aug_list:List[AUGTYPE]) -> albu.Compose:
    return albu.Compose([get_aug(aug_type=item) for item in aug_list])