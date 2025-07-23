from PIL import Image
import numpy as np
import albumentations as albu

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

def augs_default():
    # geometric transform
    vertical_flip = albu.VerticalFlip(p=0.5)
    horizontal_flip = albu.HorizontalFlip(p=0.5)
    rotate = albu.RandomRotate90(p=0.5)
    # Pixel transforms
    jpeg_compression = albu.ImageCompression(quality_range=(70, 100), p=0.2)
    gaussian_blur = albu.GaussianBlur(blur_limit=(3, 7), p=0.2)
    return albu.Compose([

    ])

def postprocess(transform : albu.Compose):
    def wrapper(img:Image.Image, mask:Image.Image):
        img_array = np.array(img)
        mask_array = np.array(mask)
        if img_array.shape[-1] == 4:
            img_array = rgba2rgb(img_array)

        transformed = transform(image=img_array, mask=mask_array)

    return wrapper



if __name__ == "__main__":
    """
    test
    """
    import os
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

