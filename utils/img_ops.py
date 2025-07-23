from PIL import Image
import numpy as np
import albumentations as albu
from typing import Tuple, Optional


def augs_default(prob_Spatial=0.5, prob_Pixel=0.2, resize: Optional[Tuple[int, int]] = None):
    # geometric transform
    rotate = albu.RandomRotate90(p=prob_Spatial)
    vertical_flip = albu.VerticalFlip(p=prob_Spatial)
    horizontal_flip = albu.HorizontalFlip(p=prob_Spatial)
    # Pixel transforms
    jpeg_compression = albu.ImageCompression(quality_range=(70, 100), p=prob_Pixel)
    gaussian_blur = albu.GaussianBlur(blur_limit=(3, 7), p=prob_Pixel)
    # resize transform
    crop = albu.Resize(height=resize[0], width=resize[1], p=1.0) if resize else albu.NoOp(p=1.0)
    return albu.Compose([
        rotate,
        vertical_flip,
        horizontal_flip,
        jpeg_compression,
        gaussian_blur,
        crop,
    ])

def postprocess(transform : albu.Compose):
    def wrapper(img_array:np.ndarray, mask_array:np.ndarray):
        """
        args:
            img (np.ndarray): input image
            mask (np.ndarray): input mask
        returns:
            img_transformed (np.ndarray): transformed image
            mask_transformed (np.ndarray): transformed "prob" mask

        """
        # Apply the transformations
        transformed = transform(image=img_array, mask=mask_array)
        img_transformed = transformed['image']
        mask_transformed = transformed['mask']

        # Convert mask to probability mask
        if mask_transformed.max() > 1.0:
            mask_transformed = mask_transformed.astype(np.float32) / 255.0
            mask_transformed[mask_transformed > 0.5] = 1.0
            mask_transformed[mask_transformed <= 0.5] = 0.0
        return img_transformed, mask_transformed
    return wrapper



if __name__ == "__main__":
    """
    test
    """
    from utils.dataset import ImageDataset
    import matplotlib.pyplot as plt
    dataset = ImageDataset(r"./data/simple")
    print(f"Dataset {dataset.dataset_name} loaded with {len(dataset)} items.")
    img, mask = dataset[1]
    images = [img, mask]
    titles = ['Image', 'Mask']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    print(f"img shape:{img.shape}")
    print(f"mask shape:{mask.shape}")
    postprocess_fn = postprocess(augs_default(prob_Spatial=1, prob_Pixel=1, resize=(256,256)))
    img, mask = postprocess_fn(img, mask)
    images.append(img)
    images.append(mask)
    titles.append('Transformed Image')
    titles.append('Transformed prob Mask')
    print("=================================================")
    for ax, img, title in zip(axes.flatten(), images, titles):
        img_array = np.asarray(img)
        print(f"{title} shape: {img_array.shape}, pixel range: {img_array.min()} - {img_array.max()}")
        ax.imshow(img_array, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

