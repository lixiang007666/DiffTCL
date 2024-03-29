import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import albumentations as A


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask



def crop_od(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    # half_area = 2500
    half_area = 3500

    while True:  # Keep trying until a suitable crop is found
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img_cropped = img.crop((x, y, x + size, y + size))
        mask_cropped = mask.crop((x, y, x + size, y + size))

        mask_array = np.array(mask_cropped)
        mask_nonzero_area = np.count_nonzero(mask_array)

        if mask_nonzero_area >= half_area:
            return img_cropped, mask_cropped


# 修改上面的代码，要求相应mask裁剪出的区域非0像素点面积占一半以上。
def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def normalize_l(img, mask=None):
    img = transforms.Compose([
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def elastic_deform(img, alpha_range, sigma_range):
    img_array = np.array(img)  # Convert PIL Image to NumPy array
    
    # Generate a random alpha value from the specified range
    alpha = float(np.random.uniform(alpha_range[0], alpha_range[1]))
    
    # Define an elastic deformation transformation
    transform = A.ElasticTransform(alpha=alpha, sigma=10, alpha_affine=0, p=1)
    
    # Apply the transformation
    augmented = transform(image=img_array)
    deformed_img = augmented['image']
    
    return Image.fromarray(deformed_img.astype(np.uint8))  # Convert back to PIL Image

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
