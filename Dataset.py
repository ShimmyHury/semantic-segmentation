import os
import cv2
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import numpy as np


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image = np.array(image).astype(np.float16) / 255.0
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        albu.augmentations.geometric.resize.Resize(height=256, width=384, always_apply=True),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            albu.GridDistortion(p=0.5),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)], p=0.8),
        albu.CLAHE(p=0.8),
        albu.RandomBrightnessContrast(p=0.8),
        albu.RandomGamma(p=0.8)
    ]

    return albu.Compose(train_transform, p=0.9)


def get_validation_augmentation():
    """Resize to make image shape divisible by 32"""
    test_transform = [
        albu.augmentations.geometric.resize.Resize(height=256, width=384, always_apply=True)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(mean, std):
    """Construct preprocessing transform

    Args:
        mean and std of data
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.augmentations.transforms.Normalize(mean=mean, std=std, max_pixel_value=1.0, always_apply=False, p=1.0),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def calc_mean_std(img_path):
    ids = os.listdir(img_path)
    images_fps = [os.path.join(img_path, image_id) for image_id in ids]
    images_all = []
    for i in range(len(images_fps)):
        image = cv2.imread(images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        images_all.append(image)
    images_all = np.stack(images_all, axis=0)
    mean, std = np.mean(images_all, axis=(0, 1, 2)), np.std(images_all, axis=(0, 1, 2))
    del images_all
    return mean, std
