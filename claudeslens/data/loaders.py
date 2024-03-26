import os
import json
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DEFAULT_TRANSFORM_SODA_VIT = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

DEFAULT_TRANSFORM_SODA = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

DEFAULT_TRANSFORM_MNIST = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DEFAULT_TRANSFORM_MNIST_VIT = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class SODADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            split (string): 'train' or 'val' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.subdir = 'labeled_trainval' if split in [
            'train', 'val'] else 'labeled_test'
        self.transform = transform
        self.annotations_file = f'instance_{split}.json'
        self.images, self.annotations = self._load_annotations()

    def _load_annotations(self):
        annotation_path = os.path.join(
            self.root_dir, self.subdir, 'SSLAD-2D', 'labeled', 'annotations', self.annotations_file)

        load_categories(annotation_path)
        with open(annotation_path) as f:
            data = json.load(f)

        images = {image['id']: image for image in data['images']}
        annotations = {anno['image_id']: anno for anno in data['annotations']}
        return images, annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_info = self.images[img_id]
        img_filename = img_info['file_name']

        img_path = os.path.join(self.root_dir, self.subdir, 'SSLAD-2D',
                                'labeled', self.split, img_filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if img_id in self.annotations:
            anno = self.annotations[img_id]
            target = anno['category_id'] - 1
        else:
            target = 0

        target = torch.tensor(target, dtype=torch.long)

        return image, target


def load_categories(annotation_path):
    with open(annotation_path) as f:
        data = json.load(f)
    categories = data['categories']
    category_mapping = {category['id']: category['name']
                        for category in categories}
    return category_mapping


def load_SODA(dataset_path, batch_size=32, ViT=False):
    train = SODADataset(
        root_dir=dataset_path, split='train', transform=DEFAULT_TRANSFORM_SODA_VIT if ViT else DEFAULT_TRANSFORM_SODA)
    val = SODADataset(
        root_dir=dataset_path, split='val', transform=DEFAULT_TRANSFORM_SODA_VIT if ViT else DEFAULT_TRANSFORM_SODA)
    test = SODADataset(
        root_dir=dataset_path, split='test', transform=DEFAULT_TRANSFORM_SODA_VIT if ViT else DEFAULT_TRANSFORM_SODA)

    train_batch = batch_size
    val_batch = batch_size
    test_batch = batch_size
    if not isinstance(batch_size, int):
        train_batch = len(train)
        val_batch = len(val)
        test_batch = len(test)

    train = DataLoader(train, batch_size=train_batch,
                       shuffle=True, num_workers=4, pin_memory=True)
    val = DataLoader(val, batch_size=val_batch, shuffle=True,
                     num_workers=4, pin_memory=True)
    test = DataLoader(test, batch_size=test_batch,
                      shuffle=True, num_workers=4, pin_memory=True)

    return train, val, test


def load_MNIST(root_dir='../extra/datasets', batch_size=16, ViT=False):
    train = torchvision.datasets.MNIST(
        root=root_dir, train=True, download=True, transform=DEFAULT_TRANSFORM_MNIST_VIT if ViT else DEFAULT_TRANSFORM_MNIST)
    test = torchvision.datasets.MNIST(
        root=root_dir, train=False, download=True, transform=DEFAULT_TRANSFORM_MNIST_VIT if ViT else DEFAULT_TRANSFORM_MNIST)

    train = DataLoader(train, batch_size=batch_size,
                       shuffle=True, num_workers=4, pin_memory=True)
    test = DataLoader(test, batch_size=batch_size,
                      shuffle=False, num_workers=4, pin_memory=True)

    return train, test
