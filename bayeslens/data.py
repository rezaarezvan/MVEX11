import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class SSLADDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            split (string): 'train' or 'val' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images, self.annotations = self._load_annotations(split)

    def _load_annotations(self, split):
        annotation_path = os.path.join(
            self.root_dir, 'SSLAD-2D', 'labeled', 'annotations', f'instance_{split}.json')
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

        img_path = os.path.join(self.root_dir, 'SSLAD-2D',
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


def load_data(dataset_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train = SSLADDataset(
        root_dir=dataset_path, split='train', transform=transform)
    val = SSLADDataset(
        root_dir=dataset_path, split='val', transform=transform)

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = DataLoader(val, batch_size=batch_size, shuffle=False)

    return train, val
