import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
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


def load_data(dataset_path, batch_size=32, transform=DEFAULT_TRANSFORM):
    train = SODADataset(
        root_dir=dataset_path, split='train', transform=transform)
    val = SODADataset(
        root_dir=dataset_path, split='val', transform=transform)
    test = SODADataset(
        root_dir=dataset_path, split='test', transform=transform)

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = DataLoader(val, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train, val, test
