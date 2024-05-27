from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
from pathlib import Path

class Datasets(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = Path(data_dir)
    
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "**/*.JPEG"),recursive=True))
        self.transforms = transforms
    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB') 
        return self.transforms(image)
    def __len__(self):
        return len(self.image_path)


def get_loader(train_data_dir, test_data_dir, image_size, batch_size):
    train_dataset = Datasets(train_data_dir, image_size)
    test_dataset = Datasets(test_data_dir, image_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader




class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)
