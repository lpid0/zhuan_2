import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, matching_coords, non_matching_coords):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.matching_coords = matching_coords
        self.non_matching_coords = non_matching_coords

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Extract index from the image name
        index_str = img_name.split('_')[-1].split('.')[0]
        if 'index' in index_str:
            index_str = index_str.replace('index', '')
        img_id = int(index_str)

        # Determine if the image is a matching image
        is_matching = 1 if img_id in [0, 8] else 0

        # Select a non-matching image
        non_matching_coord = self.non_matching_coords[img_id % 9]
        img2 = img.crop(non_matching_coord).copy()

        img = self.transform(img)
        img2 = self.transform(img2)

        return img, img2, torch.tensor(is_matching, dtype=torch.float32)


