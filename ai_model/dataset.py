import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RoomDataset(Dataset):
    def __init__(self, root_dir):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")

        valid_ext = ('.jpg', '.jpeg', '.png')

        self.images = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(valid_ext)
            and os.path.exists(os.path.join(self.target_dir, f))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)

        try:
            input_img = Image.open(input_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx + 1) % len(self.images))

        return self.transform(input_img), self.transform(target_img)
