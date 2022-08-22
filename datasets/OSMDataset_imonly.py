import os
import numpy as np
import torch

from torch.utils.data import Dataset

class OSMDataset_imonly(Dataset):
    def __init__(
        self, 
        root_dir, 
        transforms=None,
        ):

        self.root_dir = root_dir
        self.done = set(os.listdir(self.output_dir))
        self.all = set(os.listdir(self.root_dir))
        self.inputs = list(self.all - self.done)
        self.transforms = transforms


    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, index):
        image_filename = self.inputs[index]

        # Load image
        image = np.load(os.path.join(self.root_dir, image_filename))
        image = torch.Tensor(image).permute(2, 0, 1)[None, :3, :, :] ##Converts to 1,C,H,W -- the NAIP imagery is RGBA, so need to index up to 3

        # Apply transforms if any
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)

        # Convert image to [0 1] and C, H, W
        image = image.squeeze()
        image = image.float() / 255.0 # Converts image from [0 255] to [0 1] fp

        batch = {'image': image, 'name': [image_filename.replace(".npy", "")]}

        return batch
        