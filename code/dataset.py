import numpy as np
import pdb
import os
from torch.utils.data import Dataset
from PIL import Image

class CatDogDataset(Dataset):

    def __init__(self, train_dir, transform = None):
        
        self.train_dir = train_dir
        self.transform = transform
        self.images = os.listdir(train_dir)
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, self.images[index])
        label = self.images[index].split(".")[0]

        label = 0 if label == 'cat' else 1
        
        image = np.array(Image.open(image_path))
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label
