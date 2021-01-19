import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class img_data(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root,"%s/ground" % mode)+"/*.*")) 
        self.files_B = sorted(glob.glob(os.path.join(root,"%s/noise20&20" % mode)+"/*.*")) 

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index%len(self.files_A)])
        image_B = Image.open(self.files_B[index%len(self.files_B)])

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"groundtruth": item_A, "noise": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))