"# -- coding: UTF-8 --"
import sys
sys.path.append('/mnt/nfs/data/home/1120241486/ZZHNet/')

from torch.utils.data import Dataset
import torchvision
from data import helper_augmentations
from PIL import Image
import glob

class Dataset(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, files, transform=None):

        image_path1 = glob.glob(files + '/A' + '/*.png')
        image_path1.sort()
        self.image_path1 = image_path1
        image_path2 = glob.glob(files + '/B' + '/*.png')
        image_path2.sort()
        self.image_path2 = image_path2
        target = glob.glob(files + '/label' + '/*.png')
        target.sort()
        self.target = target
        self.transform = transform

    def __len__(self):
        # return len(self.data_dict)
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        mask = Image.open(self.target[idx])
        sample = {'A': images1, 'B': images2, 'label': mask}
        # Handle Augmentations
        if self.transform:
            A = sample['A']
            B = sample['B']
            label = sample['label']
            # Dont do Normalize on label, all the other transformations apply...
            for t in self.transform.transforms:
                if (isinstance(t, helper_augmentations.SwapReferenceTest)) or (isinstance(t, helper_augmentations.JitterGamma)):
                    A, B = t(sample)
                else:
                    # All other type of augmentations
                    A = t(A)
                    B = t(B)
                
                # Don't Normalize or Swap
                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    # ToTensor divide every result by 255
                    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#to_tensor
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        label = t(label) * 255.0
                    else:
                        if not isinstance(t, helper_augmentations.SwapReferenceTest):
                            if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                                if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                    if not isinstance(t, helper_augmentations.JitterGamma):
                                        label = t(label)
                              
            sample = {'A': A, 'B': B, 'label': label}

        return sample