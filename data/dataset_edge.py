from torch.utils.data import Dataset
import torchvision
from PIL import Image
import glob

class Dataset_edge(Dataset):
    def __init__(self, files, transform=None):
        #glob.glob找到所哟
        image_path1 = glob.glob(files + '/A' + '/*.png')
        image_path1.sort()
        self.image_path1 = image_path1
        image_path2 = glob.glob(files + '/B' + '/*.png')
        image_path2.sort()
        self.image_path2 = image_path2
        target = glob.glob(files + '/label' + '/*.png')
        target.sort()
        self.target = target
        # label_edge
        target_edge = glob.glob(files + '/label_edge' + '/*.png')
        target_edge.sort()
        self.target_edge = target_edge
        self.transform = transform

    def __len__(self):
        #如果长度不相等，则抛出断言
        #len和getitem 继承dataset必须写
        assert len(self.image_path1) == len(self.image_path2)
        return len(self.image_path1)

    def __transform(self, A, B, Label, Label_edge=None):
        for t in self.transform.transforms:
            if (not isinstance(t, torchvision.transforms.transforms.ColorJitter)
                    or isinstance(t, torchvision.transforms.transforms.RandomGrayscale)
                        or (isinstance(t, torchvision.transforms.transforms.ToTensor)) and isinstance(t, torchvision.transforms.transforms.Normalize)):
                Label = t(Label)
                Label_edge = t(Label_edge)
            A = t(A)
            B = t(B)
        return {'A': A, 'B': B, 'Label': Label, 'Label_edge': Label_edge}

    def __getitem__(self, idx):
        images1 = Image.open(self.image_path1[idx])
        images2 = Image.open(self.image_path2[idx])
        mask = Image.open(self.target[idx])
        mask_edge = Image.open(self.target_edge[idx])
        #前面只是对预处理方式的定义，在这里执行
        #sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label, 'label_edge': trf_label_edge}
        return self.__transform(images1, images2, mask, mask_edge)
