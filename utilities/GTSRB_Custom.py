from pandas import read_csv
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
import torchvision.transforms as T



class GTSRB(Dataset):
    def __init__(self, root, csv_file, transform=T.ToTensor()):
        self.root = root
        self.data = read_csv(root / csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = tensor( self.data.iloc[index, 0] )
        img = Image.open(self.root / self.data.iloc[index, 1])

        if self.transform:
            img = self.transform(img)

        return (img, label)