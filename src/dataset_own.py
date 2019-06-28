import torch
import torch.utils.data as data_utl

from src import transforms_own
from src.transforms_own import *
from torchvision import transforms


class Dataset_own(data_utl.Dataset):
    # todo

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data_length = 100000

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):

        # Dummy data for temporal running
        frames, label = np.zeros([16,240,320,3]), np.asarray(0)
        frames = np.asarray(self.transform(frames))

        # Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (C x T x H x W)
        return torch.from_numpy(frames.transpose([3,0,1,2])).float(), torch.from_numpy(label)



if __name__ == '__main__':

    # Test your own Dataset
    train_transforms = transforms.Compose([transforms_own.CenterCrop(224), transforms_own.Scale(112)])
    dataset_nasmo = Dataset_own(root='/', transform=train_transforms)
    frames, label = dataset_nasmo[0]
    print(np.shape(frames), (label))
