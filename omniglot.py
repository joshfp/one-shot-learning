import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder


class OmniglotDataset(Dataset):
    """
    Load Omniglot dataset from a path.
    
    Attributes:
        images (FloatTensor): Vectorized images (N x 1 x img_sz x img_sz).
        labels (LongTensor): Label of each image (N).
    """
    
    def __init__(self, path, img_sz=28):
        """
        Args:
            path (:obj:´Path´): Path to directory.
            img_sz (int, optional): Image size for resizing.
        """
        super().__init__()
        self.images = []
        self.labels = []
        tfms = transforms.Compose([
            transforms.Resize(img_sz),
            transforms.ToTensor(),
            #transforms.Normalize((0.9221,), (0.2681,))
        ]) 

        for file in list(path.glob('*/*/*')):
            for rot in [0, 90, 180, 270]:          
                img = Image.open(file).rotate(rot)
                self.images.append(tfms(img))
                self.labels.append(f'{file.parents[1].name}/{file.parent.name}/{rot}')   
        self.images = torch.stack(self.images)                
        self.labels = torch.from_numpy(LabelEncoder().fit_transform(self.labels))        
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Return items by index."""
        return self.images[idx], self.labels[idx]