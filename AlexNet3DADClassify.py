import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import os
import numpy as np

# Define the 3D AlexNet model
class AlexNet3DADClassify(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet3DADClassify, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Dataset class for ADNI MRI scans
class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.nii') or fname.endswith('.nii.gz')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = nib.load(img_path).get_fdata()
        image = np.expand_dims(image, axis=0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        label = 1 if 'AD' in img_path else 0  # Simple assumption: filenames with 'AD' are positive cases
        return image, label

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Set up data loader
def get_loader(root_dir, batch_size=4, transform=None):
    dataset = ADNIDataset(root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Example usage
if __name__ == "__main__":
    data_dir = 'path/to/your/data'  # Update this path
    model = AlexNet3DADClassify(num_classes=2)
    data_loader = get_loader(data_dir, batch_size=4, transform=transform)

    # Example: Iterate over data
    for images, labels in data_loader:
        outputs = model(images.float())  # Make sure input is float
        print(outputs)
