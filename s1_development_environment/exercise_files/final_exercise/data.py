import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MyDataset(Dataset): 
    def __init__(self, data_path, data_type):
        super().__init__()
        self.images, self.labels = self._load_data(data_path, data_type)

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    
    def _load_data(self, data_path, data_type):
        data = []
        # Check if type is passed correctly
        if ((data_type in "train") == False) & ((data_type in "test") == False): 
            print("Wrong type, choose between train and test")
            return

        for file in os.listdir(data_path):
            if (data_type in file) == True:
                data.append(np.load(data_path+"/"+file))
        images = torch.tensor(np.concatenate([d['images'] for d in data]), dtype=torch.float32).reshape(-1, 1, 28, 28)
        labels = torch.tensor(np.concatenate([d['labels'] for d in data]))

        return images, labels




if __name__ == '__main__':
    data_path = "C:/Users/victo/OneDrive/Escritorio/DTU/Machine_Learning_Operations/dtu_mlops/data/corruptmnist"

    train_set = MyDataset(data_path, 'train')

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    for img, label in train_loader: 
        print(img.shape)