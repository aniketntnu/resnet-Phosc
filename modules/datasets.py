import os

import torch
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from utils import generate_phoc_vector, generate_phos_vector

import pandas as pd
import numpy as np


class phosc_dataset(Dataset):
    def __init__(self, csvfile, root_dir, transform=None):
        self.df_all = pd.read_csv(csvfile)

        print("\n\t1.no of rows:",self.df_all.shape)
        self.df_all = self.df_all[self.df_all['Word'].notna()]

        print("\n\t2.no of rows:",self.df_all.shape,"\t columns:",self.df_all.columns)
        #self.df_all = self.df_all.dropna(subset = ["Word"], inplace=True)        
        #print("\n\t3.no of rows:",self.df_all.shape,"\t columns:",self.df_all.columns)

        self.root_dir = root_dir
        self.transform = transform

        words = self.df_all["Word"].values
        #print("\n\t words:",words[:10])
        phos_vects = []
        phoc_vects = []
        phosc_vects = []

        for word in words:

            #print("\n\t word:",word)
            phos = generate_phos_vector(word)
            phoc = np.array(generate_phoc_vector(word))
            phosc = np.concatenate((phos, phoc))

            phos_vects.append(phos)
            phoc_vects.append(phoc)
            phosc_vects.append(phosc)

        self.df_all["phos"] = phos_vects
        self.df_all["phoc"] = phoc_vects
        self.df_all["phosc"] = phosc_vects

        # print(self.df_all)

        # print(self.df_all.iloc[0, 5].shape)
        # print(self.df_all.to_string())

    def __getitem__(self, index):

        try:
            img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
            image = io.imread(img_path)

            image=resize(image, (50, 250))

        except Exception as e:
            
            img_path = os.path.join(self.root_dir, self.df_all.loc[index,"cropName2"])
            image = io.imread(img_path)
            image=resize(io.imread(img_path), (50, 250))

        #y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1])

        y = torch.tensor(self.df_all.loc[index,"phosc"])


        if self.transform:
            image = self.transform(image)

        #return image.float(), y.float(), self.df_all.iloc[index, 1]

        return image.float(), y.float(), self.df_all.loc[index,"Word"]


    def __len__(self):
        return len(self.df_all)


class CharacterCounterDataset(Dataset):
    def __init__(self, longest_word_len, csvfile, root_dir, transform=None):
        self.df_all = pd.read_csv(csvfile)
        self.root_dir = root_dir
        self.transform = transform

        words = self.df_all["Word"].values

        targets = []

        for word in words:
            target = np.zeros((longest_word_len))
            target[len(word)-1] = 1
            targets.append(target)

        self.df_all["target"] = targets

        # print(self.df_all)

        # print(self.df_all.iloc[0, 5].shape)
        # print(self.df_all.to_string())

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df_all.iloc[index, 0])
        image = io.imread(img_path)

        y = torch.tensor(self.df_all.iloc[index, len(self.df_all.columns) - 1])

        if self.transform:
            image = self.transform(image)

        # returns the image, target vector and the corresponding word
        return image.float(), y.float(), self.df_all.iloc[index, 1]

    def __len__(self):
        return len(self.df_all)


if __name__ == '__main__':
    from torchvision.transforms import transforms

    dataset = CharacterCounterDataset(17, 'image_data/IAM_Data/IAM_valid_unseen.csv', 'image_data/IAM_Data/IAM_valid', transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, 512)
    # print(dataset.df_all)


    for img, target, word in dataloader:
        print(img.shape)
        print(target.shape)
        print('word')
        quit()

    # print(dataset.__getitem__(0))
