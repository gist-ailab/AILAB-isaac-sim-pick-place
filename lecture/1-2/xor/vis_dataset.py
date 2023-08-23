import numpy as np
from torch.utils.data import Dataset
import torch

class VIS_DATASET(Dataset):
    def __init__(self, a):
        a = np.array(a)
        coor = []
        for k in range(1,3):
            init_coor = np.where(a==k)
            for i in range(len(init_coor[0])):
                coor.append([[init_coor[1][i]/(len(a)), init_coor[0][i]/(len(a))],k-1])

        self.coor = coor
    
    def __len__(self):
        return len(self.coor)
        
    def __getitem__(self,idx):
        data = torch.tensor(self.coor[idx][0]).float()
        label = torch.tensor([self.coor[idx][1]]).float()
        return data, label
    
class DOT_DATASET(Dataset):
    def __init__(self, num):
        a = torch.tensor([[[i/(num),j/(num)] for i in range(num)] for j in range(num)])
        self.a = a
        self.num = num
    
    def __len__(self):
        return len(self.a) * len(self.a[0])
        
    def __getitem__(self,idx):
        data = self.a[int(idx/self.num)][int(idx%self.num)]
        return data
    