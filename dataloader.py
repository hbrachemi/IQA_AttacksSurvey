from skimage import io, transform
import numpy as np
import pickle 
import os 
import torch 
from torch.utils.data import Dataset

class iqa_dataset(Dataset):
    """dataset."""

    def __init__(self,labels_path='scores.pickle' ,db_path='',
                 ids_path='IDs.pickle',part='', transform=None):
        """
        Args:
            labels_path (string): Path to the pickle file dictionnary containing scores.
            db_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.labels_path=labels_path
        self.db_path=db_path
        self.ids_path=ids_path
        self.part=part
        self.transform=transform
        
        self.ids_path=self.ids_path.replace('.pickle','')
        self.ids_path=self.ids_path+"_"+str(part)+".pickle"
        pickle_in = open(self.ids_path,'rb')
        self.list_IDs= pickle.load(pickle_in)
        self.list_IDs=list(self.list_IDs)
        pickle_in.close()

        pickle_in = open(self.labels_path,'rb')
        self.labels = pickle.load(pickle_in)
        pickle_in.close()
        
        
    def __len__(self):
        return int(len(self.list_IDs))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.db_path,
                                self.list_IDs[idx])
        image = io.imread(img_name)
        image = np.array(image)
        image = image.astype('float32')
        image = image / 255

        if self.transform:
            image = self.transform(image)
        

        label = float(self.labels[self.list_IDs[idx]])
     
        return image,label
