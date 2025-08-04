
from torch.utils.data.dataset import Dataset
import cv2


class MRIDataset(Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df  = dataframe.reset_index(drop=True)
        self.tfm = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img = cv2.imread(self.df.loc[idx, 'image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  ## cv2 read image in BGR format
        msk = cv2.imread(self.df.loc[idx, 'mask_path'], cv2.IMREAD_GRAYSCALE)

        if self.tfm:
            out   = self.tfm(image=img, mask=msk)
            img, msk = out['image'], out['mask']

        
        msk = msk.float().unsqueeze(0) / 255.  ## make pixels value from 0 -1 
        return img, msk
    
