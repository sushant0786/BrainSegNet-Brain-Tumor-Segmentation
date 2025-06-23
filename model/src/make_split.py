from sklearn.model_selection import train_test_split
import pandas as pd
from MRIDataset import MRIDataset
from torch.utils.data import DataLoader
from augment import train_tf, val_tf
import os
import glob
import cv2
import numpy as np

DATA_PATH = "/kaggle/input/lgg-mri-segmentation/kaggle_3m/"
data=[]
data=[]
for sub_dir_path in glob.glob(DATA_PATH+"*"):
  if os.path.isdir(sub_dir_path): ## to ignore data.csv 
    dirname=sub_dir_path.split("/")[-1]
    for filename in os.listdir(sub_dir_path):
      image_path=sub_dir_path+"/"+filename
      data.extend([dirname,image_path])
          
df=pd.DataFrame({
  "dirname":data[::2],
  "path":data[1::2]
})

#for sorting 
BASE_LEN=89      # "/data/archive/lgg-mri-segmentation/kaggle_3m\TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_"
END_IMG_LEN=4    #".tif"
END_MASK_LEN=9   #"_mask.tif"

df_images=df[~df["path"].str.contains("mask")]
df_masks=df[df["path"].str.contains("mask")]

images =sorted(df_images["path"].values ,key=lambda x:int(x[BASE_LEN:-END_IMG_LEN]))
masks =sorted(df_masks["path"].values ,key=lambda x:int(x[BASE_LEN:-END_MASK_LEN]))

df=pd.DataFrame({
  "patient":df_images.dirname.values,
  "image_path":images,
  "mask_path":masks
})

def isTumor(path):
  max_pixel=np.max(cv2.imread(path))  #  return np array cv2.imread(path) and None if wrong path
  if max_pixel>0:
    return 1       ## 0 means black , so if >0 i.e 255 return tumor
  else:
    return 0
  
df["is_tumor"]=df["mask_path"].apply(lambda x:isTumor(x))

df_images=df[~df["path"].str.contains("mask")]
df_masks=df[df["path"].str.contains("mask")]

images =sorted(df_images["path"].values ,key=lambda x:int(x[BASE_LEN:-END_IMG_LEN]))
masks =sorted(df_masks["path"].values ,key=lambda x:int(x[BASE_LEN:-END_MASK_LEN]))


#for sorting 
BASE_LEN=89      # "/data/archive/lgg-mri-segmentation/kaggle_3m\TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_"
END_IMG_LEN=4    #".tif"
END_MASK_LEN=9   #"_mask.tif"


NUM_WORKERS=4
BATCH_SIZE=32

train_df, val_df = train_test_split(df, test_size=0.10, stratify=df.is_tumor, random_state=42)
train_df, test_df = train_test_split(train_df, test_size=0.15, stratify=train_df.is_tumor, random_state=42)

train_ds = MRIDataset(train_df, train_tf)
val_ds   = MRIDataset(val_df,   val_tf)
test_ds  = MRIDataset(test_df,  val_tf)

train_dl = DataLoader(train_ds, BATCH_SIZE, True,  num_workers=NUM_WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)
test_dl  = DataLoader(test_ds,  BATCH_SIZE, False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}")                                        
