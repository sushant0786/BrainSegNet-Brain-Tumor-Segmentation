import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE=256 

train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize(), ToTensorV2(),
    ])
val_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(), ToTensorV2(),
    ])
