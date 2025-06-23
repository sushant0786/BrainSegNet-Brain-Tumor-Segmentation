import torch
from unet_parts import DoubleConv,DownSample,UpSample
from unet import UNet
best_model_PATH = "../models/unet_best.pth"  
from MRIDataset import MRIDataset
from loss_fun_helper import dice_coef, criterion
from make_split import test_dl


model= UNet(in_channels=3, num_classes=1)  
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## loading model
best_model=torch.load(best_model_PATH,map_location=DEVICE)
model.load_state_dict(best_model)


print("Unet model inference")

model.eval()

test_loss, test_dice, num_samples = 0.0, 0.0, 0

with torch.no_grad():
    for step, (imgs, msks) in enumerate(test_dl, 1):
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)

        logits = model(imgs)
        probs  = torch.sigmoid(logits)

        loss = criterion(logits, msks)
        dice = dice_coef(probs, msks)

        bs = imgs.size(0)           # current batch size
        test_loss += loss.item()  * bs
        test_dice += dice.item()  * bs
        num_samples += bs

        print(f"Step {step:03d}/{len(test_dl)} | "
              f"Loss {loss.item():.4f} | Dice {dice.item():.4f}")

test_loss /= num_samples
test_dice /= num_samples
print(f"\nTest Loss: {test_loss:.4f} | Test Dice: {test_dice:.4f}")

    
        
