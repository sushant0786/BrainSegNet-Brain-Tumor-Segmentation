import torch
from unet_parts import DoubleConv,DownSample,UpSample
from unet import UNet
best_model_PATH = "../models/unet_best.pth"  
from MRIDataset import MRIDataset
from loss_fun_helper import dice_coef, criterion
from make_split import train_dl,val_dl,train_ds,val_ds

IMG_SIZE      = 256
BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-5


model= UNet(in_channels=3, num_classes=1)  
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


optimizer = torch.optim.AdamW(model.parameters(),lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=3)

EPOCHS            = 60          # maximum epochs
ES_PATIENCE       = 6           # stop after 6 epochs without val-Dice gain
best_val          = 0.0
epochs_no_improve = 0
SAVE_DIR      = '../models/'

for epoch in range(1, EPOCHS + 1):


    model.train()
    tot_loss, tot_dice = 0.0, 0.0

    for step, (imgs, msks) in enumerate(train_dl, 1):
        imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, msks)
        loss.backward()
        optimizer.step()

        batch_dice = dice_coef(torch.sigmoid(logits), msks).item()
        tot_loss  += loss.item() * imgs.size(0)
        tot_dice  += batch_dice   * imgs.size(0)


        print(f"Epoch {epoch:02d} | Step {step:03d}/{len(train_dl)} "
              f"| Loss {loss.item():.4f} | Dice {batch_dice:.4f}")

    train_loss = tot_loss / len(train_ds)
    train_dice = tot_dice / len(train_ds)

    #VALIDATION
    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for imgs, msks in val_dl:
            imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
            probs = torch.sigmoid(model(imgs))
            val_dice += dice_coef(probs, msks).item() * imgs.size(0)
    val_dice /= len(val_ds)

    scheduler.step(val_dice)

    print(f"[{epoch:02d}]  train_loss={train_loss:.4f}  "
          f"train_dice={train_dice:.4f}  val_dice={val_dice:.4f}")

    #CHECKPOINTING
    torch.save({"epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "val_dice":    val_dice},
               f"{SAVE_DIR}/unet_epoch_{epoch}.pth")

    if val_dice > best_val:            # new best model
        best_val = val_dice
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_best.pth")
        print(f"  ↳ NEW BEST (val Dice = {best_val:.4f})")
        epochs_no_improve = 0
    else:                              # no improvement
        epochs_no_improve += 1
        print(f"  ↳ no improvement for {epochs_no_improve} epoch(s)")

    #EARLY STOPPING
    if epochs_no_improve >= ES_PATIENCE:
        print(f"\nEarly stopping triggered — "
              f"no val-Dice improvement in {ES_PATIENCE} consecutive epochs.")
        break
