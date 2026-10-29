"""Train one binary U-Net. See backend/README.md for expected dataset layout."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from unet import UNet

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MasksDataset(Dataset):
    def __init__(self, task, split):
        self.images = ROOT / "dataset" / task / split / "images"
        self.masks = ROOT / "dataset" / task / split / "masks"
        self.files = sorted([p for p in self.images.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} and (self.masks / (p.stem + ".png")).exists()])
        if not self.files: raise RuntimeError(f"No image/mask pairs in {self.images}; masks must be PNG files named like the image stem.")
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]; mp = self.masks / (p.stem + ".png")
        im = Image.open(p).convert("L").resize((256,256), Image.Resampling.BILINEAR)
        ma = Image.open(mp).convert("L").resize((256,256), Image.Resampling.NEAREST)
        x = torch.from_numpy((np.asarray(im, dtype=np.float32)/255).copy()).unsqueeze(0)
        y = torch.from_numpy((np.asarray(ma)>0).astype(np.float32).copy()).unsqueeze(0)
        return x,y

def dice_loss(logits, y):
    p = torch.sigmoid(logits); dims=(1,2,3)
    dice = (2*(p*y).sum(dims)+1e-6)/(p.sum(dims)+y.sum(dims)+1e-6)
    return 1-dice.mean()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--task', choices=['vessel','lesion'], required=True); parser.add_argument('--epochs', type=int, default=30); parser.add_argument('--batch-size', type=int, default=4); args=parser.parse_args()
    tr=DataLoader(MasksDataset(args.task,'train'), batch_size=args.batch_size, shuffle=True)
    va=DataLoader(MasksDataset(args.task,'val'), batch_size=args.batch_size)
    model=UNet().to(DEVICE); opt=torch.optim.Adam(model.parameters(), lr=1e-4); bce=nn.BCEWithLogitsLoss(); best=float('inf')
    out=ROOT/'backend'/'weights'/f'{args.task}_unet.pth'; out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); train_loss=0
        for x,y in tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); logits=model(x); loss=bce(logits,y)+dice_loss(logits,y); loss.backward(); opt.step(); train_loss+=loss.item()
        model.eval(); val_loss=0
        with torch.no_grad():
            for x,y in va:
                x,y=x.to(DEVICE),y.to(DEVICE); val_loss+=(bce(model(x),y)+dice_loss(model(x),y)).item()
        val_loss/=max(1,len(va)); print(f'Epoch {epoch+1}/{args.epochs} train={train_loss/max(1,len(tr)):.4f} val={val_loss:.4f}')
        if val_loss<best: best=val_loss; torch.save(model.state_dict(),out); print('Saved best checkpoint:',out)
    print('Finished. Evaluate on a separate held-out test set before reporting metrics.')
if __name__=='__main__': main()
