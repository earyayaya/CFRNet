"""Portable training entry point adapted from the available source script."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cfrnet import CFRNet
from inference import read_image, read_weights
from losses import JointLoss


class PairedImages(Dataset):
    """Pair exact filenames; use identical crops/flips for input and target."""
    def __init__(self, low, normal, patch):
        self.low, self.normal, self.patch = low, normal, patch
        def names(root):
            return {p.name for p in root.iterdir() if p.is_file() and
                    p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")}
        low_names, normal_names = names(low), names(normal)
        if not low_names or low_names != normal_names:
            raise ValueError("Low/normal folders must contain the same nonempty set of image filenames.")
        self.files = sorted(low_names)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = self.files[index]
        low = read_image(self.low / name)[0]
        normal = read_image(self.normal / name)[0]
        if low.shape != normal.shape:
            raise ValueError(f"Paired image sizes differ: {name}")
        height, width = low.shape[-2:]
        if min(height, width) < self.patch:
            raise ValueError(f"Image smaller than --patch: {name}")
        top = torch.randint(height - self.patch + 1, ()).item()
        left = torch.randint(width - self.patch + 1, ()).item()
        low = low[:, top:top+self.patch, left:left+self.patch]
        normal = normal[:, top:top+self.patch, left:left+self.patch]
        for dimension in (1, 2):
            if torch.rand(()) < 0.5:
                low, normal = low.flip(dimension), normal.flip(dimension)
        return low, normal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--init", type=Path, help="Backbone or full CFRNet weights; not optimizer resume")
    initialization.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--steps", type=int, required=True, help="Set the full cosine schedule explicitly")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.steps, args.width, args.batch_size, args.patch, args.save_every) < 1:
        parser.error("Steps, width, batch size, patch size and save interval must be positive.")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu for a small execution check.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(2)
    model = CFRNet(width=args.width)
    torch.nn.init.zeros_(model.cfrb.fusion.weight)
    torch.nn.init.zeros_(model.cfrb.fusion.bias)
    if args.init:
        missing, unexpected = model.load_state_dict(read_weights(args.init), strict=False)
        allowed = {key for key in model.state_dict() if key.startswith("cfrb.")}
        if unexpected or set(missing) not in (set(), allowed):
            raise RuntimeError(f"Incompatible initialization: missing={missing}; unexpected={unexpected}")
        print("Loaded full CFRNet weights." if not missing else "Loaded backbone; CFRB is newly initialized.")
    dataset = PairedImages(args.low, args.normal, args.patch)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=args.device == "cuda")
    model = model.to(args.device).train()
    criterion = JointLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-7)
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    step = 0
    while step < args.steps:
        for low, normal in loader:
            if step >= args.steps:
                break
            low, normal = low.to(args.device), normal.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(low), normal)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % 100 == 0:
                print(f"step={step} loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.3g}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                torch.save(model.state_dict(), args.output / f"cfrnet_step_{step}.pth")


if __name__ == "__main__":
    main()
