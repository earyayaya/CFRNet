# CFRNet

Code for **CFRNet: Conjugate-Coupled Spectral Recalibration for Low-Light Image Restoration**.

## Installation

Python 3.10-3.12.

```bash
pip install -r requirements.txt
```

## Pretrained models

The checkpoint files are in the repository root.

| File | Dataset |
|---|---|
| `cfrnet_lolv2_real.pth` | LOL-v2-Real |
| `cfrnet_lolv2_synthetic.pth` | LOL-v2-Synthetic |

## Inference

```bash
python inference.py --weights cfrnet_lolv2_synthetic.pth --input path/to/low_images --output results
```

`--input` accepts an image or a folder. Add `--device cuda` to use a GPU.
Restored images are saved as PNG files.

## Evaluation

For paired images with matching filenames:

```bash
pip install lpips==0.1.4
python inference.py --weights cfrnet_lolv2_real.pth --input path/to/Test/Low --gt path/to/Test/Normal --output results_real --padding reflect --lpips
```

Results are saved to `metrics.csv`, including RGB/Y PSNR and SSIM, and RGB
LPIPS-VGG. Metrics use the rounded 8-bit prediction. Omit `--lpips` if LPIPS
is not needed.

## Training

```bash
python train.py --help
```

The training entry point accepts paired low/normal image folders, an output
directory, and a training-step budget (`--steps`). Choose an initialization
checkpoint with `--init`, or use `--from-scratch` for random initialization.

## Acknowledgment

The encoder-decoder backbone is adapted from
[DarkIR](https://github.com/Fundacion-Cidaut/DarkIR).
The original MIT license is retained in `LICENSE`.
