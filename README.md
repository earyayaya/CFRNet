# CFRNet

Compact PyTorch code for CFRNet low-light image enhancement. Includes the
model, training loss, training entry point, inference/evaluation and two
trained checkpoints. No datasets or comparison-method implementations are included.

## Setup

Python 3.10–3.12. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py --smoke-test
```

For Windows, activate with `.venv\Scripts\activate`. The requirements match
the source environment's package versions; choose the appropriate PyTorch
CUDA build for your system if using a GPU. CPU inference is the default.
The smoke test uses random weights and verifies execution, not image quality.

## Enhance images

```bash
python inference.py --weights weights/cfrnet_lolv2_synthetic_65000.pth --input path/to/low_images --output results
```

`--input` accepts one image or a folder. Add `--device cuda` for a CUDA GPU.
Images are read as RGB in [0,1]; predictions are clamped, rounded to 8 bits
and saved losslessly as `<original_filename>.png`. Existing output files
are not overwritten. There is no test-time augmentation or brightness matching.

| Checkpoint | Source checkpoint step | Width |
|---|---:|---:|
| `cfrnet_lolv2_real_15000.pth` | LOL-v2 Real, 15,000 | 32 |
| `cfrnet_lolv2_synthetic_65000.pth` | LOL-v2 Synthetic, 65,000 | 32 |

These files contain trained parameters, not random initialization. Only the
parameter names were converted; all tensor values are unchanged. The Synthetic
checkpoint is the one used for the accompanying manuscript's feature visualization;
it is not asserted to reproduce every benchmark table in the manuscript.

## Evaluate paired images

Obtain the relevant dataset from its original provider and keep matching
filenames in the low-light and normal-light folders. Dataset images are not
redistributed here.

```bash
python inference.py --weights weights/cfrnet_lolv2_real_15000.pth --input path/to/Test/Low --gt path/to/Test/Normal --output results_real --padding reflect
```

This writes per-image and mean RGB/Y PSNR and SSIM to `metrics.csv`. The
protocol follows the available source evaluator: 8-bit rounded predictions,
studio-range Y conversion, `data_range=255`, default scikit-image 7×7 SSIM,
no border crop. `--padding reflect` reproduces that evaluator's preprocessing;
the default `--padding zero` uses the architecture's internal zero padding.
Use a consistent setting when comparing results.

Optional LPIPS-VGG: `pip install lpips==0.1.4`, then add `--lpips` to the above
command. It uses the source evaluator's raw, unclipped float prediction,
mapped as `2*x-1`. The first run may download the pretrained VGG weights.

## Train

The entry point uses matching image filenames, synchronized random crops and
horizontal/vertical flips. Supply a fresh output directory and the total
cosine-schedule length explicitly:

```bash
python train.py --low path/to/Train/Low --normal path/to/Train/Normal --init path/to/backbone_initialization.pth --steps 50000 --output runs/new_training
```

The initial checkpoint is supplied separately. A full CFRNet checkpoint may
also be used for fine-tuning. `--init` loads model parameters, not optimizer
state. Use `--from-scratch` instead of `--init` only when random initialization
is intended; it does not reproduce the pretrained experiment.

Defaults: width 32, batch 16, patch 256, AdamW (0.9, 0.999), learning rate
1e-4 → 1e-7, gradient clipping 0.5, and loss weights
1.0/0.05/0.5/0.01 for pixel/edge/color/perceptual terms. VGG-19's first 30
feature layers use `2*x-1` normalization as in the source script. Training
may download its pretrained VGG-19 weights on first use.

**Training provenance:** this is a portable adaptation of the available
training script, not a recovered immutable historical run. That script
hard-codes weight decay **1e-4**. The current manuscript and release default
now use **1e-4** consistently; the legacy YAML still contains **2e-4**, which
that script does not read. `--weight-decay` exposes
the choice. The 50,000-step command is an example, not a verified historical
schedule for either included checkpoint. Exact original initialization,
schedule and checkpoint-selection provenance require confirmation before
claiming training-from-initialization reproduction of the manuscript tables.

## Model correspondence and verification

`cfrnet.py` defines CFRNet, CFRB (Channel-wise Frequency Response Bottleneck),
RISE (Real–Imaginary Squeeze Encoding), and CCFR (Channel-wise Complex Frequency
Recalibrator). `backbone.py` contains EBlock, DBlock and their utilities.
RISE averages signed spectral coefficients; CCFR jointly predicts real and
imaginary channel gains. Second-order statistics are an analytical
interpretation, not an extra implemented statistics-estimation module.

Both checkpoints load strictly. On the release checks (CPU, PyTorch 2.14.1),
outputs match the original implementation exactly for two random image sizes
and one real image; input gradients also match exactly. Original source
runtime: PyTorch 2.5.1+cu124 / torchvision 0.20.1+cu124. These execution checks
are not a new full benchmark evaluation.

On that source runtime, a one-step CPU training run with the full VGG-19 loss
saved a checkpoint successfully; paired-image evaluation including LPIPS-VGG
also completed successfully.

Checkpoint SHA-256:

```text
7ec8aa300729dbaf47629f58f7bb30f3775cb55759d9c2d871e5e25e7ce463e1  cfrnet_lolv2_real_15000.pth
3ed529ebfac6b6eba4ac214feb6c7957774a687916bbfc838f29723898570ce2  cfrnet_lolv2_synthetic_65000.pth
```

## Attribution

The encoder–decoder backbone is adapted from
[DarkIR](https://github.com/Fundacion-Cidaut/DarkIR). Its original MIT license
and copyright notice are retained in `LICENSE`. CFRB and the CFRNet-specific
training/inference changes are the additions in this implementation. Renaming
modules does not change the origin of the reused backbone. Please cite the
backbone work when using it.
