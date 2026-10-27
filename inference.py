"""Strict checkpoint loading and single-image/folder inference."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from cfrnet import CFRNet


def read_weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for field in ("params_ema", "params_g", "params", "state_dict", "model"):
        if field in checkpoint and isinstance(checkpoint[field], dict):
            checkpoint = checkpoint[field]
            break
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state dictionary checkpoint.")
    converted = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint entry: {key}")
        key = key.removeprefix("module.")
        key = key.replace("spectral_gate.freq_gate.1.", "cfrb.ccfr.predictor.0.")
        key = key.replace("spectral_gate.freq_gate.3.", "cfrb.ccfr.predictor.2.")
        key = key.replace("spectral_gate.fusion.", "cfrb.fusion.")
        if key in converted:
            raise ValueError(f"Duplicate checkpoint entry after renaming: {key}")
        converted[key] = value
    return converted


def load_model(path, device="cpu"):
    weights = read_weights(path)
    if "intro.weight" not in weights:
        raise ValueError("Missing intro.weight; this is not a supported CFRNet checkpoint.")
    model = CFRNet(width=weights["intro.weight"].shape[0])
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def read_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--input", type=Path, help="Image file or folder (non-recursive)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--padding", choices=("zero", "reflect"), default="zero",
                        help="zero: model default; reflect: legacy evaluation preprocessing")
    parser.add_argument("--gt", type=Path, help="Optional same-filename reference folder; reports RGB and Y PSNR/SSIM")
    parser.add_argument("--lpips", action="store_true", help="Also report LPIPS-VGG; requires --gt and pip install lpips")
    parser.add_argument("--smoke-test", action="store_true", help="Check random-weight forward/backward; not restoration quality")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    if args.smoke_test:
        torch.manual_seed(0)
        model = CFRNet().to(args.device)
        x = torch.rand(1, 3, 31, 37, device=args.device, requires_grad=True)
        y = model(x)
        y.square().mean().backward()
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        print("PASS: forward/backward and original image size. Random weights; no quality claim.")
        return
    if args.weights is None or args.input is None:
        parser.error("--weights and --input are required unless using --smoke-test.")
    if args.lpips and args.gt is None:
        parser.error("--lpips requires --gt.")
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    )
    if not files:
        parser.error("No supported input images found.")
    if args.gt:
        for source in files:
            if not (args.gt / source.name).is_file():
                parser.error(f"Missing reference: {source.name}")
        if (args.output / "metrics.csv").exists():
            parser.error("metrics.csv already exists; choose a new output directory.")
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    perceptual = None
    if args.lpips:
        import lpips
        perceptual = lpips.LPIPS(net="vgg", version="0.1").to(args.device).eval()
    outputs = [args.output / (p.name + ".png") for p in files]
    for source, target in zip(files, outputs):
        if source.resolve() == target.resolve() or target.exists():
            parser.error(f"Refusing to overwrite: {target}. Choose a new output folder.")
    model = load_model(args.weights, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for source, target in zip(files, outputs):
            inputs = read_image(source).to(args.device)
            height, width = inputs.shape[-2:]
            if args.padding == "reflect":
                pad_h, pad_w = (-height) % 8, (-width) % 8
                if pad_h >= height or pad_w >= width:
                    parser.error("Image is too small for reflect padding; use --padding zero.")
                inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
            output = model(inputs)[:, :, :height, :width]
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Non-finite model output for {source}")
            array = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            quantized = np.rint(array * 255).astype(np.uint8)
            if args.gt:
                reference_tensor = read_image(args.gt / source.name)
                reference = np.rint(reference_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if reference.shape != quantized.shape or min(height, width) < 7:
                    raise ValueError(f"Reference size mismatch or image smaller than SSIM window: {source.name}")
                # Match the source evaluator: 8-bit RGB, MATLAB-style studio-range Y,
                # skimage's default 7x7 SSIM window, no border crop, data_range=255.
                coefficients = np.array([65.481, 128.553, 24.966]) / 255
                output_y = quantized @ coefficients + 16
                reference_y = reference @ coefficients + 16
                row = {"image": source.name,
                       "PSNR_RGB": peak_signal_noise_ratio(reference, quantized, data_range=255),
                       "SSIM_RGB": structural_similarity(reference, quantized, data_range=255, channel_axis=2,
                                                         win_size=7, gaussian_weights=False, use_sample_covariance=True),
                       "PSNR_Y": peak_signal_noise_ratio(reference_y, output_y, data_range=255),
                       "SSIM_Y": structural_similarity(reference_y, output_y, data_range=255,
                                                       win_size=7, gaussian_weights=False, use_sample_covariance=True)}
                if perceptual is not None:
                    # Use the exact RGB8 images scored above and saved to PNG.
                    pred_lpips = torch.from_numpy(quantized.copy()).permute(2, 0, 1)[None]
                    gt_lpips = torch.from_numpy(reference.copy()).permute(2, 0, 1)[None]
                    pred_lpips = pred_lpips.to(args.device, dtype=torch.float32) / 255 * 2 - 1
                    gt_lpips = gt_lpips.to(args.device, dtype=torch.float32) / 255 * 2 - 1
                    row["LPIPS_VGG"] = perceptual(pred_lpips, gt_lpips).item()
                rows.append(row)
            Image.fromarray(quantized).save(target)
            print(f"Saved {target}")
    if rows:
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image"}
        with (args.output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows + [{"image": "MEAN", **mean}])
        print(mean)


if __name__ == "__main__":
    main()
