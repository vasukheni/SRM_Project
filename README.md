# Deep Learning Based Super Resolution Mapping (SRM) — Prototype

A working PyTorch prototype for turning medium-resolution satellite imagery
(e.g. Sentinel-2 10m, LISS-3 23.5m, AWiFS 56m) into higher-resolution output
using a CNN-based super-resolution network, with an optional GAN fine-tuning
stage — suitable as a Smart India Hackathon demo.

## What's in here

| File | Purpose |
|---|---|
| `model.py` | SRResNet-style generator (residual blocks + sub-pixel/PixelShuffle upsampling) and an optional PatchGAN discriminator |
| `dataset.py` | Builds LR/HR training pairs from high-res tiles by simulating sensor degradation (bicubic downsampling) |
| `metrics.py` | PSNR and SSIM evaluation |
| `train.py` | Training loop; plain content-loss (L1) training, or adversarial fine-tuning with `--gan` |
| `infer.py` | Runs a trained model on a new image and produces an SR output + bicubic baseline + comparison figure |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
pip install -r requirements.txt

# 1. Put a folder of high-resolution image tiles (aim for 50-200+ for a real
#    training run) at, e.g., data/hr_tiles/  — PNG/JPG/TIFF, any size >= 96px.
#    Good sources: Bhuvan / VEDAS / Bhoonidhi high-res crops, Sentinel-2 10m
#    tiles, PlanetScope samples, or regular high-res photos while prototyping.

# 2. Train (content-loss only — the recommended starting point):
python train.py --data_dir data/hr_tiles --scale 4 --epochs 30 --out_dir checkpoints

# 3. (Optional) fine-tune adversarially for sharper textures, starting from
#    the model you just trained:
python train.py --data_dir data/hr_tiles --scale 4 --epochs 10 --gan \
    --resume checkpoints/best.pth --out_dir checkpoints_gan

# 4. Run inference on a new medium-res tile:
python infer.py --checkpoint checkpoints/best.pth --input path/to/lr_tile.png \
    --scale 4 --out_dir results

# With a known ground-truth high-res tile, to get PSNR/SSIM numbers:
python infer.py --checkpoint checkpoints/best.pth --input path/to/lr_tile.png \
    --ground_truth path/to/hr_tile.png --scale 4 --out_dir results
```

## How the training data is built

True paired (real medium-res, real high-res, same place, same time) satellite
imagery is hard to get. So `dataset.py` uses the standard SR-literature trick:

1. Take a high-resolution tile.
2. Randomly crop a patch.
3. Bicubic-downsample it to simulate what the medium-resolution sensor would
   have captured.
4. Train the network to invert that downsampling — i.e. map the simulated
   low-res patch back to the real high-res patch.

**Once you have real co-registered medium-res / high-res pairs of the same
area** (e.g. Sentinel-2 vs Cartosat over the same AOI, resampled to align),
swap the downsampling step for loading the real low-res tile directly — the
model, training loop and metrics all stay the same.

## Working with real multispectral satellite bands (GeoTIFF)

`dataset.py` includes `load_geotiff(path, bands=(1,2,3))`, which uses
`rasterio` to read specific bands (e.g. R/G/B, or add a 4th NIR band) from a
GeoTIFF. To train on 4-band imagery, set `--in_channels 4` when calling
`train.py`/`infer.py` and swap the image-loading calls in `dataset.py` /
`infer.py` to use `load_geotiff` instead of `PIL.Image.open`.

## Tuning for your hardware / timeline

- **CPU-only / quick hackathon demo**: `--base_channels 16 --num_res_blocks 4
  --patch_size 24 --batch_size 4` trains in minutes on a small dataset.
- **GPU / real training run**: defaults (`base_channels=64`,
  `num_res_blocks=8`, `patch_size=48`) are a good starting point; bump
  `num_res_blocks` to 16 and add more data for better quality.
- `--scale` supports 2, 3, 4, or 8 (matches common LR:HR resolution ratios,
  e.g. AWiFS 56m -> ~7m is roughly 8x, Sentinel-2 10m -> 2.5m is 4x).

## Evaluating quality

- **PSNR / SSIM** (computed automatically in `train.py` validation and
  optionally in `infer.py`) tell you pixel/structural accuracy against a
  known high-res tile.
- For a stronger hackathon demo, also show a **downstream task** improving:
  e.g. run a simple edge/building/road detector or land-cover classifier on
  the bicubic baseline vs. the SR output and show the SR version performs
  better — this is usually more convincing to judges than PSNR alone.
- Be upfront about **hallucination risk**: GAN-based sharpening can invent
  plausible-looking but false detail. It's worth showing both the pure
  content-loss model (more faithful, safer for analysis) and the GAN
  fine-tuned model (sharper, more visually impressive) and discussing the
  trade-off.

## Suggested extensions for a full SIH submission

- Multi-image super-resolution (MISR): fuse several time-steps of the same
  area (e.g. repeat Sentinel-2 passes) instead of single-image SR, which
  tends to recover more real detail.
- A simple web front-end (e.g. a Leaflet/Bhuvan-style map) where a user picks
  an AOI and sees the SR output rendered.
- Domain adaptation / fine-tuning per sensor and per region for better
  generalization across India's diverse terrain.
