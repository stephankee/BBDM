# BBDM Codebase Instructions

## Project Overview
This is an implementation of **Brownian Bridge Diffusion Models (BBDM)** for image-to-image translation. The architecture supports both pixel-space (BBDM) and latent-space (LBBDM) diffusion with VQGAN encoders.

### Key Application: Image Denoising
**BBDM is IDEAL for denoising tasks**, including specialized data like Range-Doppler Maps (RDMs):
- **Brownian Bridge Process**: Interpolates between noisy input (y) and clean output (x0)
- **Conditional Generation**: Takes noisy image as condition, generates denoised result
- **Single Image Processing**: Can denoise individual images without paired training data (using colorization/inpainting pattern)
- **For denoising RDMs or other noisy images**: Use `custom_aligned` dataset with noisy images in A/, clean images in B/

## Core Architecture

### Two-Model System
1. **BBDM (Pixel Space)**: Direct image-to-image translation using UNet in pixel space
2. **LBBDM (Latent Space)**: Uses frozen VQGAN encoder/decoder for latent diffusion (f4/f8/f16 variants)
   - VQGAN models are pretrained from LDM project and must be downloaded separately
   - VQGAN stays frozen (eval mode) - only the UNet denoising network trains
   - Set `model_type: "LBBDM"` in config for latent models

### Key Components Flow
```
main.py → get_runner() → BBDMRunner → BrownianBridgeModel/LatentBrownianBridgeModel
                       ↓
                  BaseRunner (training loop, EMA, checkpointing)
```

### Register Pattern
- All datasets and runners use decorator-based registration via `Register.py`
- Register with: `@Registers.runners.register_with_name('BBDMRunner')`
- Retrieve with: `Registers.runners[name]` or `Registers.datasets[name]`
- New custom components MUST be registered to be discoverable

## Configuration System

### YAML Config Structure
- `runner`: Must match a registered runner name (e.g., "BBDMRunner")
- `model.model_type`: "BBDM" (pixel) or "LBBDM" (latent)
- `model.model_name`: Used in output path generation
- `data.dataset_type`: Must match registered dataset name (e.g., 'custom_aligned', 'custom_colorization_RGB', 'custom_inpainting')
- Templates in `configs/Template-*.yaml` show working configurations

### Critical Config Fields
- `model.VQGAN.params.ckpt_path`: **Required for LBBDM** - path to pretrained VQGAN checkpoint
- `model.normalize_latent`: If True, computes/loads latent mean/std from training data
- `model.latent_before_quant_conv`: Controls where quantization happens in VQGAN pipeline
- `model.BB.params.condition_key`: "nocond", "first_stage", or "SpatialRescaler" determines conditioning

## Dataset Conventions

### Directory Structure Requirements
**Paired translation** (custom_aligned):
```
dataset_path/
  train/A/  # condition images
  train/B/  # ground truth images
  val/A/
  val/B/
  test/A/
  test/B/
```

**Colorization/Inpainting** (custom_colorization_RGB, custom_inpainting):
```
dataset_path/
  train/  # ground truth only (conditions generated on-the-fly)
  val/
  test/
```

### Dataset Returns
All datasets return: `((image, image_name), (condition, condition_name))`
- Used consistently in `loss_fn()` and `sample()` methods

## Training Workflow

### Starting Training
```bash
# From scratch
python3 main.py --config configs/your_config.yaml --train --sample_at_start --save_top --gpu_ids 0

# Resume training
python3 main.py --config configs/your_config.yaml --train --gpu_ids 0 \
  --resume_model path/to/latest_model_X.pth \
  --resume_optim path/to/latest_optim_sche_X.pth
```

### Distributed Training
- Multi-GPU via PyTorch DDP: `--gpu_ids 0,1,2,3`
- DDP automatically enabled when multiple GPUs specified
- Uses `nccl` backend with localhost on port specified by `--port` (default: 12355)

### Checkpoint Management
- **Latest checkpoints**: `latest_model_{epoch}.pth`, `latest_optim_sche_{epoch}.pth` (cleaned up each epoch)
- **Last checkpoint**: `last_model.pth`, `last_optim_sche.pth` (always present)
- **Top checkpoint**: `top_model_epoch_X.pth` (saved when `--save_top` and validation loss improves)
- Model checkpoints contain: model state, EMA shadow, epoch, step, latent mean/std (if used)
- Optimizer checkpoints contain: optimizer state, scheduler state

### EMA (Exponential Moving Average)
- Configured in `model.EMA` section of config
- Shadow weights tracked separately, applied during validation/testing with `apply_ema()` / `restore_ema()`
- Updates every `update_ema_interval` steps, starts at `start_ema_step`

## Testing & Evaluation

### Generate Test Samples
```bash
python3 main.py --config configs/your_config.yaml --sample_to_eval --gpu_ids 0 \
  --resume_model path/to/model_ckpt
```
- Outputs to `results/{dataset_name}_{model_name}/sample_to_eval/`
- Creates: `condition/`, `ground_truth/`, `{sample_step}/` directories
- For `sample_num > 1`, saves multiple samples per image in subdirectories

### Evaluation Metrics
Use `preprocess_and_evaluation.py`:
```bash
# LPIPS (perceptual similarity)
python3 preprocess_and_evaluation.py -f LPIPS -s source_dir -t target_dir -n 1

# Diversity (for multiple samples per image)
python3 preprocess_and_evaluation.py -f diversity -s source_dir -n 5

# FID (use external fidelity package)
fidelity --gpu 0 --fid --input1 path1 --input2 path2
```

## Development Patterns

### Adding a New Dataset
1. Create class in `datasets/custom.py` extending `Dataset`
2. Register with `@Registers.datasets.register_with_name('your_name')`
3. Return format: `((image, name), (condition, name))`
4. Set `data.dataset_type: 'your_name'` in config

### Using BBDM for Denoising Tasks
**Denoising with Paired Data** (Recommended for RDMs):
1. Prepare dataset: noisy images in `train/A/`, clean images in `train/B/`
2. Use `dataset_type: 'custom_aligned'` in config
3. Model learns: noisy (condition) → clean (target)
4. At inference: pass noisy image, model outputs denoised result

**Denoising without Paired Data**:
1. Use clean images only in `train/` directory
2. Use `dataset_type: 'custom_colorization_RGB'` or create custom dataset that adds noise
3. Modify dataset to add synthetic noise as condition
4. Model learns to remove the synthetic noise pattern

**Key Insight**: The Brownian Bridge formulation naturally handles the noisy→clean transformation by interpolating between the noisy condition (y) and clean target (x0) through the diffusion process.

### Brownian Bridge Mechanics
- Forward process: `q_sample()` interpolates from x0 to y with noise
- Reverse process: `p_sample()` denoises step-by-step from y to x0
- Three objective types: 'grad' (default), 'noise', 'ysubx'
- Timestep schedule: 'linear' or 'sin' via `mt_type`

### Loss Function
Located in `BBDMRunner.loss_fn()`:
- Encodes images to latent (for LBBDM) or uses directly (BBDM)
- Calls `net.forward(x, x_cond)` which internally calls `p_losses()`
- Returns L1 or L2 loss between predicted and true objective

## Common Gotchas

1. **VQGAN checkpoint must exist** before training LBBDM - check path in config
2. **Latent normalization** requires computing statistics over training set at startup (adds ~10min)
3. **Command-line args override config**: `--resume_model` has higher priority than `model.model_load_path`
4. **DDP training** requires barrier synchronization - don't add prints/saves in non-main processes
5. **Dataset path structure** must exactly match expected format (A/B subdirs for aligned, flat for colorization)
6. **Config YAML anchors**: Use `!!python/tuple` for tuple parameters in UNetParams

## File Organization
- `model/BrownianBridge/`: Core diffusion models and UNet
- `model/VQGAN/`: VQGAN encoder/decoder (frozen for LBBDM)
- `runners/DiffusionBasedModelRunners/`: Training/testing logic
- `datasets/`: Dataset implementations
- `evaluation/`: LPIPS, diversity metrics
- `configs/`: YAML configuration templates
- `shell/`: Example training scripts
