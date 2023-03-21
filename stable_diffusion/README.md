<h1 align="center">Text 2 image: Stable Diffusion (SD)</h1>

*This is a WIP branch to add a stable diffusion benchmark to MLCommons*

Based on ColossalAI repo here: https://github.com/hpcaitech/ColossalAI


# Docker image

## Build the image
```bash
./scripts/docker/build.sh
```

## Launch the container
The launch scripts mounts the following directories (<host_path>:<mount_path>):
* benchmark code: `${PWD}:/pwd`
* laion aesthetic in webdataset format: `/datasets/laion2B-en-aesthetic/webdataset:/datasets/laion2B-en-aesthetic`
* coco-2017 : `/datasets/coco-2017:/datasets/coco-2017`
* huggingface cache: `${PWD}/nogit/cache/huggingface:/root/.cache/huggingface`
* results folder : `${PWD}/nogit/results:/results`

```bash
./scripts/docker/launch.sh
```

# Prepare datasets

## download Laion aesthetic
```bash
./scripts/datasets/download_laion2B-en-aesthetic.sh
```

## coco2017 (validation)

```bash
./scripts/datasets/download_coco-2014-validation.sh
```

then pre-process the dataset with:
```bash
./scripts/datasets/process_coco_validation.py
```
The script will:
1. ressize the validation images to 256x256
2. creates a prompts.json file with a random 30k labels

# Training
## Single node (with docker):
```bash
./scripts/train.sh
```

## Multi-node (with SLURM)
TODO

# Run validation
## Generate images (inference)
### Single node (with docker):
```bash
./scripts/batch_txt2img.sh <prompts_json>
```

### Multi-node (with SLURM)
TODO

## Calculate FID score
### Single node (with docker):
```bash
./scripts/fid_score.sh <resized_coco_folder> <generated_images>
```

### Multi-node (with SLURM)
TODO

## Calculate CLIP score
### Single node (with docker):
```bash
./scripts/clip_score.sh <prompts_json> <generated_images>
```

### Multi-node (with SLURM)
TODO
