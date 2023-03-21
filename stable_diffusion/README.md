<h1 align="center">Text 2 image: Stable Diffusion (SD)</h1>

*This is a WIP branch to add a stable diffusion benchmark to MLCommons*

Based on ColossalAI repo here: https://github.com/hpcaitech/ColossalAI


# Build image

```bash
./scripts/docker/build.sh
```

# Download dataset
## Laion aesthetic
```bash
./scripts/datasets/download_laion2B-en-aesthetic.sh
```

## coco2017

```bash
./scripts/datasets/download_coco-2014-validation.sh
```
# Run training
# Run validation
## Preprocess dataset
## Generate images
### Calculate FID score
### Calculate CLIP score


