HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIFFUSERS_OFFLINE=1

python main.py --train \
    --logdir /results  \
    -b ./configs/train_from_scratch.yaml
