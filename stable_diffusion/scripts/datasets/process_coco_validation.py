import os
import json
import argparse
from multiprocessing import Pool

import pandas as pd
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--captions-file", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--num-samples", type=int, default=30000)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--resize-processes", type=int, default=4)
parser.add_argument("--allow-duplicate-images", type=bool, default=False)

args = parser.parse_args()


def resize_image(input_image, output_image, width, height, resample=Image.BICUBIC):
    image = Image.open(input_image)
    image = image.resize((width, height), resample=resample)
    image.save(output_image)


# Load coco annotations
with open(args.captions_file, "r") as f:
    captions = json.load(f)
    annotations = captions["annotations"]

# Convert to dataframe
df = pd.DataFrame(annotations)

# Shuffle the dataframe
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Keep a single captions per image
if not args.allow_duplicate_images:
    df = df.drop_duplicates(subset=["image_id"], keep="first")

# Take a subset
df = df[:args.num_samples]

# Sort by id
df = df.sort_values(by=["id"])

# Save the subset to a tsv file
df.to_csv(os.path.join(args.output_dir, "captions.tsv"), sep="\t", index=False)


# Create output image directory if it doesn't exist
output_images_dir = os.path.join(args.output_dir, "images")
if not os.path.exists(output_images_dir):
    os.makedirs(output_images_dir)

# resize images with a worker pool
with Pool(args.resize_processes) as p:
    for i, row in df.iterrows():
        image_fname = f"{row['image_id']:012}.jpg"
        input_img = os.path.join(args.input_dir, image_fname)
        output_img = os.path.join(output_images_dir, image_fname)

        p.apply_async(resize_image, args=(input_img, output_img, args.width, args.height, Image.BICUBIC))
