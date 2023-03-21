from functools import partial

from torch.utils.data import default_collate
from torchvision import transforms
import lightning.pytorch as pl

from einops import rearrange
import webdataset as wds

from ldm.util import instantiate_from_config


def identity(x):
    return x

def keys_filter(keys):
    def filter_fn(sample):
        return {k: v for k, v in sample.items() if k in keys}
    return filter_fn

def metadata_filter(key, predicate, value):
    def filter_fn(sample):
        if predicate == "eq":
            return sample[key] == value
        elif predicate == "neq":
            return sample[key] != value
        elif predicate == "gt":
            return sample[key] > value
        elif predicate == "lt":
            return sample[key] < value
        elif predicate == "gte":
            return sample[key] >= value
        elif predicate == "lte":
            return sample[key] <= value
        else:
            raise ValueError(f"Unknown predicate: {predicate}")
    return filter_fn

def instantiate_transforms_from_config(config):
    if config.target in ['torchvision.transforms.RandomResizedCrop', 'torchvision.transforms.Resize']:
        # the isinstance is necessary because instantiate_transforms_from_config might be called multiple times
        # and isinstance(config['params']['interpolation'] already caseted from str to InterpolationMode
        if "interpolation" in config['params'] and isinstance(config['params']['interpolation'], str):
            config.params.interpolation = interpolation_from_string(config['params']['interpolation'])
    return instantiate_from_config(config)

def interpolation_from_string(interpolation):
    interpolation_map = {
        'nearest': transforms.InterpolationMode.NEAREST,
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'box': transforms.InterpolationMode.BOX,
        'hamming': transforms.InterpolationMode.HAMMING,
        'lanczos': transforms.InterpolationMode.LANCZOS,
    }
    return interpolation_map[interpolation]

def rearrange_transform(pattern):
    return transforms.Lambda(lambda x: rearrange(tensor=x, pattern=pattern))


def debug(input):
    print("###########################################################################")
    print(input)
    print(input.shape)
    print("###########################################################################")
    return input

class DataModuleFromWebDataloaders(pl.LightningDataModule):

    def __init__(self, train=None, validation=None, test=None, predict=None):
        super().__init__()
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = partial(self._gen_dataloader, mode="train")
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._gen_dataloader, mode="validation")
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._gen_dataloader, mode="test")
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = partial(self._gen_dataloader, mode="predict")

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

    def _gen_dataloader(self, mode):
        return self.datasets[mode]

def build_webdataloader(
        urls,
        batch_size,
        shuffle=-1,
        partial=False,
        metadata_filters=None,
        keep_only_keys=None,
        image_transforms=None,
        txt_transforms=None,
        num_workers=1,
        cache_size=-1,
        cache_dir=None,
        persistent_workers=True):
    # TODO(ahmadki): WebDataset supports a "PipeLine" format which is more convenient than
    # the "fluid" format used here. But that one results in an error (TypeError: 'FilterFunction' object is not iterable)
    # which I haven't been able to debug yet.

    image_transforms = transforms.Compose([instantiate_transforms_from_config(t) for t in image_transforms]) if image_transforms is not None else identity
    txt_transforms = transforms.Compose([instantiate_from_config(t) for t in txt_transforms]) if txt_transforms is not None else identity

    dataset = wds.WebDataset(urls=urls, resampled=True, cache_size=cache_size, cache_dir=cache_dir)

    for filter in metadata_filters or []:
        dataset = dataset.select(instantiate_from_config(filter))

    dataset = dataset.shuffle(size=shuffle).decode("pil")

    if keep_only_keys:
        dataset = dataset.map(keys_filter(keep_only_keys))

    if image_transforms or txt_transforms:
        dataset = dataset.map_dict(jpg=image_transforms, txt=txt_transforms)

    # dataset = dataset.map_dict(jpg=debug, txt=identity)

    dataset = dataset.batched(batch_size, partial=partial, collation_fn=default_collate)
    return wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
