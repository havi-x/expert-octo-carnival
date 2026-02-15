from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from pathlib import Path
import torch

def init_dataloaders(cfg: object, split: str):
    dataset_name = cfg.dataset_name
    supported_datasets_map = {
        'mnist': (datasets.MNIST, (28, 28)),
        'cifar10': (datasets.CIFAR10, (32, 32)),
        'fashion-mnist': (datasets.FashionMNIST, (28, 28)),
    }
    assert dataset_name in supported_datasets_map, f"{dataset_name=} dataset not supported, it should be one of {supported_datasets_map.keys()}"
    assert split in ['train', 'val']
    cache_dir = Path("./dataset")
    cache_dir.mkdir(exist_ok=True, parents=True)

    data_fn, img_size = supported_datasets_map[dataset_name]

    transform = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    train = split == 'train'
    dataset = data_fn(root=cache_dir, train=train, download=True, transform=transform)

    dataloader_kwargs = dict(cfg.dataloader)
    dataloader_kwargs['drop_last'] = split=="train" and dataloader_kwargs.get('drop_last', False)
    dataloader_kwargs['shuffle'] = split=="train"
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    return dataloader