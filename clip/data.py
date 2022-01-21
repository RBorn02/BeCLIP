import logging
import os
import Image

import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from clip import tokenize

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]
        return images, texts


def get_csv_dataloader(args, preprocess_fn):
    input_filename = args.data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    if args.distributed:
       sampler = DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader

def get_imagenet_dataloader(args, config):
    # get data for evaluation
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(config['image_resolution']),
                                    transforms.Normalize(mean, std)])
    val_dir = os.path.join(args.imagenet_dir, 'val')

    test_data = ImageFolder(val_dir, transform)

    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return test_loader

    