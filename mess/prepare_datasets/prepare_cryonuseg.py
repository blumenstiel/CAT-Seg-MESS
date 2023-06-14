
# run python mess/datasets/prepare/prepare_cryonuseg.py

import tqdm
import os
from pathlib import Path
import gdown
import kaggle

import numpy as np
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading kaggle
    try:
        kaggle.api.authenticate()
    except:
        raise Exception('Please install kaggle and save credentials in ~/.kaggle/kaggle.json, '
                        'see https://github.com/Kaggle/kaggle-api')
    # CLI: kaggle datasets download -d ipateam/segmentation-of-nuclei-in-cryosectioned-he-images
    kaggle.api.dataset_download_cli('ipateam/segmentation-of-nuclei-in-cryosectioned-he-images', path=ds_path, unzip=True)


def main():
    dataset_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets'))
    ds_path = dataset_dir / 'CryoNuSeg'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    # create directories
    img_dir = ds_path / 'images_detectron2'
    anno_dir = ds_path / 'annotations_detectron2'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    for img_path in tqdm.tqdm((ds_path / 'tissue images').glob('*.tif')):
        id = img_path.stem
        # Move image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img.save(img_dir / f'{id}.png', 'PNG')

        # Open mask
        mask = Image.open(ds_path / 'Annotator 1 (biologist second round of manual marks up)'
                        / 'Annotator 1 (biologist second round of manual marks up)' / 'mask binary' / f'{id}.png')
        # Edit annotations
        # Binary encoding: (0, 255) -> (0, 1)
        mask = np.uint8(np.array(mask) / 255)
        # Save mask
        Image.fromarray(mask).save(anno_dir / f'{id}.png')

    print(f'Saved images and masks of {ds_path.name} dataset')


if __name__ == '__main__':
    main()