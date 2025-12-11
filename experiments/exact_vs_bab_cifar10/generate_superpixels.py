"""Generate superpixel masks with increasing numbers of superpixels."""

import argparse
import random
from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed
from skimage.util import img_as_float
from torchvision.datasets import CIFAR10
from tqdm import tqdm


def segment(img, n_segments):
    segments_slic = slic(img, n_segments=n_segments)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=n_segments)
    return segments_slic, segments_watershed


def num_segments(segments):
    return len(np.unique(segments))


def segments_to_mask(segments):
    num = num_segments(segments)
    mask = (segments[..., None] == np.arange(1, num + 1)).astype(np.bool_)
    mask = np.moveaxis(mask, -1, 0).reshape(num, 1, 32, 32)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", type=str, default=None)
    parser.add_argument("--min-segments", type=int, default=1)
    parser.add_argument("--max-segments", type=int, default=100)
    parser.add_argument("--step-segments", type=int, default=1)
    args = parser.parse_args()

    dataset = CIFAR10(root=".datasets", train=False, download=True)

    to_calculate = list(
        range(args.min_segments, args.max_segments + 1, args.step_segments)
    )
    test = list(range(args.min_segments, args.max_segments * 2))

    progress = tqdm(total=len(to_calculate))
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    num_imgs = 100
    for i, (img, _) in enumerate(dataset):
        # after 100 images, go with merging segments
        if i >= num_imgs:
            break

        img = img_as_float(img)
        for n_segments in test:
            segments1, segments2 = segment(img, n_segments)
            num1, num2 = num_segments(segments1), num_segments(segments2)

            if num1 not in results and num1 in to_calculate:
                mask1 = segments_to_mask(segments1)
                results[num1] = (img, mask1)
                progress.update(1)
            if num2 not in results and num2 in to_calculate:
                mask2 = segments_to_mask(segments2)
                results[num2] = (img, mask2)
                progress.update(1)

    print(f"Missing after {num_imgs} images:")
    missing = [n for n in to_calculate if n not in results]
    print(missing)

    for n in (n for n in reversed(to_calculate) if n not in results):
        next_larger = next(m for m in results if m > n)
        img, mask = results[next_larger]
        for _ in range(next_larger - n):
            # Merge two random masks
            num = mask.shape[0]
            i1, i2 = random.sample(range(num), k=2)
            mask[i1] = mask[i1] | mask[i2]
            mask = np.delete(mask, i2, axis=0)

        assert mask.shape[0] == n
        results[n] = (img, mask)
        progress.update(1)

    print("Missing now:")
    missing = [n for n in to_calculate if n not in results]
    print(missing)

    for n, (_, mask) in results.items():
        assert n == mask.shape[0]

    out_name = args.out_name
    if out_name is None:
        out_name = f"cifar10_superpixels_{num_imgs}"
    output_dir = Path(__file__).parent / "resources" / out_name
    output_dir.mkdir(parents=True, exist_ok=False)

    images = {f"{i}": img for i, (img, _) in results.items()}
    masks = {f"{i}": mask for i, (_, mask) in results.items()}
    np.savez_compressed(output_dir / "images.npz", **images)
    np.savez_compressed(output_dir / "masks.npz", **masks)
