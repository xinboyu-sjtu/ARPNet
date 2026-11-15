"""NIfTI preprocessing helpers for ARPNet.

These utilities operate on binary masks that have already been segmented,
virtually extracted and registered (for example in 3D Slicer).  They cover
resampling, centre cropping and the largest-connected-component cleanup used
by the inference script.
"""

from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def read_image(path: str) -> Tuple[np.ndarray, sitk.Image]:
    """Read a NIfTI image and return ``(array, sitk_image)``."""
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return array, image


def write_image(
    array: np.ndarray,
    reference: sitk.Image,
    path: str,
) -> None:
    """Write ``array`` as a NIfTI image using the metadata of ``reference``."""
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing(reference.GetSpacing())
    image.SetOrigin(reference.GetOrigin())
    image.SetDirection(reference.GetDirection())
    sitk.WriteImage(image, path)


def resample_image(
    image: sitk.Image,
    spacing: Sequence[float] = (0.2, 0.2, 0.2),
    interpolator: int = sitk.sitkNearestNeighbor,
) -> sitk.Image:
    """Resample an image to an isotropic spacing."""
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def crop_or_pad_at(
    array: np.ndarray,
    start: Sequence[int],
    target_size: Sequence[int],
) -> np.ndarray:
    """Crop or zero-pad ``array`` using a fixed start index.

    ``start`` may be negative, in which case the corresponding border is
    padded.  The same start can be applied to the preoperative and
    postoperative volumes to preserve their spatial alignment.
    """
    target_size = tuple(int(v) for v in target_size)
    start = [int(v) for v in start]
    output = np.zeros(target_size, dtype=array.dtype)

    src_start = [max(0, -start[i]) for i in range(3)]
    dst_start = [max(0, start[i]) for i in range(3)]
    size = [
        min(array.shape[i] - src_start[i], target_size[i] - dst_start[i])
        for i in range(3)
    ]

    if any(v <= 0 for v in size):
        return output

    output[
        dst_start[0]:dst_start[0] + size[0],
        dst_start[1]:dst_start[1] + size[1],
        dst_start[2]:dst_start[2] + size[2],
    ] = array[
        src_start[0]:src_start[0] + size[0],
        src_start[1]:src_start[1] + size[1],
        src_start[2]:src_start[2] + size[2],
    ]
    return output


def crop_pair(
    pre: np.ndarray,
    post: np.ndarray,
    target_size: int = 192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop or pad a registered pre/post pair around the preoperative ROI.

    The crop centre is the centroid of the preoperative mask.  The same
    bounding box is applied to both volumes, so the preoperative and
    postoperative masks remain aligned.
    """
    target = (target_size, target_size, target_size)
    coords = np.argwhere(pre >= 0.5)
    if len(coords) == 0:
        center = np.array(pre.shape, dtype=int) // 2
    else:
        center = np.round(coords.mean(axis=0)).astype(int)

    start = [int(center[i] - target[i] // 2) for i in range(3)]
    return (
        crop_or_pad_at(pre, start, target),
        crop_or_pad_at(post, start, target),
    )


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 26-connected component of a binary 3D mask."""
    binary = (mask >= 0.5).astype(np.uint8)
    labeled, num_components = ndimage.label(binary, structure=np.ones((3, 3, 3)))
    if num_components == 0:
        return binary

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(mask.dtype)


def binarize(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a binary float32 mask."""
    return (array >= threshold).astype(np.float32)
