import argparse
import csv
import math
import os
import re
import sys
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.filters import gaussian
from skimage.io import imread, imsave
from skimage.transform import downscale_local_mean
from tqdm import tqdm
import cv2


def angles_to_vectors(angles: np.ndarray):
    """Converts angles (in radians) to unit vectors (of the form (y, x)).
    """
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1)


def _scan_for_centre(
        interpolator,
        centred_sample_points,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        step: float,
):
    costs = []
    for i in np.arange(y_min, y_max + step, step=step):
        row = []
        costs.append(row)
        for j in np.arange(x_min, x_max + step, step=step):
            sample_points = centred_sample_points + [i, j]
            values = interpolator(sample_points)
            cost = values.std(axis=-1).sum()
            row.append(cost)

    costs = np.asarray(costs)
    bmy, bmx = np.unravel_index(np.argmin(costs), costs.shape)

    best_x = x_min + bmx * step
    best_y = y_min + bmy * step
    best_cost = costs[bmy, bmx]

    return (best_x, best_y), best_cost


def find_centre_of_symmetry(
        image: np.ndarray,
        n_angles: int,
        r0: float,
        r1: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        r_step: int = 1,
        shift_dir: str = '',
):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    interpolator = RegularGridInterpolator((y, x), image, bounds_error=False, fill_value=0)

    dir_vectors = angles_to_vectors(np.linspace(0, 2 * np.pi, n_angles, endpoint=False))

    # If there is a shift in the centre position of the flywell image, remove some angles to be
    # analysed, given the part of the flywell they should fall on may be out of frame.
    if shift_dir != '':
        # Assume a unit circle broken up into 8x octants (directions). Remove angles surrounding
        # the direction that the flywell is shifted in. Right = oct 0, down = oct 2, left = oct 4,
        # etc.
        _SECT_IDXS = {'r': 0, 'dr': 1, 'd': 2, 'dl': 3, 'l': 4, 'ul': 5, 'u': 6, 'ur': 7}
        excl_sect = _SECT_IDXS[shift_dir]

        # Determine the index into dir_vectors of the centre angle to be removed
        centre_idx = round((n_angles / len(_SECT_IDXS)) * excl_sect)

        # Determine the number of points that should be removed (to remove ~ a quarter of angles)
        num_pts_to_remove = max(1, round(2 * math.log(n_angles, 2) - 3))

        # Work out which indices should be removed (num_pts_to_remove centred on centre_idx)
        remove_idxs = np.asarray([centre_idx - num_pts_to_remove // 2 + i
                                  for i in range(num_pts_to_remove)])
        remove_idxs[remove_idxs < 0] = n_angles + remove_idxs[remove_idxs < 0]

        # Remove the indexes from dir_vectors
        dir_vectors = np.delete(dir_vectors, remove_idxs, axis=0)

    centred_sample_points = dir_vectors[None, :, :] * np.arange(r0, r1, r_step)[:, None, None]

    (best_x, best_y), best_cost = _scan_for_centre(
        interpolator=interpolator,
        centred_sample_points=centred_sample_points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        step=1,
    )

    # Sub-pixel fine-tuning
    (best_x, best_y), best_cost = _scan_for_centre(
        interpolator=interpolator,
        centred_sample_points=centred_sample_points,
        x_min=best_x - 4,
        x_max=best_x + 4,
        y_min=best_y - 4,
        y_max=best_y + 4,
        step=1 / 4,
    )

    return (best_x, best_y), best_cost


def register_images(input_dir: str, output_dir: str, mask: bool = True, output_radius: int = None, output_size: int = None, 
                   inner_radius: int = 1230, outer_radius: int = 1380, flywell_shift_dir: str = '', max_shift: int = 64, 
                   n_angles: int = 27, downscale_factor: int = 8):
    """
    Registers images by applying alignment, masking, and resizing operations based on the provided parameters.

    Parameters:
    -----------
    input_dir : str
        Path to the directory containing input images to be registered.
        
    output_dir : str
        Path to the directory where the registered images will be saved.

    mask : bool, optional, default=True
        Whether to apply a mask to the output images, setting pixels outside the "well" to black.

    output_radius : int, optional, default=None
        Specifies the radius for output images. The side length of the output images will be twice this value. 
        If not provided, it defaults to the outer radius.

    output_size : int, optional, default=None
        Overrides `output_radius`. The output images will be resized to have a side length equal to this value.

    inner_radius : int, optional, default=1230
        The inner radius of the "rim" of the well, in pixels.

    outer_radius : int, optional, default=1380
        The outer radius of the "rim" of the well, in pixels.

    flywell_shift_dir : str, optional, default=''
        Specifies the direction of the flywell's shift in the image frame. Useful when the flywell is not 
        centered or partially out of frame. Allowed values:
        - 'u': up
        - 'ur': up-right
        - 'r': right
        - 'dr': down-right
        - 'd': down
        - 'dl': down-left
        - 'l': left
        - 'ul': up-left
        - '': no shift (default)

    max_shift : int, optional, default=64
        The maximum expected shift between two consecutive images, in pixels. 
        Smaller values increase accuracy but may interfere with registration if the shift is larger.
        Larger values are safer but may slow down processing.

    n_angles : int, optional, default=27
        The number of angular sampling points around each circle when measuring symmetry. 
        Higher values may improve accuracy but increase computation time.

    downscale_factor : int, optional, default=8
        The downscale factor applied when detecting the rim. Larger values speed up processing 
        but may reduce accuracy.

    Returns:
    --------
    None
        Processes and saves registered images in the specified output directory.
    """

    base_input_dir = input_dir
    base_output_dir = output_dir

    input_dirs = [e.name for e in os.scandir(base_input_dir) if e.is_dir()]
    input_dirs.sort()

    for input_dir_basename in tqdm(input_dirs):
        input_dir = os.path.join(base_input_dir, input_dir_basename)
        output_dir = os.path.join(base_output_dir, input_dir_basename)

        os.makedirs(output_dir, exist_ok=True)

        filenames = []
        for filename in os.listdir(input_dir):
            m = re.match(r'^.*-\d-\d-[A-Z]\d-(\d+).*\.jpg$', filename)
            if not m:
                continue
            filenames.append((int(m.groups()[0]), filename))
        filenames.sort()

        r0 = inner_radius
        r1 = outer_radius

        downscale_factor = downscale_factor

        cy = None
        cx = None

        csv_path = os.path.join(output_dir, f'registration-{input_dir_basename}.csv')
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['image_number', 'filename', 'centre_x', 'centre_y'])

        for image_number, filename in tqdm(filenames, desc=input_dir_basename):
            image_rgb = imread(os.path.join(input_dir, filename))

            image = rgb2gray(image_rgb)

            scaled_image = downscale_local_mean(image, (downscale_factor, downscale_factor))
            scaled_image = gaussian(scaled_image, 24 // downscale_factor)

            first_image = cx is None or cy is None

            # Set up the r1 divisors to use when identifying the area to analyse in.
            # These are expanded if any shift occurs (default = downscale factor)
            # Double the divisors if we say the flywell is shifted in that direction
            x_min_div = downscale_factor * 2 if 'l' in flywell_shift_dir else downscale_factor
            y_min_div = downscale_factor * 2 if 'u' in flywell_shift_dir else downscale_factor
            x_max_div = downscale_factor * 2 if 'r' in flywell_shift_dir else downscale_factor
            y_max_div = downscale_factor * 2 if 'd' in flywell_shift_dir else downscale_factor

            if first_image:
                cy = scaled_image.shape[0] // 2
                cx = scaled_image.shape[1] // 2
                margin = np.inf
            else:
                margin = max_shift // downscale_factor

            (best_x, best_y), best_cost = find_centre_of_symmetry(
                image=scaled_image,
                n_angles=n_angles,
                r0=r0 // downscale_factor,
                r1=r1 // downscale_factor,
                x_min=max(cx - margin, r1 // x_min_div),
                x_max=min(cx + margin, scaled_image.shape[1] - r1 // x_max_div),
                y_min=max(cy - margin, r1 // y_min_div),
                y_max=min(cy + margin, scaled_image.shape[0] - r1 // y_max_div),
                r_step=2,
                shift_dir=flywell_shift_dir,
            )

            if best_x - cx >= 0.9 * margin or best_y - cy >= 0.9 * margin:
                warnings.warn('Increase relative margin')

            cy = best_y
            cx = best_x

            best_x = round(best_x * downscale_factor)
            best_y = round(best_y * downscale_factor)

            out_radius = output_radius
            if out_radius is None:
                out_radius = outer_radius

            # Assign data in this way to handle partially out-of-frame flywells
            num_channels = None if image_rgb.ndim == 2 else image_rgb.shape[2]
            height, width = 2*out_radius, 2*out_radius
            if num_channels is not None:
                cropped = np.zeros((height, width, num_channels), dtype=image_rgb.dtype)
            else:
                cropped = np.zeros((height, width), dtype=image_rgb.dtype)

            # Determine if the crop is beyond the bounds of the image
            x1_off = 0 if best_x - out_radius > 0 else 0 - (best_x - out_radius)
            y1_off = 0 if best_y - out_radius > 0 else 0 - (best_y - out_radius)
            x2_off = 0 if best_x + out_radius < image_rgb.shape[1] else (best_x + out_radius) - image_rgb.shape[1]
            y2_off = 0 if best_y + out_radius < image_rgb.shape[0] else (best_y + out_radius) - image_rgb.shape[0]

            # Set the area to take from the RGB image
            rgb_x1, rgb_y1 = max(best_x - out_radius, 0), max(best_y - out_radius, 0)
            rgb_x2, rgb_y2 = min(best_x + out_radius, image_rgb.shape[1]), min(best_y + out_radius, image_rgb.shape[0])

            # Place the flywell crop within the cropped image
            cropped[0+y1_off:height-y2_off, 0+x1_off: width-x2_off] = image_rgb[rgb_y1:rgb_y2, rgb_x1:rgb_x2]

            if mask:
                rr, cc = disk((cropped.shape[0] // 2, cropped.shape[1] // 2), out_radius)
                mask = np.ones_like(cropped, dtype=bool)
                mask[rr, cc] = False
                cropped[mask] = 0

            # Resize
            if output_size:
                cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)

            # Save output image
            basename = os.path.splitext(filename)[0]
            imsave(os.path.join(output_dir, f'{image_number:03d}-{basename}-registered.jpg'), cropped)

            # Save registration coordinates
            writer.writerow([image_number, filename, best_x, best_y])
            csv_file.flush()

        csv_file.close()
