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


def cli_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', default='data', type=str, help='Path to images')
    parser.add_argument('--output-dir', default='inference_input', type=str, help='Path to overlaid images')

    # Output configuration
    parser.add_argument('--mask', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to mask the output images by setting pixels which are not '
                             'within the well to black.')
    parser.add_argument('--output-radius', type=int, default=None,
                        help='The side length of output images will be twice this value. '
                             'By default, this will be set to the outer radius.')
    parser.add_argument('--output-size', type=int, default=None,
                        help='Override the --output-radius.'
                             'The output images will be resized to have the size length of this value.')

    # Registration algorithm configuration
    parser.add_argument('--inner-radius', type=int, default=1280,
                        help='The inner radius of the rim of the well, in pixels.')
    parser.add_argument('--outer-radius', type=int, default=1440,
                        help='The outer radius of the rim of the well, in pixels.')
    parser.add_argument('--flywell-shift-dir', type=str,
                        choices=['u', 'ur', 'r', 'dr', 'd', 'dl', 'l', 'ul'],
                        help='The direction the flywell is shifted in relative to the image. Use '
                             'this if the flywell is not well-centred in the frame, and/or '
                             'partially out of frame.', default='')
    parser.add_argument('--max-shift', type=int, default=64,
                        help='The maximum amount of shift expected between two subsequent '
                             'images, in pixels. '
                             'Setting this value too low will interfere with registration. '
                             'Setting this value too high is safe, but will slow things down.')
    parser.add_argument('--n-angles', type=int, default=27,
                        help='The number of points to sample around each circle when measuring '
                             'symmetry')
    parser.add_argument('--downscale-factor', type=int, default=8,
                        help='The resize factor to use when detecting the rim. '
                             'A larger value will result in faster registration.')

    return parser.parse_args(argv)


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


def main(argv: list[str]):
    opts = cli_options(argv)

    base_input_dir = opts.input_dir
    base_output_dir = opts.output_dir

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

        r0 = opts.inner_radius
        r1 = opts.outer_radius

        downscale_factor = opts.downscale_factor

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
            x_min_div = downscale_factor * 2 if 'l' in opts.flywell_shift_dir else downscale_factor
            y_min_div = downscale_factor * 2 if 'u' in opts.flywell_shift_dir else downscale_factor
            x_max_div = downscale_factor * 2 if 'r' in opts.flywell_shift_dir else downscale_factor
            y_max_div = downscale_factor * 2 if 'd' in opts.flywell_shift_dir else downscale_factor

            if first_image:
                cy = scaled_image.shape[0] // 2
                cx = scaled_image.shape[1] // 2
                margin = np.inf
            else:
                margin = opts.max_shift // downscale_factor

            (best_x, best_y), best_cost = find_centre_of_symmetry(
                image=scaled_image,
                n_angles=opts.n_angles,
                r0=r0 // downscale_factor,
                r1=r1 // downscale_factor,
                x_min=max(cx - margin, r1 // x_min_div),
                x_max=min(cx + margin, scaled_image.shape[1] - r1 // x_max_div),
                y_min=max(cy - margin, r1 // y_min_div),
                y_max=min(cy + margin, scaled_image.shape[0] - r1 // y_max_div),
                r_step=2,
                shift_dir=opts.flywell_shift_dir,
            )

            if best_x - cx >= 0.9 * margin or best_y - cy >= 0.9 * margin:
                warnings.warn('Increase relative margin')

            cy = best_y
            cx = best_x

            best_x = round(best_x * downscale_factor)
            best_y = round(best_y * downscale_factor)

            out_radius = opts.output_radius
            if out_radius is None:
                out_radius = opts.outer_radius

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

            if opts.mask:
                rr, cc = disk((cropped.shape[0] // 2, cropped.shape[1] // 2), out_radius)
                mask = np.ones_like(cropped, dtype=bool)
                mask[rr, cc] = False
                cropped[mask] = 0

            # Resize
            if opts.output_size:
                cropped = cv2.resize(cropped, (opts.output_size, opts.output_size), interpolation=cv2.INTER_AREA)

            # Save output image
            basename = os.path.splitext(filename)[0]
            imsave(os.path.join(output_dir, f'{image_number:03d}-{basename}-registered.jpg'), cropped)

            # Save registration coordinates
            writer.writerow([image_number, filename, best_x, best_y])
            csv_file.flush()

        csv_file.close()


if __name__ == '__main__':
    main(sys.argv[1:])
