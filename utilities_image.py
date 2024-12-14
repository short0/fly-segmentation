"""Utilities related to interacting with image data.
"""
import os
from typing import Tuple, Optional, Union, Sequence, List

import cv2
import numpy as np
from numpy.typing import NDArray

from utilities_geometry import get_region_dimensions


def opencv_interpolation(interpolation_mode: str):
    if interpolation_mode == 'area':
        return cv2.INTER_AREA
    if interpolation_mode == 'nearest':
        # See https://ppwwyyxx.com/blog/2021/Where-are-Pixels/.
        # For consistency, we will prefer type (2) sampling grid methods which means
        # NEAREST_EXACT instead of NEAREST.
        return cv2.INTER_NEAREST_EXACT
    elif interpolation_mode == 'nearest-inexact':
        return cv2.INTER_NEAREST

    raise ValueError(f'Invalid value for interpolation_mode: '
                     f'expected "area" or "nearest", '
                     f'got {repr(interpolation_mode)}.')


def hsv_to_rgb(h: int, s: int, v: int) -> tuple[int, int, int]:
    """Convert a colour from HSV to RGB.

    Args:
        h: Hue (0 to 255).
        s: Saturation (0 to 255).
        v: Value (0 to 255).

    Returns:
        The converted (red, green, blue) tuple with values between 0 and 255.
    """
    h_region, h_remainder = divmod(h, 43)
    h_remainder *= 6
    p = (v * (255 - s)) >> 8
    q = (v * (255 - ((s * h_remainder) >> 8))) >> 8
    t = (v * (255 - ((s * (255 - h_remainder)) >> 8))) >> 8
    if h_region == 0: return (v, t, p)
    if h_region == 1: return (q, v, p)
    if h_region == 2: return (p, v, t)
    if h_region == 3: return (p, q, v)
    if h_region == 4: return (t, p, v)
    return (v, p, q)


# ##################################################################################################
#                                   Data Loading/Writing Utilities
# ##################################################################################################
def load_image(filepath, colour_space='rgb'):
    """Loads an image into a numpy array

    Args:
        filepath (str): Filepath to the image to be loaded
        colour_space (str): The colour space to load the data in

    Returns:
        (numpy ndarray): The loaded RGB image, with channel ordering HWC
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Cannot load file: {filepath} - Does not exist.')
    return swap_colour_space(cv2.imread(filepath), 'bgr', colour_space)


def write_image(filepath: str, image: NDArray[np.uint8], *, overwrite: bool = False):
    """Writes an image to file.

    Args:
        filepath: The filepath to write the data to.
        image: The image data to write. Should be RGB, HWC axis ordering.
        overwrite: Whether to overwrite the existing file if it exists.

    Raises:
        FileExistsError: If `filepath` exists and `overwrite` is False.
    """
    # Check if the file already exists
    if not overwrite and os.path.isfile(filepath):
        raise FileExistsError(f'File \'{filepath}\' already exists. Set the overwrite flag '
                              f'to overwrite the file')

    # Write the data
    cv2.imwrite(filepath, swap_colour_space(image, 'rgb', 'bgr'))


def get_windowed_load_crop(
    image_wh: Tuple[int, int],
    window: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Given a window, determines where to place a crop of the valid image data into that window

    Args:
        image_wh: The width, height of the image being loaded
        window: The x1, y1, x2, y2 coordinates of the window to be loaded

    Returns:
        The x1, y1, x2, y2 coordinates of where the image should be positioned in an array with
        shape `window`.
    """
    window_x1, window_y1 = window[:2]
    window_w, window_h = get_region_dimensions(window)

    # Calculate the bounds of the "valid pixels" within the window.
    cx1 = max(0, min(-window_x1, window_w))
    cy1 = max(0, min(-window_y1, window_h))
    cx2 = max(0, min(image_wh[0] - window_x1, window_w))
    cy2 = max(0, min(image_wh[1] - window_y1, window_h))

    return (cx1, cy1, cx2, cy2)


def calculate_level_mpps(
    image_mpp_xy: tuple[float, float],
    max_sublevel_resolution_mpp: float = 0.25,
    max_sublevels: int = 7,
    sublevel_creation_threshold: float = 1.9,
) -> list[tuple[float, float]]:
    """Calculate level resolutions for an image pyramid.

    Args:
        image_mpp_xy:
            The full-sized image resolution in MPP as an (x, y) tuple.
        max_sublevel_resolution_mpp:
            The maximum allowed resolution for a sublevel, in MPP.
        max_sublevels:
            The maximum number of sublevels possible. Implicitly defines a minimum resolution
            for a sublevel as ``max_sublevel_resolution_mpp * 2 ** (max_sublevels - 1)``.
        sublevel_creation_threshold:
            Minimum downscaling factor required for a sublevel to be created.

    Returns:
        A list of tuples representing the (x, y) resolutions (in MPP) of each level. The first
        element of the list will always be `image_mpp_xy`.
    """
    level_mpps = [image_mpp_xy]
    for i in range(max_sublevels):
        mpp = max_sublevel_resolution_mpp * 2 ** i
        # We only create levels that are significantly smaller than the previous level.
        if mpp / min(level_mpps[-1]) >= sublevel_creation_threshold:
            level_mpps.append((mpp, mpp))
    return level_mpps


# ##################################################################################################
#                                   Image Inspection Utilities
# ##################################################################################################
def get_image_aspect_ratio(image):
    """Determines the aspect ratio (Height/Width) of an image stored in HWC format"""
    return image.shape[0] / image.shape[1]


# ##################################################################################################
#                               Image Storage Manipulation Utilities
# ##################################################################################################
def swap_colour_space(image, from_space, to_space):
    """Swaps the colour space from one space to another

    This builds up an internal map from space to space, expecting a lambda that describes how the
    space conversion should occur

    NOTE:
        HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].

    Args:
        image (numpy ndarray): The image to swap the colour space of. Stored in the form HWC
        from_space (str): The original colour space of the image
        to_space (str): The desired colour space of the image

    Returns:
        (numpy ndarray): The image converted from the 'from_space' colour space to the 'to_space'
            colour space
    """
    # Build up a mapping from space -> space. Attempt to find the shortest path when converting
    space_map = [
        ['rgb', 'bgr', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)],
        ['bgr', 'rgb', lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)],
        ['rgb', 'gray', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)],
        ['gray', 'rgb', lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)],
        ['gray', 'bgr', lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)],
        ['bgr', 'gray', lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)],
        ['rgb', 'hsv', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)],
        ['bgr', 'hsv', lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV)],
        ['hsv', 'rgb', lambda x: cv2.cvtColor(x, cv2.COLOR_HSV2RGB)],
        ['rgb', 'lab', lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)],
    ]

    # Capitalization invariant
    from_space = from_space.lower()
    to_space = to_space.lower()

    # Early exit
    if from_space == to_space:
        return image

    # Find the shortest path between spaces. First attempt to find manually with list comprehension
    conversion_function = [lst for lst in space_map if lst[0] == from_space and lst[1] == to_space]

    if len(conversion_function) != 1:
        raise NotImplementedError(f'Conversion from {from_space} to {to_space} not supported')

    image = conversion_function[0][2](image)

    return image


def apply_colour_map(image, colour_map='hot'):
    """Applies a colour map to a given image, returning the RGB colour map

    Args:
        image (numpy ndarray): The HxW ndarray. Should be given as np.uint8 or np.float32. If given
            as float32, image is first multiplied by 255 then converted to np.uint8
        colour_map (str): The colour map to apply. See here for types:
            https://docs.opencv.org/4.2.0/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
            Should specify the suffix of the colour map as a string

    Returns:
        (numpy ndarray): The HxWx3 RGB image with colour map applied
    """
    # Validate dimensionality of image
    if image.ndim != 2:
        raise ValueError(f'Error. Image should have 2 dimensions. Has: {image.ndim} dimensions.')

    # Validate datatype of image
    if image.dtype == 'float32':
        image = (image * 255).astype(np.uint8)
    elif image.dtype not in ('uint8', ):
        raise ValueError(f'Error. Image datatype {image.dtype} unsupported.')

    # Get the colour map
    colour_map = colour_map.upper()
    try:
        colour_map = getattr(cv2, f'COLORMAP_{colour_map}')
    except AttributeError:
        raise ValueError(f'Error. Colour map: cv2.COLORMAP_{colour_map} not found!')

    # Apply the colour map
    mapped_image = cv2.applyColorMap(image, colour_map)

    # Convert to RGB
    mapped_image = swap_colour_space(mapped_image, 'bgr', 'rgb')

    # Return mapped image
    return mapped_image


# ##################################################################################################
#                                   Image Data Extraction Utilities
# ##################################################################################################
def crop_image(image: NDArray, box: Union[Tuple[int, int, int, int], List[int]],
               allow_pad: bool = False, pad_fill_value: int = 0) -> NDArray:
    """Crops an image given a bounding box region

    The box should be in the form (x1, y1, x2, y2), as the top-left and bottom-right coordinates

    Assumes image stored in HWC format (or HW if a grayscale image)

    Args:
        image: Image to be cropped
        box: Bounding box coordinates of region to crop
        allow_pad: Whether padding is allowed (applied if the box has negative coordinates, or
            coordinates beyond the image dimensions).
        pad_fill_value: If padding applied, the pad colour.

    Returns:
        The cropped image.
    """
    # If padding not allowed, crop directly from the image
    if not allow_pad:
        return image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), ...]

    # Convert box to numpy array (makes a copy to not modify original box)
    box = np.array(box)

    # Get the size of the crop to take
    crop_size = get_region_dimensions(box)

    # H, W, C
    if image.ndim == 2:
        image_data = np.full((crop_size[1], crop_size[0]), fill_value=pad_fill_value,
                             dtype=image.dtype)
    else:
        image_data = np.full((crop_size[1], crop_size[0], image.shape[2]),
                             fill_value=pad_fill_value, dtype=image.dtype)

    # Get the region where data should be loaded into
    cx1, cy1, cx2, cy2 = get_windowed_load_crop((image.shape[1], image.shape[0]), box)

    # Bound box crop to image
    box[[0, 2]] = np.clip(box[[0, 2]], a_min=0, a_max=image.shape[1])
    box[[1, 3]] = np.clip(box[[1, 3]], a_min=0, a_max=image.shape[0])

    # Load image data
    image_data[cy1:cy2, cx1:cx2, ...] = image[box[1]:box[3], box[0]:box[2], ...]

    return image_data


# ##################################################################################################
#                               Image Size Manipulation Utilities
# ##################################################################################################
def pad_image(image, target_height, target_width, pad_colour=(0, 0, 0)):
    """Given an image, spatially pads it within the given dimensions

    This results in a centred image

    Rounds the target height and width to the nearest integer
    Image must be passed in with axis order: HWC

    If the images' width or height is > target_width/target_height, a ValueError is raised

    Args:
        image (numpy ndarray): The image to be padded, with channel ordering HWC
        target_height (int): The target height of the padded image
        target_width (int): The target width of the padded image
        pad_colour (int/float 3-tuple): The colour to pad the image with. This should be specified
            as RGB. If your input image is stored as uint8 or float32, set the pad_colour
            accordingly

    Returns:
        (numpy ndarray): The padded image, with channel ordering HWC
    """
    target_height = int(round(target_height))
    target_width = int(round(target_width))

    # Get the current height and width
    current_height, current_width = image.shape[:2]

    # Early exit
    if current_height == target_height and current_width == target_width:
        return image

    # Validate the current dimensions are <= the target dimensions
    if current_height > target_height or current_width > target_width:
        raise ValueError(f'Error. Image dimensions are larger than the target padded size. '
                         f'Consider calling resize_image with preserving aspect ratio instead. '
                         f'Image dimensions: {current_height}x{current_width}, '
                         f'Target dimensions: {target_height}x{target_width}.')

    # Setup an empty array and initialize it with the pad colour
    target_image = np.zeros((target_height, target_width, image.shape[-1]), dtype=image.dtype)
    target_image[..., :] = pad_colour

    # Compute the amount of top/left padding (bottom/right padding is implied)
    top_pad = round((target_height - current_height) / 2)
    left_pad = round((target_width - current_width) / 2)

    # Update that region within the image with the original image
    target_image[top_pad:top_pad + image.shape[0], left_pad:left_pad + image.shape[1], :] = image

    return target_image


def get_optimal_size_ratio(current_size, target_size):
    """Returns the optimal size to resize current to fit within target

    Given a current/target height/width, determines the best size to resize current to fit within
    target whilst preserving the aspect ratio

    This size ratio enforces that the resulting image height & width will be <= the targets

    Args:
        current_size ((int, int)): The current height and width of the image
        target_size ((int, int)): The target height and width of the image

    Returns:
        (int, int): The final height/width such that the aspect ratio is preserved and the current
            height/width is within the target height/width
    """
    # Get the current aspect ratio (H/W)
    current_aspect_ratio = current_size[0] / current_size[1]

    # Compute the candidate height/width combinations
    height_fixed_size = (target_size[0], round(target_size[0] / current_aspect_ratio))
    width_fixed_size = (round(target_size[1] * current_aspect_ratio), target_size[1])

    # Ensure valid ratios (valid if not greater than desired size)
    height_fixed_size_valid = height_fixed_size[1] <= target_size[1]
    width_fixed_size_valid = width_fixed_size[0] <= target_size[0]

    # Choose the width/height combination. If both
    if not height_fixed_size_valid and not width_fixed_size_valid:
        # If both are invalid, we have found an unsolvable ratio
        raise ValueError(f'Error. Cannot find a solution to preserve aspect ratios')
    elif height_fixed_size_valid and width_fixed_size_valid:
        if height_fixed_size == width_fixed_size:
            size_ratio = height_fixed_size
        else:
            raise ValueError(f'Error. Found multiple valid ratios. '
                             f'{height_fixed_size} and {width_fixed_size}')
    elif height_fixed_size_valid:
        size_ratio = height_fixed_size
    else:
        size_ratio = width_fixed_size
    return size_ratio


def get_optimal_scale_factor(current_size, target_size):
    """Returns the optimal scale factor to resize current to fit within target

    Given a current/target height/width, determines the x/y scale factors to resize current to fit
    within target whilst preserving the aspect ratio

    The returned scale factors are guaranteed to be very similar (though may not be identical due
    to rounding)

    Args:
        current_size ((int, int)): The current height and width of the image
        target_size ((int, int)): The target height and width of the image

    Returns:
        (float, float): The sy, sx scale factors. current_size should be scaled by these.
            <1 = gets smaller, >1 = gets larger
    """
    size_ratio = get_optimal_size_ratio(current_size, target_size)
    return size_ratio[0] / current_size[0], size_ratio[1] / current_size[1]


def resize_image(image, target_height, target_width, preserve_aspect_ratio=False,
                 pad_colour=(0, 0, 0), interpolation=None):
    """Given an image, spatially resizes it to the given dimensions

    Rounds the given target_height and width to the nearest integer in case they are non-ints
    Expects channel ordering: HWC

    Users have the option to preserve the aspect ratio. When selected:

    1. We fix the target_width/target_height individually and find the corresponding height/width
       using the original aspect ratio.

    2. We then choose the pair with calculated {height, width} <= {target_height, target_width}.

    3. To meet the target height/width, we use padding for the remaining area.

    If the aspect ratio is preserved, the morphology of the image will be preserved, i.e. no
    skew/warped data. Otherwise, depending on the target height/width, the resulting image will
    likely be skewed.

    Args:
        image (numpy ndarray): The image to be resized, with channel ordering HWC
        target_height (int): The target height of the resized image
        target_width (int): The target width of the resized image
        preserve_aspect_ratio (bool): If True, will aim to preserve the original aspect ratio as
            closely as possible, and pad any extra areas with black
        pad_colour (int/float 3-tuple): The colour to pad the image with. This should be specified
            as RGB. If your input image is stored as uint8 or float32, set the pad_colour
            accordingly
        interpolation: The OpenCV interpolation method.

    Returns:
        (numpy ndarray): The resized image, with channel ordering HWC
    """
    if not preserve_aspect_ratio:
        return cv2.resize(image, (round(target_width), round(target_height)),
                          interpolation=interpolation)

    # Determine the best size ratio to get an image with height/width < targets
    #   This gives us the actual target height/width size
    size_ratio = get_optimal_size_ratio(image.shape[:2], (target_height, target_width))

    # Resize the image (Note: Our ratio stored as (H, W), OpenCV requires (W, H))
    resized_image = cv2.resize(image, size_ratio[::-1], interpolation=interpolation)

    # Pad the surrounding area and return the image
    return pad_image(resized_image, target_height, target_width, pad_colour)


def rescale_image(image, rescale_factor):
    """Rescales an image by a given factor. Factor must be positive

    If 1 < rescale_factor, the image will be upscaled
    If 0 < rescale_factor < 1, the image will be downscaled

    Args:
        image (numpy ndarray): The image to be rescaled, with channel ordering HWC
        rescale_factor (float/(float, float)): The factor to rescale the image by. If given as a
            tuple is expected as (sf_y, sf_x). If single value given, scale is applied equally

    Returns:
        (numpy ndarray): The rescaled image, with channel ordering HWC
    """
    if not isinstance(rescale_factor, (tuple, list)):
        rescale_factor = (rescale_factor, rescale_factor)
    if rescale_factor[0] <= 0 or rescale_factor[1] <= 0:
        raise ValueError(f'Error. Rescale factor must be > 0. You specified: {rescale_factor}')
    return resize_image(image, image.shape[0] * rescale_factor[0],
                        image.shape[1] * rescale_factor[1])


# ##################################################################################################
#                                       Heatmap Creation
# ##################################################################################################
def generate_gaussian_point_heatmap(
    points: NDArray,
    sigma: float,
    heatmap_size: Sequence[int] = None,
    out_heatmap: Optional[NDArray[np.float32]] = None,
) -> NDArray[np.float32]:
    """Creates a heatmap with Gaussians centred at each point.

    If any parts of Gaussians overlap, the maximal value at that location is taken.
    The Gaussians are not normalized, so the largest value of each Gaussian will be 1.

    This code is adapted from the Stacked Hourglass 2D Gaussian implementation:
    https://github.com/bearpaw/pytorch-pose/blob/edc7aece651649927764435ce0c1f03cffa2edaf/pose/utils/imutils.py#L52

    And with other inspiration taken from:
    https://stackoverflow.com/a/69026901

    Args:
        points: The set of (N, 2) points within the heatmap to create 2D Gaussians.
        sigma: The standard deviation of the Gaussians.
        heatmap_size: The desired (height, width) of the Gaussian to create. Not required if
            specifying out_heatmap.
        out_heatmap: If provided, the Gaussians will be drawn onto this heatmap, otherwise a new
            heatmap will be created.

    Returns:
        The generated heatmap with Gaussians created at each point in points.
    """
    if heatmap_size is None and out_heatmap is None:
        raise ValueError('Must specify heatmap_size or out_heatmap.')

    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f'Points should have shape: (N, 2). Has shape: {points.shape}')

    if out_heatmap is None:
        out_heatmap = np.zeros(heatmap_size, dtype=np.float32)

    if out_heatmap.ndim != 2:
        raise ValueError(f'Heatmap should have 2 dimensions (H, W). Has dimensionality: '
                         f'{out_heatmap.shape}')

    # Generate the Gaussian to place at each point
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    # To normalize we would multiply by 1/(2*pi*sigma^2)
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Iterate through all points and draw the Gaussian
    for pt in points:
        # Check that any part of the Gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] >= out_heatmap.shape[1] or ul[1] >= out_heatmap.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, don't do anything with this point
            continue

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], out_heatmap.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], out_heatmap.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], out_heatmap.shape[1])
        img_y = max(0, ul[1]), min(br[1], out_heatmap.shape[0])

        out_heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            out_heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )

    return out_heatmap


# ##################################################################################################
#                                   Other Image Manipulation
# ##################################################################################################
def overlay_images(original, data_to_overlay, alpha, beta=None, gamma=0):
    """Overlays an image ontop of another one

    Inspiration: https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    """
    if beta is None:
        beta = 1 - alpha
    return cv2.addWeighted(data_to_overlay, alpha, original, beta, gamma)
