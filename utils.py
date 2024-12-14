"""Most utilities copied from HistoL framework
"""

import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from math import sqrt
import wandb
from deeplab_v3 import DeepLabV3
from typing import Dict, List
from numpy.typing import NDArray
import matplotlib.patches as mpatches
from segformer_models.segformer.model import segformer_b0, segformer_b1, segformer_b2, segformer_b3, segformer_b4, segformer_b5


NUM_CLASSES = 2
CLASS_LIST = ['background', 'foreground']
CLASS_COLOUR_MAP = {
    'background': (0, 0, 0),
    'foreground': (255, 0, 0),
}
COLOUR_LIST = np.array(list(CLASS_COLOUR_MAP.values())).astype('uint8')
CLASS_IDX_MAP = {0: 0, 1: 1, 2: 1, 3: 1}
CLASS_WEIGHTS = torch.tensor([0.1, 0.9])

# Heatmap colours.
#: Heatmap (0%)
HEATMAP_HOT_0 = (0, 0, 0)
#: Heatmap (25%)
HEATMAP_HOT_25 = (160, 0, 0)
#: Heatmap (50%)
HEATMAP_HOT_50 = (254, 65, 0)
#: Heatmap (75%)
HEATMAP_HOT_75 = (254, 225, 1)
#: Heatmap (100%)
HEATMAP_HOT_100 = (255, 255, 255)


class MeanValueMeter:
    """A meter for calculating the mean and standard deviation of values."""
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sum_of_squares = 0
        self.mean = np.nan
        self.stddev = np.nan

    def add(self, value, n=1):
        """Add a value to the meter.

        Args:
            value: the value to add.
            n (int): the number of units represented by the value (default is 1).
        """
        if n <= 0:
            raise ValueError(f'Error. n must be positive. Is: {n}')

        self.sum += value
        self.sum_of_squares += value ** 2
        self.n += n
        self.mean = self.sum / self.n

        if self.n == 1:
            self.stddev = np.inf
        else:
            variance = (self.sum_of_squares - self.n * self.mean ** 2) / (self.n - 1)
            self.stddev = sqrt(max(variance, 0))

    def get_mean(self):
        """Gets the mean of values added to the meter

        The mean value is returned as a python float

        Returns:
            (float): Mean value
        """
        if isinstance(self.mean, torch.Tensor):
            return self.mean.item()
        return self.mean

    def value(self):
        """Get statistics of values added to the meter.

        Returns:
            tuple: the mean and standard deviation of values added to the meter.
        """
        return self.mean, self.stddev

    def reset(self):
        self.n = 0
        self.sum = 0
        self.sum_of_squares = 0
        self.mean = np.nan
        self.stddev = np.nan


def log_wandb(data, split, commit=True):
    """Logs a set of data to wandb belonging to a specific split

    Before calling this, wandb should be initialized

    Args:
        data (dict): A dictionary of data to log to wandb
        split (str): The split the data belongs to (train/val/test). This will be used to group
            metrics on a single chart
        commit (bool): Whether to commit the data (update the internal step). This should usually
            be True unless you want to make separate logs to the same step. In which case, set
            commit=False to all but the last log
    """
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'Error. Split should be: \'train\', \'val\' or \'test\'. Is: {split}')
    data_to_log = {f'{k}/{split}': v for k, v in data.items()}
    wandb.log(data_to_log, commit=commit)


def overlay_output(image, output, class_list, class_colour_map):
    """Given an image and output data, overlays the output data onto the image

    Along with the image and output, this method also expects a class_list and class_colour_map

    The class_list is a list of class names, where their positions in the list correspond to
    the index of that class as predicted by the model
    The class_colour_map is a dictionary that maps class names (as defined in class_list) to
    colours (represented as RGB 3-tuples)

    An example of these is as follows:
        class_list = ['Tumour', 'Stroma', 'Necrosis']
        class_colour_map = {'Tumour': (255, 0, 0), 'Stroma': (0, 255, 0), 'Necrosis': (0, 0, 0)}

    This also means that:
        Tumour corresponds to Class 0
        Stroma corresponds to Class 1
        Necrosis corresponds to Class 2

    Args:
        image (np.ndarray): The image to draw on
        output (dict): Output data corresponding to that image
        class_list (list): A list of class names, with indexes corresponding to class indices
        class_colour_map (dict): A mapping from class name to RGB 3-tuple

    Returns:
        image (np.ndarray): The image overlaid with output data
    """
    # Extract the integer-encoded mask from the output
    mask = output

    # Ensure mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure datatype is np.uint8
    mask = mask.astype(np.uint8)

    # Set up colours as a list of colours
    colour_list = [class_colour_map[name] for name in class_list]

    # Get RGB version of the mask, then overlay onto image
    rgb_mask = segmentation_integer_encoded_to_rgb(mask, colour_list)
    image_gray = swap_colour_space(
        swap_colour_space(image, 'rgb', 'gray'), 'gray', 'rgb')
    overlaid_image = overlay_images(image_gray, rgb_mask, 0.25)

    return overlaid_image


def write_output(output, directory, identifier, scale_info=None, write_grayscale=False, write_rgb=False):
    """Given output from the model, writes that output to disk

    This writes all data to a grayscale image and a RGB image for visualization

    Args:
        output (dict): Output data to write to disk
        directory (str): Directory that outputs should be written to
        identifier (str): A unique identifier to be used to create unique filenames
        scale_info (dict/None): Used to specify that the outputs should first be scaled before
            being written to disk. If provided, should be a dictionary with keys 'from_dims'
            and 'to_dims', each containing a tuple of the dimensions to scale from and to
    """
    # Extract the integer-encoded mask from the output
    mask = output

    # Ensure mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure datatype is np.uint8
    mask = mask.astype(np.uint8)

    # Scale the mask if required
    if scale_info is not None:
        # Do this by resizing the mask to the size of 'to_dims'
        # Ensure we use nearest neighbour interpolation to preserve integer encoded mask
        width, height = scale_info['to_dims']
        mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST_EXACT)

    # Write as a grayscale image to disk
    if write_grayscale:
        os.makedirs(os.path.join(directory, 'grayscale_masks'), exist_ok=True)
        cv2.imwrite(os.path.join(directory, 'grayscale_masks', f'mask_gray-{identifier}.png'), mask*255)

    # Write as a RGB image to disk
    if write_rgb:
        os.makedirs(os.path.join(directory, 'rgb_masks'), exist_ok=True)
        rgb_mask = segmentation_integer_encoded_to_rgb(mask, COLOUR_LIST)
        write_image(os.path.join(directory, 'rgb_masks', f'mask_rgb-{identifier}.png'), rgb_mask, overwrite=True)


def segmentation_integer_encoded_to_one_hot_encoded(mask, num_classes):
    """Given an integer encoded segmentation mask, converts it to one-hot encoded.

    An integer encoded mask has dimensionality [H, W] and dtype uint8. Per-pixel, the values should
    be an integer representing the class index.

    This will generate a one-hot encoded mask with dimensionality [H, W, C] and dtype bool. C is
    the number of classes (i.e. each class is assigned its own channel). Within a channel, if a
    pixel has the value 1, it is assumed to belong to that class. Within the whole mask, a pixel
    should have the value 1 on exactly ONE channel.

    Args:
        mask: An integer encoded mask. Dimensionality of [H, W].
        num_classes: The number of classes we expect to find in the mask

    Returns:
        A one-hot encoded mask. Dimensionality of [H, W, C].
    """
    # Ensure correct format of mask
    if mask.ndim != 2:
        raise RuntimeError(f'Mask must have 2 dimensions. Has dimensionality: {mask.shape}')
    if mask.dtype != np.uint8:
        raise RuntimeError(f'Mask must have datatype \'np.uint8\'. Has datatype: {mask.dtype}')

    # Create the mask
    one_hot_mask = np.stack([mask == cls_idx for cls_idx in range(num_classes)], axis=-1)

    # Ensure we only have the value of 1 per-pixel across all channels
    #   To do this, we sum across channels, and ensure each pixel has a value of exactly 1
    #   If speed is required, we could assume this
    if not (one_hot_mask.sum(axis=-1) == 1).all():
        raise RuntimeError(f'Error generating one-hot mask. Not all pixels assigned a value')

    return one_hot_mask


def segmentation_integer_encoded_to_rgb(mask, class_colour_list):
    """Converts an integer encoded segmentation mask to an RGB image based on a colour map

    The class_colour_list should a list of the form:
    [(class_0_rgb_tuple), (class_1_rgb_tuple), ..., (class_n_rgb_tuple)]

    This works by converting integer -> one_hot, then one_hot -> rgb

    Args:
        mask (np.ndarray): An integer encoded mask. Dimensionality of [H, W]. dtype of np.uint8
        class_colour_list (list): A mapping from class index to desired RGB tuple

    Returns:
        (np.ndarray): An RGB image with dimensionality [H, W, 3] and dtype of np.uint8,
            representing the segmentation mask
    """
    # Convert integer mask to one_hot mask
    one_hot_mask = segmentation_integer_encoded_to_one_hot_encoded(mask, len(class_colour_list))

    # Then return the one_hot mask converted to RGB
    return segmentation_one_hot_encoded_to_rgb(one_hot_mask, class_colour_list)


def segmentation_one_hot_encoded_to_rgb(mask, class_colour_list):
    """Converts a one-hot encoded segmentation mask to an RGB image based on a colour map

    The class_colour_list should a list of the form:
    [(class_0_rgb_tuple), (class_1_rgb_tuple), ..., (class_n_rgb_tuple)]

    Args:
        mask: A one-hot encoded segmentation mask with dimensionality [H, W, C],
            where C == the number of classes.
        class_colour_list: A mapping from class index to desired RGB tuple

    Returns:
        An RGB image with dimensionality [H, W, 3] representing the segmentation mask
    """
    # Ensure correct format of mask
    if mask.ndim != 3:
        raise RuntimeError(f'Mask must have 3 dimensions. Has dimensionality: {mask.shape}')
    if mask.dtype != np.bool_:
        raise RuntimeError(f'Mask must have datatype \'np.bool_\'. Has datatype: {mask.dtype}')

    # Convert colour map to an array of dimensions [C, 3], where C = num classes and 3 = RGB tuple
    class_colour_list = np.asarray(class_colour_list, dtype=np.uint8)

    # Can compute the RGB mask by matrix multiplying the one-hot encoded mask with the colour matrix
    # This works because the one-hot encoding has values 1 if pixel is that class otherwise 0
    # By multiplying with the colour map, the pixel is coloured based on the corresponding class
    # The RGB mask is then a matrix of dimensions: [H, W, C] @ [C, 3] == [H, W, 3]
    return mask @ class_colour_list


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


def overlay_images(original, data_to_overlay, alpha, beta=None, gamma=0):
    """Overlays an image ontop of another one

    Inspiration: https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    """
    if beta is None:
        beta = 1 - alpha
    return cv2.addWeighted(data_to_overlay, alpha, original, beta, gamma)


def overlay_mask(image, output, class_list, class_colour_map, alpha=0.25):
    # Extract the integer-encoded mask from the output
    mask = output

    # Ensure mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure datatype is np.uint8
    mask = mask.astype(np.uint8)

    # Set up colours as a list of colours
    colour_list = [class_colour_map[name] for name in class_list]

    # Overlay the mask on the image
    overlay = np.copy(image)
    for class_index in range(1, len(class_list)):  # Not overlaying the class 0 (background)
        overlay[mask == class_index] = colour_list[class_index]

    overlaid_image = cv2.addWeighted(image, alpha, overlay, 1-alpha, gamma=0)

    return overlaid_image


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


def write_image(filepath: str, image, *, overwrite: bool = False):
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


def prepare_deeplabv3_model(num_classes=NUM_CLASSES,
                            backbone=None,
                            pretrained=True,
                            freeze_backbone=False,
                            context_window=1):
    model = DeepLabV3(num_classes=num_classes,
                      backbone=backbone,
                      pretrained=pretrained,
                      freeze_backbone=freeze_backbone)
    model = model.model
    if context_window != 0:
        if backbone in ['resnet50', 'resnet101']:
            model.backbone.conv1 = nn.Conv2d(3 * (context_window*2 + 1), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if backbone == 'mobilenet_v3_large':
            model.backbone['0'][0] = nn.Conv2d(3 * (context_window*2 + 1), 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    return model


def prepare_segformer_model(num_classes=NUM_CLASSES,
                            pretrained_model='segformer_b0',
                            pretrained=True,
                            freeze_encoder=False,
                            context_window=1):
    # model = SegFormerSemSeg(num_classes=num_classes,
    #                         pretrained_model=pretrained_model,
    #                         pretrained=pretrained,
    #                         freeze_encoder=freeze_encoder)
    # model = model.model
    # if context_window != 0:
    #     if pretrained_model == 'segformer_b0':
    #         model.backbone.stages[0].patch_embed.proj = nn.Conv2d(3 * (context_window*2 + 1), 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
    #     if pretrained_model in ('segformer_b1', 'segformer_b2', 'segformer_b3', 'segformer_b4', 'segformer_b5'):
    #         model.backbone.stages[0].patch_embed.proj = nn.Conv2d(3 * (context_window*2 + 1), 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
    
    if pretrained_model == 'segformer_b0':
        model = segformer_b0(pretrained=pretrained, num_classes=num_classes, context_window=context_window)
    elif pretrained_model == 'segformer_b1':
        model = segformer_b1(pretrained=pretrained, num_classes=num_classes, context_window=context_window)
    elif pretrained_model == 'segformer_b2':
        model = segformer_b2(pretrained=pretrained, num_classes=num_classes, context_window=context_window)
    elif pretrained_model == 'segformer_b3':
        model = segformer_b3(pretrained=pretrained, num_classes=num_classes, context_window=context_window)
    elif pretrained_model == 'segformer_b4':
        model = segformer_b4(pretrained=pretrained, num_classes=num_classes, context_window=context_window)
    elif pretrained_model == 'segformer_b5':
        model = segformer_b5(pretrained=pretrained, num_classes=num_classes, context_window=context_window)

    return model


def map_integer_encoded_mask(
    mask: NDArray[np.uint8],
    class_idx_map: Dict[int, int],
) -> NDArray[np.uint8]:
    """Maps an integer encoded mask to another mask through the class_idx map.

    class_idx_map should be a dictionary which maps original classes (keys) to new classes (values).

    Args:
        mask: The mask to be mapped. Should have dimensionality HxW.
        class_idx_map: A mapping from class indices in the original mask to class indices in
            the new mask.

    Returns:
        The mask mapped to the new set of classes. Has shape: HxW.

    Raises:
        ValueError: If there are elements in the mask which are not keys in `class_idx_map`.
    """
    if not set(np.unique(mask)).issubset(class_idx_map.keys()):
        raise ValueError(f'Segmentation mask contains elements not included in the class map. '
                         f'This may be caused by an incorrectly specified region whitelist '
                         f'or class map. Values in mask: {np.unique(mask)}. Class map: '
                         f'{class_idx_map}.')

    mappings: Dict[int, List[NDArray[np.int_]]] = {
        new_class_idx: ([], []) for new_class_idx in set(class_idx_map.values())
    }

    for class_idx, new_class_idx in class_idx_map.items():
        row_ind, col_ind = np.asarray(mask == class_idx).nonzero()
        row_ind_list, col_ind_list = mappings[new_class_idx]
        row_ind_list.append(row_ind)
        col_ind_list.append(col_ind)

    for new_class_idx, (row_ind_list, col_ind_list) in mappings.items():
        flat_row_ind = np.concatenate(row_ind_list)
        flat_col_ind = np.concatenate(col_ind_list)
        mask[flat_row_ind, flat_col_ind] = new_class_idx

    return mask


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



def create_custom_legend(names, colours):
    """Creates a custom legend handle object given some names and colours

    Args:
        names (list): A list of names to be inserted to the legend handle
        colours (list of 3-tuple): A list of RGB [0-255] colours to be associated with each name

    Returns:
        (mpatches.Patch): A handle usable with plt.legend(handle=...) calls
    """
    # Convert colours to RGB [0-1]
    colours = [(r / 255, g / 255, b / 255) for (r, g, b) in colours]

    # Return the custom handle
    return [mpatches.Patch(color=colour, label=name) for name, colour in zip(names, colours)]
