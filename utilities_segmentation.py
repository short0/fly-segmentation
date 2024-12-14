"""Utility functions to help with segmentation"""
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from utilities_geometry import polygon_to_region, get_region_smaller_edge
from utilities_image import crop_image


# ##################################################################################################
#                       Conversion between segmentation representations
# ##################################################################################################
# TODO: Write a unit test for converting between mask encodings
def segmentation_integer_encoded_to_one_hot_encoded(
    mask: NDArray[np.uint8],
    num_classes: int,
) -> NDArray[np.bool_]:
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


def segmentation_one_hot_encoded_to_integer_encoded(mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Given a one-hot encoded segmentation mask, converts it to integer encoded.

    See :func:`.segmentation_integer_encoded_to_one_hot_encoded` for definitions of one-hot and
    integer encoded segmentation masks.

    Args:
        mask: A one-hot encoded mask. Dimensionality of [H, W, C].

    Returns:
        An integer encoded mask. Dimensionality of [H, W].
    """
    # Ensure correct format of mask
    if mask.ndim != 3:
        raise RuntimeError(f'Mask must have 3 dimensions. Has dimensionality: {mask.shape}')
    if mask.dtype != np.bool_:
        raise RuntimeError(f'Mask must have datatype \'np.bool_\'. Has datatype: {mask.dtype}')

    # Create the mask
    integer_mask = np.sum(
        np.asarray([mask[..., cls_idx] * cls_idx for cls_idx in range(mask.shape[-1])]),
        axis=0, dtype=np.uint8)

    return integer_mask


def segmentation_one_hot_encoded_to_rgb(
    mask: NDArray[np.bool_],
    class_colour_list: list,
) -> NDArray[np.uint8]:
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


# ##################################################################################################
#                                   General mask utilities
# ##################################################################################################
def determine_mask_type(mask):
    """Given a mask, determines the type of the mask (one_hot encoded or integer encoded)

    This is determined by analyzing the data type and the shape of the mask

    Args:
        mask (ndarray): The mask to evaluate the type of

    Returns:
        (str): 'one_hot' or 'integer' depending on the detected mask type
    """
    if mask.dtype == np.bool_ and mask.ndim == 3:
        return 'one_hot'
    elif mask.dtype == np.uint8 and mask.ndim == 2:
        return 'integer'
    else:
        raise RuntimeError(f'Could not determine the mask type! Mask has datatype: '
                           f'{mask.dtype} and shape: {mask.shape} ({mask.ndim} dimensions)')


def get_binary_mask_bounding_foreground_coords(mask) -> tuple[int, int, int, int]:
    """Returns the coordinates of the box bounding the binary mask foreground.

    Inspired by: https://stackoverflow.com/a/31402351

    Args:
        mask: The mask to find the bounds of the foreground in.

    Returns:
        The x1, y1, x2, y2 coordinates of the box bounding the mask foreground.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return cmin, rmin, cmax, rmax


# ##################################################################################################
#                       Mapping masks between segmentation classes
# ##################################################################################################
def map_one_hot_encoded_mask(
    mask: NDArray[np.bool_],
    num_classes: int,
    class_idx_map: dict,
) -> NDArray[np.bool_]:
    """Maps a one hot encoded mask to another mask through the class_idx map

    class_idx_map should be a dictionary with keys = original classes, values = new classes

    Args:
        mask: The mask to be mapped. Should have dimensionality HxWxN, where
            N = the number of classes (one class per-channel)
        num_classes: The number of classes in the mapped mask (i.e. the returned mask will
            have dimensionality HxWx<num_classes>
        class_idx_map: A mapping from class indices in the original mask to class indices in
            the new mask

    Returns:
        The mask mapped to the new set of classes. Has shape: HxWx<num_classes>
    """
    # Generate a new mask with the right number of channels
    new_mask = np.zeros((*mask.shape[:2], num_classes), dtype=np.bool_)

    # NOTE: Given multiple original classes can map to the same new class, we need to add each mask
    #       instead of assigning (like the old approach)
    for class_idx, new_class_idx in class_idx_map.items():
        new_mask[..., new_class_idx] += mask[..., class_idx]

    # Old approach
    # class_idxs, new_class_idxs = zip(*class_idx_map.items())
    # new_mask[..., new_class_idxs] = mask[..., class_idxs]

    return new_mask


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


# ##################################################################################################
#                               Extracting regions from a mask
# ##################################################################################################
def one_hot_mask_to_annotation(
        mask,
        annotation: str = 'boxes',
        min_sidelength: int = 8,
        crop_location: Optional[Sequence[int]] = None,
        repeat_first_poly_coord: bool = False
) -> List[NDArray]:
    """Returns a set of bounding boxes/polygons enclosing regions in the given mask

    This works by using the OpenCV findContours() function to detect regions in the mask

    IMPORTANT: Assumes no hierarchy/nesting of regions

    Can specify crop_location which accepts box coordinates (x1, y1, x2, y2) of the region to crop
    first before looking for annotation

    Args:
        mask: The mask to detect annotations in. Should be one-hot encoded and have no channels
            dimension (i.e. [H,W]).
        annotation: The type of annotation to retrieve. Can be either 'boxes' or 'polygons'.
        min_sidelength: The minimum sidelength of a box or polygon to be returned.
        crop_location: The x1, y1, x2, y2 coordinates of the crop to be taken before searching for
            annotations. Note: All annotations are returned relative to the whole image.
        repeat_first_poly_coord: If extracting polygons, this determines if the last coordinate of
            the polygon will be forced to equal the first coordinate of the polygon.

    Returns:
        The list of bounding boxes or polygons enclosing regions in the mask. Bounding boxes are of
            the form: [x1, y1, x2, y2], polygons are of the form: [(x1, y1), (x2, y2), ...].
    """
    # Validate annotation
    if annotation not in ('boxes', 'polygons'):
        raise ValueError(f'Requested annotation must be \'boxes\' or \'polygons\'. '
                         f'Is: {annotation}')

    # Put mask in format expected by OpenCV ([H,W,1], uint8)
    mask = np.expand_dims(mask, 2).astype(np.uint8)

    # Crop the mask if required
    if crop_location is not None:
        mask = crop_image(mask, crop_location)

    # Find contours (we don't care about hierarchy). Contours come back as: (Nx1x2)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if annotation == 'boxes':
        # Convert contours to boxes
        annots = [polygon_to_region(contour.squeeze(1)) for contour in contours]
        annots = [annot for annot in annots if get_region_smaller_edge(annot) >= min_sidelength]
        # Handle crop offset
        if crop_location is not None:
            for annot in annots:
                annot[[0, 2]] += crop_location[0]
                annot[[1, 3]] += crop_location[1]
    elif annotation == 'polygons':
        annots = [contour.squeeze(1) for contour in contours]
        annots = [annot for annot in annots if
                  get_region_smaller_edge(polygon_to_region(annot)) >= min_sidelength]
        # Handle crop offset
        if crop_location is not None:
            for annot in annots:
                annot += crop_location[:2]
        # Repeat the first polygon coordinate if requested
        if repeat_first_poly_coord:
            for annot_idx, annot in enumerate(annots):
                if not np.array_equal(annot[-1], annot[0]):
                    annots[annot_idx] = np.concatenate([annot, [annot[0]]], axis=0)
    else:
        raise RuntimeError(f'Error. Invalid annotation type: \'{annotation}\'')
    return annots


def integer_encoded_mask_to_annotation(mask, class_idx, annotation='boxes',
                                       min_sidelength=8, crop_location=None):
    """Returns a set of bounding boxes/polygons enclosing regions in the given mask

    Note: This is currently implemented by creating a mask of zeros, with non-zero values defined
        as where they = the class index, then calling the equivalent one-hot function

    Args:
        mask (np.uint8 ndarray): The mask to detect annotations in.
        class_idx (int): The index of the class to find enclosing regions of
        annotation (str): The type of annotation to retrieve. Can be either 'boxes' or 'polygons'
        min_sidelength (int): The minimum sidelength of a box or polygon to be returned
        crop_location (4-tuple/None): The x1, y1, x2, y2 coordinates of the crop to be taken before
            searching for annotations. Note: All annotations are returned relative to the whole
            image

    Returns:
        (list): The list of bounding boxes or polygons enclosing regions in the mask. Bounding
            boxes are of the form: [x1, y1, x2, y2], polygons are of the form:
            [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    # Create a bool mask, set to 1s based on matching class indexes in the original mask
    bool_mask = np.zeros(shape=mask.shape, dtype=np.bool_)
    bool_mask[mask == class_idx] = 1

    # Call the function usable with a one-hot mask
    return one_hot_mask_to_annotation(
        bool_mask, annotation=annotation, min_sidelength=min_sidelength,
        crop_location=crop_location)
