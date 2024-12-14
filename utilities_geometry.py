"""Helper functions relating to geometry computation"""
from typing import Tuple, Optional, Union, Sequence, List

import numpy as np
import torch
from numpy.typing import NDArray
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import polygonize, unary_union


# ##################################################################################################
#                               Geometry Intersection Helpers
# ##################################################################################################
def point_intersects_region(region: Sequence[int], point: Sequence[int]) -> bool:
    """Given a point, determines if it lies within a region

    Args:
        region: The region given as [x1, y1, x2, y2]
        point: The point to test (given as (x, y))

    Returns:
        True/False whether the point belongs to the region
    """
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def region_intersects_region(region_1: Sequence[int], region_2: Sequence[int],
                             touching_borders_intersect: bool = True) -> bool:
    """Given a region, determines if it intersects another region

    Solution adapted from here: https://www.tutorialspoint.com/rectangle-overlap-in-python

    Note in the original solution, the conditions were inclusive (meaning borders touching were
    considered non-intersecting). In this implementation, border touching behaviour is determined by
    touching_borders_intersect.

    Args:
        region_1: The first region given as [x1, y1, x2, y2]
        region_2: The second region given as [x1, y1, x2, y2]
        touching_borders_intersect: Whether two borders touching should be considered intersecting or not

    Returns:
        True/False whether the regions intersect
    """
    if touching_borders_intersect:
        return not ((region_1[0] > region_2[2]) or (region_1[2] < region_2[0]) or
                    (region_1[3] < region_2[1]) or (region_1[1] > region_2[3]))
    return not ((region_1[0] >= region_2[2]) or (region_1[2] <= region_2[0]) or
                (region_1[3] <= region_2[1]) or (region_1[1] >= region_2[3]))


def region_contains_region(region_1: Sequence[int], region_2: Sequence[int]) -> bool:
    """Given a region, determines if it directly contains another region

    Args:
        region_1: The first region given as [x1, y1, x2, y2]
        region_2: The second region given as [x1, y1, x2, y2]

    Returns:
        True/False whether the region_2 is fully contained in region_1
    """
    return (region_1[0] <= region_2[0] <= region_2[2] <= region_1[2]) and \
           (region_1[1] <= region_2[1] <= region_2[3] <= region_1[3])


def region_contains_polygon(region: Sequence[int], polygon: Sequence[Sequence[int]]) -> bool:
    """Given a region, determines if it directly contains a polygon

    Args:
        region: The region given as [x1, y1, x2, y2]
        polygon: The Nx2 polygon to test

    Returns:
        True/False whether the region contains the polygon
    """
    # Convert region as [x1, y1, x2, y2] to polygon as [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    region_as_polygon = Polygon([
        [region[0], region[1]], [region[2], region[1]],
        [region[2], region[3]], [region[0], region[3]],
        [region[0], region[1]],
    ])
    polygon_as_polygon = Polygon(polygon)
    return region_as_polygon.contains(polygon_as_polygon)


def polygon_intersects_region(region: Sequence[int], polygon: Sequence[Sequence[int]]) -> bool:
    """Given a polygon, determines if it intersects a region

    This makes use of the shapely library

    Args:
        region: The region given as [x1, y1, x2, y2]
        polygon: The Nx2 polygon to test

    Returns:
        True/False whether the polygon intersects the region
    """
    # Convert region as [x1, y1, x2, y2] to polygon as [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    region_as_polygon = Polygon([
        [region[0], region[1]], [region[2], region[1]],
        [region[2], region[3]], [region[0], region[3]],
        [region[0], region[1]],
    ])
    polygon_as_polygon = Polygon(polygon)
    return polygon_as_polygon.intersects(region_as_polygon)


def polygon_contains_polygon(
        polygon_1: Sequence[Sequence[int]], polygon_2: Sequence[Sequence[int]]
) -> bool:
    """Given a polygon, determines if it directly contains another polygon

    That is, True if polygon 1 contains polygon 2

    Args:
        polygon_1: The first Nx2 polygon
        polygon_2: The second Nx2 polygon

    Returns:
        True/False whether polygon_2 is fully contained in polygon_1
    """
    polygon_1 = Polygon(polygon_1)
    polygon_2 = Polygon(polygon_2)
    return polygon_1.contains(polygon_2)


# ##################################################################################################
#                                   Validating Geometry
# ##################################################################################################
def polygon_is_valid(polygon: Sequence[Sequence[int]]) -> bool:
    """Determines if a polygon is valid assuming the shapely definition

    Args:
        polygon: The Nx2 polygon to assess

    Returns:
        True/False if the polygon is valid
    """
    return Polygon(polygon).is_valid


def polygon_is_multi_polygon(polygon: Sequence[Sequence[int]]) -> bool:
    """Determines if a polygon is a multi polygon

    Args:
        polygon: The Nx2 polygon to assess

    Returns:
        True/False if the polygon is a multi polygon
    """
    return Polygon(polygon).geom_type == 'MultiPolygon'


def polygon_to_multi_polygon(polygon: Sequence[Sequence[int]]) -> MultiPolygon:
    """Given an invalid self-intersecting polygon, converts it to a MultiPolygon

    Based on: https://stackoverflow.com/a/35119152

    Args:
        polygon: The Nx2 polygon to split

    Returns:
        A MultiPolygon representing that polygon
    """
    # Create a linestring from the data
    ls = LineString(polygon)

    # Create a linear ring from that data (should be non-simple)
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    if lr.is_simple:
        raise RuntimeError(f'Error. Linear ring is simple, unsure how to handle!')

    # Turn into a multiline string
    mls = unary_union(lr)

    # Turn the multiline string into a MultiPolygon (and return it)
    return MultiPolygon(polygonize(mls))


# ##################################################################################################
#                                   General Geometry Helpers
# ##################################################################################################
def get_region_dimensions(region: Sequence[int], as_int: bool = True) -> Tuple[int, int]:
    """Given a region, returns the width & height of the region

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The width and height of the region
    """
    w = region[2] - region[0]
    h = region[3] - region[1]
    if as_int:
        return int(w), int(h)
    return w, h


def get_region_area(region: Sequence[int]) -> int:
    """Given a region, returns the area of the region

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The area of the region
    """
    dimensions = get_region_dimensions(region)
    return dimensions[0] * dimensions[1]


def get_region_smaller_edge(region: Sequence[int]) -> int:
    """Given a region, returns the length of the smallest edge

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The length of the smaller edge
    """
    return min(get_region_dimensions(region))


def get_region_larger_edge(region: Sequence[int]) -> int:
    """Given a region, returns the length of the largest edge

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The length of the larger edge
    """
    return max(get_region_dimensions(region))


def get_region_centre(region: Sequence[int]) -> Tuple[float, float]:
    """Given a region, returns the centre of the region

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The centre x and y position of the region
    """
    return (region[0] + region[2]) / 2, (region[1] + region[3]) / 2


def get_polygon_rectangular_dimensions(polygon: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """Given a polygon, returns the width & height of the polygon

    Args:
        polygon: The Nx2 polygon to get the dimensions of

    Returns:
        The width and height of the polygon
    """
    return get_region_dimensions(polygon_to_region(polygon))


def get_polygon_rectangular_area(polygon: Sequence[Sequence[int]]) -> int:
    """Given a polygon, returns the rectangular area of the polygon

    That is, the box that encompasses the polygon

    Args:
        polygon: The Nx2 polygon to get the area of

    Returns:
        The rectangular area of the polygon
    """
    return get_region_area(polygon_to_region(polygon))


def get_polygon_centroid(polygon: Sequence[Sequence[int]]) -> Tuple[float, float]:
    """Given a polygon, returns the centre of mass of the polygon

    Args:
        polygon: The Nx2 polygon to get the centre of

    Returns:
        The centre x and y position of the polygon
    """
    centroid_point = Polygon(polygon).centroid
    return centroid_point.x, centroid_point.y


def get_polygon_area(polygon: Sequence[Sequence[int]]) -> float:
    """Given a polygon, returns the area of the polygon.

    Args:
        polygon: The Nx2 polygon to get the area of

    Returns:
        The area of the polygon
    """
    return Polygon(polygon).area


# ##################################################################################################
#                               Geometry Conversion Helpers
# ##################################################################################################
def polygon_to_region(polygon):
    """Given a polygon, converts it to a bounding box encompassing the polygon

    Does this by finding min/max x/y coordinates and concatenating them together

    Args:
        polygon (numpy ndarray): The polygon to convert

    Returns:
        ((4,) ndarray): The bounding coordinates of the polygon in form: x1, y1, x2, y2
    """
    return np.concatenate((polygon.min(axis=0), polygon.max(axis=0)))


def polygon_to_shapely_polygon(polygon):
    """Given a polygon in a numpy array/list, converts it to a shapely polygon

    Args:
        polygon (Nx2 ndarray): The polygon to convert

    Returns:
        (Polygon): The shapely polygon
    """
    return Polygon(polygon)


def flatten_multi_polygon(coordinates):
    """Flattens nested lists of polygons.

    This works on a MultiPolygon of any given dimensionality.

    Returns:
        A flattened list of polygons, ``[<polygon_1>, <polygon_2>, ...]``.
    """
    is_single_polygon = (
        all(len(poly) == 2 for poly in coordinates)
        and all(not isinstance(poly[0], (list, np.ndarray)) for poly in coordinates)
        and all(not isinstance(poly[1], (list, np.ndarray)) for poly in coordinates)
    )

    if is_single_polygon:
        return [coordinates]

    flattened_polygon = []
    for polygon in coordinates:
        flattened_polygon.extend(flatten_multi_polygon(polygon))
    return flattened_polygon


# ##################################################################################################
#                                   Geometry Creation
# ##################################################################################################
def create_region_from_point(region_centre, width, height, as_int=False, round_int=False):
    """Creates a new box region given a centrepoint and width/height

    Args:
        region_centre (tuple/list): x, y coordinates of region centre
        width (int/float): The desired width of the region
        height (int/float): The desired height of the region
        as_int (bool): Whether coordinates should be cast to integers before returning
        round_int (bool): If as_int is True, if they should be rounded or simply cast to integers

    Returns:
        (tuple len(4)): The top left/bottom right box coordinates, given as: [x1, y1, x2, y2]
    """
    x1 = float(region_centre[0] - width / 2)
    y1 = float(region_centre[1] - height / 2)
    x2 = float(region_centre[0] + width / 2)
    y2 = float(region_centre[1] + height / 2)

    if as_int:
        if not round_int:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        else:
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        # Handle any rounding errors introduced (enforce width/height are as expected)
        if x2 - x1 != width:
            x2 = x1 + width
        if y2 - y1 != height:
            y2 = y1 + height

    return x1, y1, x2, y2


def create_region_from_cxcywh(cx, cy, w, h):
    """Given centre x, y and width, height, creates a region of the form: [x1, y1, x2, y2]"""
    return cx - w//2, cy - h//2, cx + w//2, cy + h//2


# ##################################################################################################
#                                   Geometry Modification
# ##################################################################################################
def contain_region_within_space(region, space):
    """Given a region, contains it within a given space

    Args:
        region (list/tuple/np.ndarray of length 4): The region given as [x1, y1, x2, y2]
        space (list/tuple/np.ndarray of length 4): The space the region should be contained within
            given as [x1, y1, x2, y2]

    Returns:
        (tuple of 4 ints/floats): Top region coordinates such that they are bound within the space
    """
    x0 = max(region[0], space[0])
    y0 = max(region[1], space[1])
    x1 = min(region[2], space[2])
    y1 = min(region[3], space[3])
    return x0, y0, x1, y1


def calculate_cropped_size(
    input_size: Tuple[int, int],
    crop_margin: Optional[int],
) -> Tuple[int, int]:
    """Calculate the output size given an input size and crop margin.

    Args:
        input_size: The size of the box before cropping.
        crop_margin: The amount to crop from each side of the box.

    Returns:
        The size of the box after cropping.
    """
    if crop_margin is None:
        return input_size
    output_size = (input_size[0] - crop_margin * 2, input_size[1] - crop_margin * 2)
    if any(x < 0 for x in output_size):
        raise ValueError(f'Crop margin {crop_margin} is too large for the input size {input_size}')
    return output_size


def rotate_points_in_image_180(points: NDArray, image_shape: Tuple[int, int]) -> NDArray:
    """Rotates a set of points in an image by 180 degrees.

    Args:
        points: The set of (x, y) points to be rotated. Should be a numpy array with final
            dimension of shape (2,).
        image_shape: The width and height of the image.

    Returns:
        points: The set of points rotated by 180 degrees about the image centre
    """
    return image_shape - points


# ##################################################################################################
#                                   Geometry Comparison
# ##################################################################################################
def compute_pairwise_euclidean_distance(points_a, points_b):
    """Computes the pairwise Euclidean distance between two sets of points.

    This works for both numpy arrays and torch tensors. The return type is determined by the type
    of points_a.

    Args:
        points_a: The (N, D) first set of points.
        points_b: The (M, D) second set of points.

    Returns:
        The (N, M) pairwise Euclidean distance.
    """
    # Validate dimensionality
    if points_a.shape[1] != points_b.shape[1]:
        raise ValueError(f'Dimensionality mismatch. Second dimension should have same size for '
                         f'both sets of points. Points a: {points_a.shape}, Points b: '
                         f'{points_b.shape}')

    # Determine if data is numpy or torch
    is_torch = isinstance(points_a, torch.Tensor)

    # Broadcast to return dimensionality of (N, M), computing Euclidean distance on D axis
    if is_torch:
        return torch.linalg.norm(points_a.float()[:, None, :] - points_b.float()[None, :, :], dim=-1)
    return np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=-1)
