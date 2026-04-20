# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image dimensions utils for data accessors."""

import dataclasses
import math
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import image_dimension_type
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate
from serving.logging_lib import cloud_logging_client


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
ImageDimensions = image_dimension_type.ImageDimensions


@dataclasses.dataclass(frozen=True)
class ProjectedPatch:
  start_x: int  # X coordinate of the patch in the source image
  start_y: int  # Y coordinate of the patch in the source image
  projected_read_width: int  # Width of sampled region in source image
  projected_read_height: int  # Height of sampled region in source image
  rescale_width: int  # Width of resized patch in patch coordinate system
  rescale_height: int  # Height of resized patch in patch coordinate system
  clip_start_x: int  # Clip offset for x coordinate in the resized patch memory
  clip_start_y: int  # Clip offset for y coordinate of the resized patch memory


def get_resize_image_dimensions(
    extensions: Mapping[str, Any], max_dimension: Optional[int] = None
) -> Optional[ImageDimensions]:
  """Returns optional dimension to resize input imaging level to."""
  value = extensions.get(_InstanceJsonKeys.IMAGE_DIMENSIONS, {})
  if not value:
    return None
  try:
    image_dim = ImageDimensions(**value)
  except TypeError as exp:
    if not isinstance(value, dict):
      msg = f'{_InstanceJsonKeys.IMAGE_DIMENSIONS} is not a dictionary.'
      cloud_logging_client.info(msg, exp)
      raise data_accessor_errors.InvalidRequestFieldError(msg) from exp
    keys = f'{_InstanceJsonKeys.WIDTH}, {_InstanceJsonKeys.HEIGHT}'
    msg = (
        f'{_InstanceJsonKeys.IMAGE_DIMENSIONS} dict'
        f' has invalid keys; expecting: {keys}'
    )
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.InvalidRequestFieldError(msg) from exp
  try:
    width = json_validation_utils.validate_int(image_dim.width)
    height = json_validation_utils.validate_int(image_dim.height)
  except json_validation_utils.ValidationError as exp:
    msg = 'Invalid dimensions(width and/or height).'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.InvalidRequestFieldError(msg) from exp
  if height <= 0 or width <= 0:
    msg = 'Image dimensions(width and/or height) are not positive integers.'
    cloud_logging_client.info(msg)
    raise data_accessor_errors.InvalidRequestFieldError(msg)
  if max_dimension is not None and (
      height > max_dimension or width > max_dimension
  ):
    msg = (
        f'Image width and/or height exceeds max dimension ({max_dimension} px)'
        ' supported by the endpoint.'
    )
    cloud_logging_client.info(msg)
    raise data_accessor_errors.ImageDimensionError(msg)
  return image_dim


def resize_image_dimensions(
    image: np.ndarray,
    image_dimensions: Optional[ImageDimensions],
) -> np.ndarray:
  """Resizes image to the given dimensions."""
  if image_dimensions is None:
    return image
  if (
      image.shape[1] == image_dimensions.width
      and image.shape[0] == image_dimensions.height
  ):
    return image
  current_area = image.shape[0] * image.shape[1]
  new_area = image_dimensions.width * image_dimensions.height
  new_dim = (image_dimensions.width, image_dimensions.height)
  if new_area <= current_area:
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
  else:
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
  # ensure single channel resized image shape matches input image shape
  if resized_image.ndim == 2 and image.ndim == 3 and image.shape[-1] == 1:
    return np.expand_dims(resized_image, axis=-1)
  return resized_image


def _scale_pt(pt: int, source_dim: int, target_dim: int) -> float:
  return pt * target_dim / source_dim


def _rescale_dim(
    start_pt: int,
    sample_dim: int,
    source_level_dim: int,
    target_level_dim: int,
) -> tuple[int, int, int, int]:
  """Rescales dimensions from source level to target level."""
  # start and end to reference imaging.
  new_pt_abs_start = _scale_pt(start_pt, source_level_dim, target_level_dim)
  new_pt_abs_end = _scale_pt(
      start_pt + sample_dim, source_level_dim, target_level_dim
  )
  if target_level_dim >= source_level_dim:
    # compute closest starting pixel in imaging source level
    new_pt_abs_start = int(round(new_pt_abs_start))
    # compute closest edge pixel in imaging source level
    new_pt_abs_end = int(round(new_pt_abs_end))
    # Starting pixel, width, dimension of imaging in source level, clip offset
    return new_pt_abs_start, new_pt_abs_end - new_pt_abs_start, sample_dim, 0
  else:
    # floor projected starting position and take ceil of end position
    # to fully span the dimensions of the imaging.
    new_pt_abs_start = math.floor(new_pt_abs_start)
    new_pt_abs_end = math.ceil(new_pt_abs_end)
    # scale projected positions back to output level to determine
    # the size of the sampled region.
    start_pos = int(
        round(_scale_pt(new_pt_abs_start, target_level_dim, source_level_dim))
    )
    # compute end pixel position in output imaging.
    end_pos = int(
        round(_scale_pt(new_pt_abs_end, target_level_dim, source_level_dim))
    )
    return (
        new_pt_abs_start,  # start of sampled region
        new_pt_abs_end - new_pt_abs_start,  # width sampled region
        end_pos - start_pos,  # width of sampled region
        start_pt - start_pos,  # offset for sampled region
    )


def get_projected_patch(
    pc: patch_coordinate.PatchCoordinate,
    source_level_width: int,
    source_level_height: int,
    projected_level_dimensions: Optional[ImageDimensions],
) -> ProjectedPatch:
  """Returns projection that transforms pixels from source dim to projected dim.

  Args:
    pc: Patch coordinate defining a region in a projected dim.
    source_level_width: Width of imaging that is being used as the source for
      the patch defined in pc.
    source_level_height: Height of imaging that is being used as the source for
      the patch defined in pc.
    projected_level_dimensions: Dimensions of the imaging that patch (pc) is
      being defined. If None then the patch image dimensions == source image
      dimensions.

  Returns:
    Parameters which define the patch projection from source image into
    patch coordinate system.
  """
  if projected_level_dimensions is None:
    return ProjectedPatch(
        start_x=pc.x_origin,
        start_y=pc.y_origin,
        projected_read_width=pc.width,
        projected_read_height=pc.height,
        rescale_width=pc.width,
        rescale_height=pc.width,
        clip_start_x=0,
        clip_start_y=0,
    )
  start_x, width, rescale_width, clip_start_x = _rescale_dim(
      pc.x_origin,
      pc.width,
      projected_level_dimensions.width,
      source_level_width,
  )
  start_y, height, rescale_height, clip_start_y = _rescale_dim(
      pc.y_origin,
      pc.height,
      projected_level_dimensions.height,
      source_level_height,
  )
  return ProjectedPatch(
      start_x=start_x,
      start_y=start_y,
      projected_read_width=width,
      projected_read_height=height,
      rescale_width=rescale_width,
      rescale_height=rescale_height,
      clip_start_x=clip_start_x,
      clip_start_y=clip_start_y,
  )


def resize_projected_patch(
    pc: patch_coordinate.PatchCoordinate,
    resize_projected_dim: ProjectedPatch,
    memory: np.ndarray,
) -> np.ndarray:
  """Resizes and crops memory to project patch into pc coordinate system.

  Args:
    pc: Patch coordinate defining a region in a projected dim.
    resize_projected_dim: Paramters which define the projection from sampled
      region in memory to pc coordiante system.
    memory: Pixel data sampled from an image which != projected dim.

  Returns:
    resized and cropped pixel imaging projected into pc coordinate system.
  """
  memory = resize_image_dimensions(
      memory,
      ImageDimensions(
          resize_projected_dim.rescale_width,
          resize_projected_dim.rescale_height,
      ),
  )
  return memory[
      resize_projected_dim.clip_start_y : resize_projected_dim.clip_start_y
      + pc.height,
      resize_projected_dim.clip_start_x : resize_projected_dim.clip_start_x
      + pc.width,
      ...,
  ]
