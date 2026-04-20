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

"""Shared dataclasses across requests and responses for Pete."""

import dataclasses
from typing import Any, List, Mapping, Sequence

import numpy as np

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import image_dimension_type
from data_accessors.utils import json_validation_utils
from serving.logging_lib import cloud_logging_client

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_TRUE = 'TRUE'
_FALSE = 'FALSE'


class InvalidCoordinateError(Exception):
  pass


def patch_required_to_be_fully_in_source_image(
    extensions: Mapping[str, Any],
) -> bool:
  """Returns true (default) if patches are required to be fully in image.

  Args:
    extensions: A string key dictionary of JSON formatted metadata.

  Returns:
    True if patches are required to be fully in image.
  """
  value = extensions.get(
      _InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE,
      True,
  )
  if isinstance(value, str):
    value = value.upper()
    if value == _TRUE:
      return True
    elif value == _FALSE:
      return False
  if not isinstance(value, bool):
    msg = (
        f'{_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE} is'
        ' not a boolean.'
    )
    cloud_logging_client.info(msg)
    raise data_accessor_errors.InvalidRequestFieldError(msg)
  return value


@dataclasses.dataclass(frozen=True)
class PatchCoordinate:
  """A coordinate of a patch."""

  x_origin: int
  y_origin: int
  width: int
  height: int

  def validate_patch_in_dim(
      self,
      image_shape: image_dimension_type.ImageDimensions,
  ) -> None:
    """Validates coordinate in dimension."""
    if (
        self.y_origin < 0
        or self.x_origin < 0
        or self.y_origin + self.height > image_shape.height
        or self.x_origin + self.width > image_shape.width
    ):
      raise data_accessor_errors.PatchOutsideOfImageDimensionsError(
          'Patch coordinates falls outside of image dimensions.'
      )


def create_patch_coordinate(
    patch_coordinate_map: Mapping[str, Any],
    default_width: int,
    default_height: int,
    require_patch_dim_match_default_dim: bool = False,
) -> PatchCoordinate:
  """Creates a patch coordinate."""
  x_origin = int(patch_coordinate_map[_InstanceJsonKeys.X_ORIGIN])
  y_origin = int(patch_coordinate_map[_InstanceJsonKeys.Y_ORIGIN])
  width = int(patch_coordinate_map.get(_InstanceJsonKeys.WIDTH, default_width))
  height = int(
      patch_coordinate_map.get(_InstanceJsonKeys.HEIGHT, default_height)
  )
  if require_patch_dim_match_default_dim and (
      width != default_width or height != default_height
  ):
    raise data_accessor_errors.PatchCoordinateError(
        'Patch coordinate width and height must be'
        f' {default_width}x{default_height}.'
    )
  return PatchCoordinate(
      x_origin=x_origin,
      y_origin=y_origin,
      width=width,
      height=height,
  )


def parse_patch_coordinates(
    patch_coordinates: Sequence[Mapping[str, Any]],
    default_width: int,
    default_height: int,
    require_patch_dim_match_default_dim: bool,
) -> List[PatchCoordinate]:
  """Returns patch coodianates."""
  result = []
  if not isinstance(patch_coordinates, List):
    raise InvalidCoordinateError('patch_coordinates is not list')
  for patch_coordinate in patch_coordinates:
    try:
      pc = create_patch_coordinate(
          patch_coordinate,
          default_width,
          default_height,
          require_patch_dim_match_default_dim,
      )
    except TypeError as exp:
      if not isinstance(patch_coordinate, dict):
        raise InvalidCoordinateError('Patch coordinate is not dict.') from exp
      keys = ', '.join(
          list(
              dataclasses.asdict(
                  create_patch_coordinate(
                      {
                          _InstanceJsonKeys.X_ORIGIN: 0,
                          _InstanceJsonKeys.Y_ORIGIN: 0,
                      },
                      default_width,
                      default_height,
                      require_patch_dim_match_default_dim,
                  )
              )
          )
      )
      raise InvalidCoordinateError(
          f'Patch coordinate dict has invalid keys; expecting: {keys}'
      ) from exp
    try:
      json_validation_utils.validate_int(pc.x_origin)
      json_validation_utils.validate_int(pc.y_origin)
      json_validation_utils.validate_int(pc.width)
      json_validation_utils.validate_int(pc.height)
    except json_validation_utils.ValidationError as exp:
      raise InvalidCoordinateError(
          f'Invalid patch coordinate; x_origin: {pc.x_origin}, y_origin:'
          f' {pc.y_origin}, width: {pc.width}, height: {pc.height}'
      ) from exp
    result.append(pc)
  return result


def get_patch_from_memory(
    pc: PatchCoordinate, memory: np.ndarray) -> np.ndarray:
  """Returns a patch from memory."""
  mem_width = memory.shape[1]
  mem_height = memory.shape[0]
  if (
      pc.y_origin >= 0
      and pc.x_origin >= 0
      and pc.y_origin + pc.height <= mem_height
      and pc.x_origin + pc.width <= mem_width
  ):
    return memory[
        pc.y_origin : pc.y_origin + pc.height,
        pc.x_origin : pc.x_origin + pc.width,
        ...,
    ]
  mem_shape = list(memory.shape)
  mem_shape[0] = pc.height
  mem_shape[1] = pc.width
  copy_memory = np.zeros(
      tuple(mem_shape),
      dtype=memory.dtype,
  )
  # test patch intersects with memory
  if (
      pc.x_origin + pc.width > 0
      and pc.y_origin + pc.height > 0
      and pc.x_origin < mem_width
      and pc.y_origin < mem_height
  ):
    pc_x_origin = max(0, pc.x_origin)
    pc_y_origin = max(0, pc.y_origin)
    pc_x_end = min(pc.x_origin + pc.width, mem_width)
    pc_y_end = min(pc.y_origin + pc.height, mem_height)
    mem_x_start = max(0, -pc.x_origin)
    mem_y_start = max(0, -pc.y_origin)
    copy_width = pc_x_end - pc_x_origin
    copy_height = pc_y_end - pc_y_origin
    copy_memory[
        mem_y_start : mem_y_start + copy_height,
        mem_x_start : mem_x_start + copy_width,
        ...,
    ] = memory[pc_y_origin:pc_y_end, pc_x_origin:pc_x_end, ...]
  return copy_memory
