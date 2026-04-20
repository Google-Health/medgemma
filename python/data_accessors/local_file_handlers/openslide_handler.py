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
"""local handler for handling openslide image files."""

import contextlib
import dataclasses
import io
import os
import tempfile
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import openslide
from PIL import ImageCms

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


@dataclasses.dataclass(frozen=True)
class EndpointInputDimensions:
  """Holds endpoint input dimensions."""

  width_px: int
  height_px: int


@dataclasses.dataclass(frozen=True)
class PixelSpacing:
  """Holds pixel spacing information."""

  width_mm_per_px: float
  height_mm_per_px: float


def _get_patch_from_memory(
    slide: openslide.OpenSlide,
    slide_level: int,
    projected_patch: image_dimension_utils.ProjectedPatch,
    level_width: int,
    level_height: int,
) -> np.ndarray:
  """Returns a patch from memory."""
  if (
      projected_patch.projected_read_height
      * projected_patch.projected_read_width
      > 10000 * 10000
  ):
    raise data_accessor_errors.InvalidRequestFieldError(
        'OpenSlide patch dimensions exceed 100,000,000 pixels.'
    )
  level_0_width, level_0_height = slide.dimensions
  if (
      projected_patch.start_x >= 0
      and projected_patch.start_y >= 0
      and projected_patch.start_y + projected_patch.projected_read_height
      <= level_height
      and projected_patch.start_x + projected_patch.projected_read_width
      <= level_width
  ):
    memory = np.asarray(
        slide.read_region(
            (
                int(
                    round(projected_patch.start_x * level_0_width / level_width)
                ),
                int(
                    round(
                        projected_patch.start_y * level_0_height / level_height
                    )
                ),
            ),
            slide_level,
            (
                projected_patch.projected_read_width,
                projected_patch.projected_read_height,
            ),
        )
    )
    return memory[..., 0:3]
  copy_memory = np.zeros(
      (
          projected_patch.projected_read_height,
          projected_patch.projected_read_width,
          3,
      ),
      dtype=np.uint8,
  )
  # test patch intersects with memory
  if (
      projected_patch.start_x + projected_patch.projected_read_width > 0
      and projected_patch.start_y + projected_patch.projected_read_height > 0
      and projected_patch.start_x < level_width
      and projected_patch.start_y < level_height
  ):
    pc_x_origin = max(0, projected_patch.start_x)
    pc_y_origin = max(0, projected_patch.start_y)
    pc_x_end = min(
        projected_patch.start_x + projected_patch.projected_read_width,
        level_width,
    )
    pc_y_end = min(
        projected_patch.start_y + projected_patch.projected_read_height,
        level_height,
    )
    mem_x_start = max(0, -projected_patch.start_x)
    mem_y_start = max(0, -projected_patch.start_y)
    copy_width = pc_x_end - pc_x_origin
    copy_height = pc_y_end - pc_y_origin

    memory = np.asarray(
        slide.read_region(
            (
                int(round(pc_x_origin * level_0_width / level_width)),
                int(round(pc_y_origin * level_0_height / level_height)),
            ),
            slide_level,
            (copy_width, copy_height),
        )
    )
    copy_memory[
        mem_y_start : mem_y_start + copy_height,
        mem_x_start : mem_x_start + copy_width,
        ...,
    ] = memory[..., 0:3]
  return copy_memory


def _get_patch(
    slide: openslide.OpenSlide,
    slide_level: int,
    resize_image_dimensions: Optional[image_dimension_utils.ImageDimensions],
    pc: patch_coordinate_module.PatchCoordinate,
    validate_patch_in_dim: bool,
) -> np.ndarray:
  """Returns a patch from a openslide image."""
  level_width, level_height = slide.level_dimensions[slide_level]
  if validate_patch_in_dim:
    if resize_image_dimensions is None:
      pc.validate_patch_in_dim(
          image_dimension_utils.ImageDimensions(level_width, level_height)
      )
    else:
      pc.validate_patch_in_dim(resize_image_dimensions)
  projected_patch = image_dimension_utils.get_projected_patch(
      pc, level_width, level_height, resize_image_dimensions
  )
  memory = _get_patch_from_memory(
      slide,
      slide_level,
      projected_patch,
      level_width,
      level_height,
  )
  if memory.shape[0] != pc.height or memory.shape[1] != pc.width:
    return image_dimension_utils.resize_projected_patch(
        pc, projected_patch, memory
    )
  return memory


def _create_icc_profile_image_transformation(
    source_icc_profile: Optional[ImageCms.core.CmsProfile],
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
) -> Optional[ImageCms.ImageCmsTransform]:
  """Transforms image to target ICC profile."""
  if source_icc_profile is None or target_icc_profile is None:
    return None
  return icc_profile_utils.create_icc_profile_transformation(
      source_icc_profile, target_icc_profile
  )


def _decode_open_slide_image(
    slide: openslide.OpenSlide,
    slide_level: int,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
    patch_coordinates: Sequence[patch_coordinate_module.PatchCoordinate],
    resize_image_dimensions: Optional[image_dimension_utils.ImageDimensions],
    patch_required_to_be_fully_in_source_image: bool,
) -> Iterator[np.ndarray]:
  """Decode decode openslide encoded patches."""
  if not patch_coordinates:
    if resize_image_dimensions is None:
      width, height = slide.level_dimensions[slide_level]
    else:
      width = resize_image_dimensions.width
      height = resize_image_dimensions.height
    patch_coordinates = [
        patch_coordinate_module.PatchCoordinate(0, 0, width, height)
    ]
  try:
    slide_color_profile = slide.color_profile  # pytype: disable=attribute-error
  except AttributeError:
    slide_color_profile = None
  icc_profile_image_transformation = _create_icc_profile_image_transformation(
      slide_color_profile, target_icc_profile
  )
  level_width, level_height = slide.level_dimensions[slide_level]
  if (
      resize_image_dimensions is not None
      and resize_image_dimensions.width == level_width
      and resize_image_dimensions.height == level_height
  ):
    resize_image_dimensions = None
  for pc in patch_coordinates:
    decoded_image_bytes = _get_patch(
        slide,
        slide_level,
        resize_image_dimensions,
        pc,
        patch_required_to_be_fully_in_source_image,
    )
    if icc_profile_image_transformation is not None:
      decoded_image_bytes = (
          icc_profile_utils.transform_image_bytes_to_target_icc_profile(
              decoded_image_bytes, icc_profile_image_transformation
          )
      )
    yield decoded_image_bytes


def _get_open_slide_level_from_int(
    openslide_level: int, slide: openslide.OpenSlide
) -> int:
  """Get Openslide level from level index."""
  if openslide_level >= 0:
    return min(openslide_level, slide.level_count - 1)
  else:
    openslide_level = slide.level_count + openslide_level
    return max(openslide_level, 0)


def _get_open_slide_level_from_dimensions(
    target_dim: image_dimension_utils.ImageDimensions, slide
) -> int:
  """Get Openslide level from image dimensions."""
  for level, dim in enumerate(slide.level_dimensions):
    if dim[0] < target_dim.width or dim[1] < target_dim.height:
      return max(0, level - 1)
  return slide.level_count - 1


def _get_open_slide_level_from_pixel_spacing(
    target_ps: PixelSpacing, slide
) -> int:
  """Get Openslide level from pixel spacing."""
  # Use same pixel spacing metrics as DICOM mm/px
  mm_per_px = float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) / 1000
  mm_per_py = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y]) / 1000
  width_mm, height_mm = slide.dimensions
  width_mm *= mm_per_px
  height_mm *= mm_per_py
  for level, dim in enumerate(slide.level_dimensions):
    level_width, level_height = dim
    if (width_mm / level_width > target_ps.width_mm_per_px) or (
        height_mm / level_height > target_ps.height_mm_per_px
    ):
      return max(0, level - 1)
  return slide.level_count - 1


def _get_default_openslide_level(
    slide: openslide.OpenSlide,
    patch_coordinates: Sequence[patch_coordinate_module.PatchCoordinate],
    endpoint_input_dim: EndpointInputDimensions,
) -> int:
  """Returns the default openslide level if none is provided."""
  if not patch_coordinates:
    for index in range(slide.level_count - 1, 0, -1):
      width, height = slide.level_dimensions[index]
      if (
          width >= endpoint_input_dim.width_px
          and height >= endpoint_input_dim.height_px
      ):
        return index
  return 0


def _parse_openslide_level(
    base_request: Mapping[str, Any],
) -> Optional[Union[int, image_dimension_utils.ImageDimensions, PixelSpacing]]:
  """Parse openslide level from base request."""
  try:
    level_dict = json_validation_utils.validate_str_key_dict(
        base_request.get(_InstanceJsonKeys.OPENSLIDE_PYRAMID_LEVEL, {})
    )
  except json_validation_utils.ValidationError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid JSON formatted openslide pyramid level.'
    ) from exp
  level = level_dict.get(_InstanceJsonKeys.OPENSLIDE_LEVEL_INDEX)
  if level is not None:
    try:
      return int(level)
    except ValueError:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Failed to parse OpenSlide level index; value must be an integer.'
      ) from ValueError
  width = level_dict.get(_InstanceJsonKeys.OPENSLIDE_LEVEL_WIDTH_PX)
  height = level_dict.get(_InstanceJsonKeys.OPENSLIDE_LEVEL_HEIGHT_PX)
  if width is not None and height is not None:
    try:
      return image_dimension_utils.ImageDimensions(int(width), int(height))
    except ValueError:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Failed to parse OpenSlide level width and/or height.'
      ) from ValueError
  default_ps = level_dict.get(
      _InstanceJsonKeys.OPENSLIDE_LEVEL_PIXEL_SPACING_MMP
  )
  width_mmp = level_dict.get(
      _InstanceJsonKeys.OPENSLIDE_LEVEL_WIDTH_PIXEL_SPACING_MMP, default_ps
  )
  height_mmp = level_dict.get(
      _InstanceJsonKeys.OPENSLIDE_LEVEL_HEIGHT_PIXEL_SPACING_MMP, default_ps
  )
  if width_mmp is not None and height_mmp is not None:
    try:
      return PixelSpacing(float(width_mmp), float(height_mmp))
    except ValueError:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Failed to parse OpenSlide level width and/or height pixel spacing.'
      ) from ValueError
  return None


class OpenSlideHandler(abstract_handler.AbstractHandler):
  """Reads a traditional image from local file system."""

  def __init__(self, endpoint_input_dim: EndpointInputDimensions):
    self._endpoint_input_dim = endpoint_input_dim

  def process_files(
      self,
      instance_patch_coordinates: Sequence[
          patch_coordinate_module.PatchCoordinate
      ],
      base_request: Mapping[str, Any],
      file_paths: abstract_handler.InputFileIterator,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    instance_extensions = abstract_handler.get_base_request_extensions(
        base_request
    )
    resize_image_dimensions = image_dimension_utils.get_resize_image_dimensions(
        instance_extensions
    )
    for file_path in file_paths:
      try:
        with contextlib.ExitStack() as stack:
          if isinstance(file_path, io.BytesIO):
            image_bytes = file_path.read()
            tdir = stack.enter_context(tempfile.TemporaryDirectory())
            file_path = os.path.join(tdir, 'temp')
            with open(file_path, 'wb') as f:
              f.write(image_bytes)
          with openslide.OpenSlide(file_path) as slide:
            if slide.level_count < 1:
              raise data_accessor_errors.DataAccessorError(
                  'Openslide image has no pyramid levels. Please provide a'
                  ' valid slide.'
              )
            openslide_level = _parse_openslide_level(base_request)
            if isinstance(openslide_level, int):
              slide_level = _get_open_slide_level_from_int(
                  openslide_level, slide
              )
            elif isinstance(
                openslide_level, image_dimension_utils.ImageDimensions
            ):
              slide_level = _get_open_slide_level_from_dimensions(
                  openslide_level, slide
              )
            elif isinstance(openslide_level, PixelSpacing):
              slide_level = _get_open_slide_level_from_pixel_spacing(
                  openslide_level, slide
              )
            else:
              slide_level = _get_default_openslide_level(
                  slide,
                  instance_patch_coordinates,
                  self._endpoint_input_dim,
              )
            target_icc_profile = icc_profile_utils.get_target_icc_profile(
                instance_extensions
            )
            patch_required_to_be_fully_in_source_image = patch_coordinate_module.patch_required_to_be_fully_in_source_image(
                instance_extensions
            )
            yield abstract_data_accessor.DataAcquisition(
                abstract_data_accessor.AccessorDataSource.OPEN_SLIDE_IMAGE_PYRAMID_LEVEL,
                _decode_open_slide_image(
                    slide,
                    slide_level,
                    target_icc_profile,
                    instance_patch_coordinates,
                    resize_image_dimensions,
                    patch_required_to_be_fully_in_source_image,
                ),
            )
            # mark file as being processed so custom iterator will now return
            # next file in sequence.
            file_paths.processed_file()
      except openslide.OpenSlideError:
        # The handler is purposefully eating the message here.
        # if a handler fails to process the image it returns an empty iterator.

        return
