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
"""local handler for handling traditional image files."""

from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import PIL
import PIL.Image

from data_accessors import abstract_data_accessor
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module


def _generate_images(
    img: np.ndarray,
    instance_patch_coordinates: Sequence[
        patch_coordinate_module.PatchCoordinate
    ],
    instance_extensions: Mapping[str, Any],
) -> Iterator[np.ndarray]:
  """Generates images from a numpy array image."""
  if img.ndim == 2:
    img = np.expand_dims(img, axis=2)
  resize_image_dimensions = image_dimension_utils.get_resize_image_dimensions(
      instance_extensions
  )
  if resize_image_dimensions is not None:
    img = image_dimension_utils.resize_image_dimensions(
        img, resize_image_dimensions
    )
  if not instance_patch_coordinates:
    yield img
  else:
    patch_required_to_be_fully_in_source_image = (
        patch_coordinate_module.patch_required_to_be_fully_in_source_image(
            instance_extensions
        )
    )
    image_shape = image_dimension_utils.ImageDimensions(
        width=img.shape[1],
        height=img.shape[0],
    )
    for pc in instance_patch_coordinates:
      if patch_required_to_be_fully_in_source_image:
        pc.validate_patch_in_dim(image_shape)
      yield patch_coordinate_module.get_patch_from_memory(pc, img)


class TraditionalImageHandler(abstract_handler.AbstractHandler):
  """Reads a traditional image from local file system."""

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
    target_icc_profile = icc_profile_utils.get_target_icc_profile(
        instance_extensions
    )
    for file_path in file_paths:
      try:
        with PIL.Image.open(file_path) as image:
          img = np.asarray(image)
          if (
              target_icc_profile is not None
              and img.ndim == 3
              and img.shape[2] == 3
          ):
            icc_profile_bytes = (
                icc_profile_utils.get_icc_profile_bytes_from_pil_image(image)
            )
            if icc_profile_bytes:
              transform = icc_profile_utils.create_icc_profile_transformation(
                  icc_profile_bytes, target_icc_profile
              )
              if transform is not None:
                img = icc_profile_utils.transform_image_bytes_to_target_icc_profile(
                    img, transform
                )
      except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError):
        # The handler is purposefully eating the message here.
        # if a handler fails to process the image it returns an empty iterator.
        return
      yield abstract_data_accessor.DataAcquisition(
          abstract_data_accessor.AccessorDataSource.TRADITIONAL_IMAGES,
          _generate_images(
              img, instance_patch_coordinates, instance_extensions
          ),
      )
      # mark file as being processed so custom iterator will now return next
      # file in sequence.
      file_paths.processed_file()
