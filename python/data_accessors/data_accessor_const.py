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
"""Data accessor constants."""


class InstanceJsonKeys:
  """Instance JSON Keys."""

  IMAGE = 'image'
  IMAGE_URL = 'image_url'
  INPUT_BYTES = 'input_bytes'
  GCS_SOURCE = 'gcs_source'  # generic gcs
  DICOM_SOURCE = 'dicom_source'  # generic dicom
  GCS_URI = 'gcs_uri'  # legacy single GCS_SOURCE URI
  DICOM_WEB_URI = 'dicomweb_uri'  # legacy single DICOM_SOURCE URI
  TEXT = 'text'  # generic text
  BEARER_TOKEN = 'access_credential'  # all imputs
  EXTENSIONS = 'extensions'  # generic dicom
  INSTANCES = 'instances'
  URL = 'url'

  PATCH_COORDINATES = 'patch_coordinates_list'
  X_ORIGIN = 'x_origin'  # wsi dicom
  Y_ORIGIN = 'y_origin'  # wsi dicom
  WIDTH = 'width'  # wsi dicom
  HEIGHT = 'height'  # wsi dicom

  IMAGE_DIMENSIONS = 'image_dimensions'  # wsi dicom

  TRANSFORM_IMAGING_TO_ICC_PROFILE = (  # wsi dicom
      'transform_imaging_to_icc_profile'
  )
  REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE = (  # wsi dicom
      'require_patches_fully_in_source_image'
  )
  LOAD_WHOLE_SLIDE_FRAME_RATIO = 'load_whole_slide_frame_ratio'  # wsi dicom
  EZ_WSI_STATE = 'ez_wsi_state'  # wsi dicom

  OPENSLIDE_PYRAMID_LEVEL = 'openslide_pyramid_level'
  OPENSLIDE_LEVEL_INDEX = 'index'
  OPENSLIDE_LEVEL_WIDTH_PX = 'width'
  OPENSLIDE_LEVEL_HEIGHT_PX = 'height'
  OPENSLIDE_LEVEL_PIXEL_SPACING_MMP = 'pixel_spacing_mmp'
  OPENSLIDE_LEVEL_WIDTH_PIXEL_SPACING_MMP = 'width_pixel_spacing_mmp'
  OPENSLIDE_LEVEL_HEIGHT_PIXEL_SPACING_MMP = 'height_pixel_spacing_mmp'
