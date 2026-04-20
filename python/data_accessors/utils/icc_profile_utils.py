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
"""ICC profile utils for data accessors."""
import io
from typing import Any, Mapping, Optional, Union

from ez_wsi_dicomweb import dicom_slide
import numpy as np
from PIL import ImageCms
import PIL.Image
import pydicom

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from serving.logging_lib import cloud_logging_client

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


def get_icc_profile_bytes_from_pil_image(pil_image: PIL.Image.Image) -> bytes:
  return pil_image.info.get('icc_profile', b'')


def get_icc_profile_bytes_from_compressed_image(
    compressed_image_bytes: bytes,
) -> bytes:
  try:
    with io.BytesIO(compressed_image_bytes) as frame_bytes:
      with PIL.Image.open(frame_bytes) as p_image:
        return get_icc_profile_bytes_from_pil_image(p_image)
  except PIL.UnidentifiedImageError:
    return b''


def get_dicom_icc_profile_bytes(dcm: pydicom.Dataset) -> bytes:
  if 'OpticalPathSequence' in dcm:
    for dataset in dcm.OpticalPathSequence:
      if 'ICCProfile' in dataset:
        return dataset.ICCProfile
  if 'ICCProfile' in dcm:
    return dcm.ICCProfile
  return b''


def get_transform_imaging_to_icc_profile_name(
    extensions: Mapping[str, Any],
) -> str:
  """Returns optional state for EZ-WSI DICOM web.

  Args:
    extensions: A string key dictionary of JSON formatted metadata.

  Returns:
    The ICC profile name.
  """
  state = extensions.get(
      _InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE,
      'NONE',
  )
  if not isinstance(state, str):
    msg = (
        f'{_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value'
        ' is not a string.'
    )
    cloud_logging_client.info(msg)
    raise data_accessor_errors.InvalidRequestFieldError(msg)
  return state.upper()


def get_target_icc_profile(
    extensions: Mapping[str, Any],
) -> Optional[ImageCms.core.CmsProfile]:
  """Returns optional state for EZ-WSI DICOM web.

  Args:
    extensions: A string key dictionary of JSON formatted metadata.

  Returns:
    The ICC profile or None if no ICC profile is defined.
  """
  state = get_transform_imaging_to_icc_profile_name(extensions)
  if state == 'NONE':
    return None
  if state == 'ADOBERGB':
    return dicom_slide.get_adobergb_icc_profile()
  if state == 'ROMMRGB':
    return dicom_slide.get_rommrgb_icc_profile()
  if state == 'SRGB':
    return dicom_slide.get_srgb_icc_profile()
  if state == 'DISPLAYP3':
    return dicom_slide.get_displayp3_icc_profile()
  msg = (
      f'{_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value'
      ' is not valid; expecting: ADOBERGB, ROMMRGB, SRGB, or NONE.'
  )
  cloud_logging_client.info(msg)
  raise data_accessor_errors.InvalidIccProfileTransformError(msg)


def create_icc_profile_transformation(
    dicom_slide_icc_profile: bytes,
    target_icc_profile: Union[bytes, ImageCms.core.CmsProfile],
    rendering_intent: ImageCms.Intent = ImageCms.Intent.PERCEPTUAL,
) -> Optional[ImageCms.ImageCmsTransform]:
  """Returns ICC profile transformation."""
  return dicom_slide.create_icc_profile_transformation(
      dicom_slide_icc_profile, target_icc_profile, rendering_intent
  )


def transform_image_bytes_to_target_icc_profile(
    image_bytes: np.ndarray,
    icc_profile_transformation: Optional[ImageCms.ImageCmsTransform],
) -> np.ndarray:
  """Returns image bytes transformed to target ICC profile."""
  if image_bytes.ndim != 3 or image_bytes.shape[2] != 3:
    return image_bytes
  return dicom_slide.transform_image_bytes_color(
      image_bytes, icc_profile_transformation
  )
