#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for image processing."""

import io
from typing import Any

import numpy as np
import png


_BITS_PER_BYTE = 8


def encode_png(array: np.ndarray) -> bytes:
  """Converts an unsigned integer 2-D NumPy array to a PNG-encoded string.

  Unsigned 8-bit and 16-bit images are supported.

  Args:
    array: Array to be encoded.

  Returns:
    PNG-encoded string.

  Raises:
    ValueError: If any of the following occurs:
      - `array` is not 2-D.
      - `array` data type is unsupported.
  """
  supported_types = frozenset([np.uint8, np.uint16])
  # Sanity checks.
  if array.ndim != 2:
    raise ValueError(f'Array must be 2-D. Actual dimensions: {array.ndim}')
  if array.dtype.type not in supported_types:
    raise ValueError(
        'Pixels must be either `uint8` or `uint16`. '
        f'Actual type: {array.dtype.name!r}'
    )

  # Convert to PNG.
  writer = png.Writer(
      width=array.shape[1],
      height=array.shape[0],
      greyscale=True,
      bitdepth=_BITS_PER_BYTE * array.dtype.itemsize,
  )
  with io.BytesIO() as output_data:
    writer.write(output_data, array.tolist())
    return output_data.getvalue()


def rescale_dynamic_range(image: np.ndarray) -> np.ndarray:
  """Rescales the dynamic range in an integer image to use the full bit range.

  Args:
    image: An image containing unsigned integer pixels.

  Returns:
    Rescaled copy of `image` that uses all the available bits per pixel.

  Raises:
    ValueError: If pixels are not of an unsigned integer type.
  """
  if not np.issubdtype(image.dtype, np.unsignedinteger):
    raise ValueError(
        'Image pixels must be an unsigned integer. '
        f'Actual type: {image.dtype.name!r}'
    )
  iinfo = np.iinfo(image.dtype)
  return np.interp(
      image, (image.min(), image.max()), (iinfo.min, iinfo.max)
  ).astype(iinfo)


def shift_to_unsigned(image: np.ndarray) -> np.ndarray:
  """Shifts values by the minimum value to an unsigned array suitable for PNG.

  This works with signed images and converts them to unsigned versions. It
  involves an inefficient step to convert to a larger data structure for
  shifting all values by the minimum value in the array. It also support float
  data by converting them into uint16.

  Args:
    image: An image containing signed integer pixels.

  Returns:
    Copy of `image` in an unsigned format. Note that the exact same image is
      returned when given an unsigned version.

  Raises:
    ValueError: If pixels are not of an integer type or float.
  """
  if image.dtype == np.uint16 or image.dtype == np.uint8:
    return image
  elif image.dtype == np.int16:
    image = image.astype(np.int32)
    return (image - np.min(image)).astype(np.uint16)
  elif image.dtype == np.int8:
    image = image.astype(np.int16)
    return (image - np.min(image)).astype(np.uint8)
  elif image.dtype.kind == 'f':
    uint16_max = np.iinfo(np.uint16).max
    image = image - np.min(image)
    max_image = np.max(image)
    if max_image > uint16_max:
      image *= (uint16_max / max_image)
      np.clip(image, None, uint16_max, out=image)
    np.round(image, 0, out=image)
    return image.astype(np.uint16)
  raise ValueError(
      'Image pixels must be an 8, 16 bit integer or float type. '
      f'Actual type: {image.dtype.name!r}'
  )


def window(
    image: np.ndarray,
    window_center: int,
    window_width: int,
    dtype: Any,
) -> np.ndarray:
  """Applies the Window operation on an integer image."""
  # Windowing implemented here contains two issues. Leaving unchanged to
  # avoid regression in legacy CXR embeddings.
  # 1) Actual window range should be center +- half width.
  # See https://radiopaedia.org/articles/windowing-ct?lang=us
  # 2) Post interpolation, should round not floor to nearest integer.
  # See implementation in:
  # data_accessors/local_file_handlers/generic_dicom_handler.py
  iinfo = np.iinfo(dtype)
  half_window_width = window_width // 2
  top_clip = window_center - 1 + half_window_width
  bottom_clip = window_center - half_window_width
  return np.interp(
      image.clip(bottom_clip, top_clip),
      (bottom_clip, top_clip),
      (0, iinfo.max),
  ).astype(iinfo)
