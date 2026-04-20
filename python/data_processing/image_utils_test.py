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

import io
import os
from typing import Any, Dict, List

from absl import logging
import immutabledict
import numpy as np
from PIL import Image

from absl.testing import absltest
from absl.testing import parameterized
from data_processing import image_utils


_TEST_DATA = (
    'python/data_processing/testdata'
)
PIXEL_ERROR_MARGIN = 0.0005

# Test image suffixes (16-bit).
_IMAGE_SUFFIXES = ('1336', '6130', '8101', '8708')


def _decode_png(png_bytes: bytes) -> np.ndarray:
  """Converts an encoded 16-bit grayscale PNG to a 2D uint16 array."""
  # The use of np.uint8 here is for the png bytes, not the pixel values.
  byte_array = np.frombuffer(png_bytes, np.uint8)
  try:
    pixel_array = np.array(Image.open(io.BytesIO(byte_array))).astype(np.uint16)
  except IOError as exc:
    raise ValueError('Could not decode a png image from input.') from exc
  if pixel_array is None:
    raise ValueError('Could not decode a png image from input.')
  assert pixel_array.dtype == np.uint16, 'dtype is %s' % str(pixel_array.dtype)
  assert pixel_array.ndim == 2
  return pixel_array


def _get_pixel_stats(encoded_string: bytes) -> Dict[str, Any]:
  """Returns, min, max and average pixel value istrhe encoded_string."""
  decoded_array = Image.open(io.BytesIO(encoded_string))
  npdecoded_array = np.array(decoded_array)

  stats = {
      'min': np.min(npdecoded_array),
      'max': np.max(npdecoded_array),
      'ave': np.average(npdecoded_array),
  }
  return stats


class TestWindow(parameterized.TestCase):
  """Unit tests for `window()`."""

  @parameterized.named_parameters(
      ('0', 0, 0),  # Min.
      ('2047', 2047, 32759),  # floor(center).
      ('2048', 2048, 32775),  # ceil(center).
      ('4095', 4095, 65535),  # Max.
  )
  def testWithinWindow(self, input_value: int, expected_value: int):
    """Tests 12-bit window [0, 2 ^ 12 - 1] for inputs within range."""
    window_center = 2048
    window_width = 4096
    actual_array = image_utils.window(
        np.array([input_value], dtype=np.uint16),
        window_center,
        window_width,
        np.uint16,
    )
    self.assertEqual(actual_array[0], expected_value)

  @parameterized.named_parameters(
      ('2045', 2045, 0),  # Before lowest.
      ('2046', 2046, 0),  # At lowest.
      ('2047', 2047, 21845),  # After lowest.
      ('2048', 2048, 43690),  # Before highest.
      ('2049', 2049, 65535),  # At highest.
      ('2050', 2050, 65535),  # After highest.
  )
  def testClipping(self, input_value: int, expected_value: int):
    """Tests clipping for values near the edges of window [2046, 2049]."""
    window_center = 2048
    window_width = 4
    actual_array = image_utils.window(
        np.array([input_value], dtype=np.uint16),
        window_center,
        window_width,
        np.uint16,
    )
    self.assertEqual(actual_array[0], expected_value)


class TestShiftValues(parameterized.TestCase):
  """Unit tests for `shift_to_unsigned()`."""

  @parameterized.named_parameters(
      ('Unsigned 16', [3, 4, 5000], [3, 4, 5000], np.uint16, np.uint16),
      ('Unsigned 8', [0, 6, 7, 255], [0, 6, 7, 255], np.uint8, np.uint8),
      ('Signed 16', [3, 4, 5000], [0, 1, 4997], np.int16, np.uint16),
      ('Signed 8', [-128, 6, 7, 127], [0, 134, 135, 255], np.int8, np.uint8),
  )
  def testSuccess(
      self,
      input_list: List[int],
      expected_list: List[int],
      dtype: Any,
      expected_dtype: Any,
  ):
    """Tests shift behavior."""
    actual = image_utils.shift_to_unsigned(np.array(input_list, dtype=dtype))
    self.assertEqual(actual.tolist(), expected_list)
    self.assertEqual(actual.dtype, expected_dtype)

  @parameterized.parameters(
      bool,
      np.int32,
  )
  def testInvalidType(self, dtype):
    """Tests failure if input array is not an unsigned integer type."""
    with self.assertRaisesRegex(ValueError, 'Image pixels must be'):
      image_utils.shift_to_unsigned(np.array([0], dtype=dtype))


class TestRescaleDynamicRange(parameterized.TestCase):
  """Unit tests for `rescale_dynamic_range()`."""

  @parameterized.named_parameters(
      ('Flip', [3, 4, 5], [0, 32767, 65535]),
      ('NoFlip', [5, 6, 7, 10], [0, 13107, 26214, 65535]),
  )
  def testSuccess(self, input_list: List[int], expected_list: List[int]):
    """Locks down expected linear interpolation behavior."""
    actual_list = image_utils.rescale_dynamic_range(
        np.array(input_list, dtype=np.uint16)
    ).tolist()
    self.assertEqual(actual_list, expected_list)

  @parameterized.parameters(
      np.float32,
      np.float64,
  )
  def testInvalidType(self, dtype):
    """Tests failure if input array is not an integer type."""
    with self.assertRaisesRegex(ValueError, 'Image pixels must be'):
      image_utils.rescale_dynamic_range(np.array([0], dtype=dtype))


class TestEncodePNG(parameterized.TestCase):
  """Unit tests for `encode_png()`."""

  def setUp(self):
    super().setUp()
    numpy_array_dict = {}
    png_bytes_dict = {}
    for suffix in _IMAGE_SUFFIXES:
      numpy_array_file_name = 'wado_mock_wl16_instance%s.npy' % suffix
      logging.info('Loading: %s', numpy_array_file_name)
      numpy_array_file = open(
          os.path.join(_TEST_DATA, numpy_array_file_name), 'rb'
      )
      numpy_array_dict[suffix] = np.load(numpy_array_file)
      numpy_array_file.close()

      png_file_name = 'wado_mock_png_instance%s' % suffix
      logging.info('Loading: %s', png_file_name)
      with open(
          os.path.join(_TEST_DATA, png_file_name), 'rb'
      ) as f:
        png_bytes_dict[suffix] = f.read()

    self._numpy_array_dict = immutabledict.immutabledict(numpy_array_dict)
    self._png_bytes_dict = immutabledict.immutabledict(png_bytes_dict)

  @parameterized.parameters((np.uint8,), (np.uint16,))
  def testSuccess_Range(self, dtype):
    """Tests image (w, h) = (4, 2) with maximum range of values for uint*."""
    test_array = np.array(
        [[0, 1, 2, 3], [np.iinfo(dtype).max, 12000, 100, 150]]
    ).astype(dtype)
    png_text = image_utils.encode_png(test_array)
    result_array = np.array(Image.open(io.BytesIO(png_text)))
    np.testing.assert_array_equal(test_array, result_array)

  @parameterized.parameters((suffix,) for suffix in _IMAGE_SUFFIXES)
  def testSuccess_Idempotence(self, suffix):
    """Tests that `decode(encode(*))` is an identity op."""
    canonical_pixels = _decode_png(self._png_bytes_dict[suffix])
    encoded_png_bytes = image_utils.encode_png(canonical_pixels)
    actual_pixels = _decode_png(encoded_png_bytes)
    self.assertEqual(canonical_pixels.dtype, actual_pixels.dtype)
    self.assertEqual(canonical_pixels.shape, actual_pixels.shape)
    self.assertTrue(np.array_equal(canonical_pixels, actual_pixels))

  @parameterized.parameters((suffix,) for suffix in _IMAGE_SUFFIXES)
  def testSuccess_Regression(self, suffix):
    """Captures difference in outputs from OpenCV-based encoder."""
    test_png_bytes = image_utils.encode_png(self._numpy_array_dict[suffix])
    canonical_png_bytes = self._png_bytes_dict[suffix]

    canonical_pixel_stats = _get_pixel_stats(canonical_png_bytes)
    test_pixel_stats = _get_pixel_stats(test_png_bytes)

    logging.info('Suffix: %s', suffix)
    logging.info('Canonical pixels: %s', str(canonical_pixel_stats))
    logging.info('Instance pixels: %s', str(test_pixel_stats))
    logging.info(
        'Diff in average pixel values: %f',
        test_pixel_stats['ave'] - canonical_pixel_stats['ave'],
    )

    self.assertEqual(test_pixel_stats['min'], canonical_pixel_stats['min'])
    self.assertEqual(test_pixel_stats['max'], canonical_pixel_stats['max'])
    self.assertAlmostEqual(
        test_pixel_stats['ave'],
        canonical_pixel_stats['ave'],
        delta=PIXEL_ERROR_MARGIN,
    )

  @parameterized.parameters(np.int32, np.uint32, np.int16, np.int8)
  def testFailure_Dtype(self, dtype):
    """Tests failure to convert to PNG for invalid image dimensions."""
    array = np.array([[0, 1], [2, 4]], dtype=dtype)
    with self.assertRaisesRegex(
        ValueError, 'Pixels must be either `uint8` or `uint16`.'
    ):
      image_utils.encode_png(array)

  def testFailure_Dimensions(self):
    """Tests failure to convert with wrong input dimensions."""
    test_array_3d = np.ones([2, 2, 2]).astype(np.uint16)
    with self.assertRaisesRegex(ValueError, 'Array must be 2-D.'):
      image_utils.encode_png(test_array_3d)


if __name__ == '__main__':
  absltest.main()
