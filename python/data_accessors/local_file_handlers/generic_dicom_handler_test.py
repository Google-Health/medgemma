# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License");
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
"""Unit tests for generic dicom handler."""

import io
import os
from typing import Any, Mapping, Optional

from absl.testing import absltest
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import dicom_slide
import numpy as np
import PIL.Image
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import patch_coordinate
from data_accessors.utils import test_utils


_generic_dicom_handler = generic_dicom_handler.GenericDicomHandler()
_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


def _mock_instance_extension_metadata(
    extensions: Mapping[str, Any],
) -> Mapping[str, Any]:
  return {_InstanceJsonKeys.EXTENSIONS: extensions}


_MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM = (
    _mock_instance_extension_metadata(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ADOBERGB'}
    )
)


def _encapsulated_dicom_path() -> str:
  return test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')


def _mock_ct_dicom(
    path: str,
    photometric_interpretation: str,
    window_center: Optional[int] = None,
    window_width: Optional[int] = None,
) -> str:
  with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
    if 'WindowCenter' in dcm:
      del dcm['WindowCenter']
    if 'WindowWidth' in dcm:
      del dcm['WindowWidth']
    if window_center is not None and window_width is not None:
      dcm.WindowCenter = window_center
      dcm.WindowWidth = window_width
    dcm.Modality = 'CT'
    dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2.1'
    dcm.Rows = 3
    dcm.Columns = 3
    dcm.PhotometricInterpretation = photometric_interpretation
    dcm.SamplesPerPixel = 1
    dcm.BitsAllocated = 16
    dcm.BitsStored = 12
    dcm.HighBit = 11
    dcm.PixelRepresentation = 1  # twos complement
    image_pixels = np.zeros((3, 3), dtype=np.int16)
    for i in range(9):
      image_pixels[int(i // 3), int(i % 3)] = (4000 * i / 8) - 2000
    dcm.PixelData = image_pixels.tobytes()
    dcm.save_as(path)
    return path


class GenericDicomHandlerTest(parameterized.TestCase):

  def test_load_encapsulated_dicom_file_path(self):
    images = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [],
            {},
            abstract_handler.InputFileIterator([_encapsulated_dicom_path()]),
        )
    )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (1024, 1024, 1))

  def test_does_not_process_wsi_dicom(self):
    self.assertEmpty(
        test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [],
                {},
                abstract_handler.InputFileIterator([
                    test_utils.testdata_path(
                        'wsi', 'multiframe_camelyon_challenge_image.dcm'
                    )
                ]),
            )
        )
    )

  def test_load_encapsulated_dicom_from_bytes_io(self):
    with open(_encapsulated_dicom_path(), 'rb') as f:
      with io.BytesIO(f.read()) as binary_file:
        images = test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([binary_file])
            )
        )
        self.assertLen(images, 1)
        self.assertEqual(images[0].shape, (1024, 1024, 1))

  def test_loadpatches_coordinates(self):
    images = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [
                patch_coordinate.PatchCoordinate(0, 0, 10, 10),
                patch_coordinate.PatchCoordinate(10, 10, 10, 10),
            ],
            {},
            abstract_handler.InputFileIterator([_encapsulated_dicom_path()]),
        )
    )
    self.assertLen(images, 2)
    for img in images:
      self.assertEqual(img.shape, (10, 10, 1))

  def test_load_dicom_file_missing_photometric_interpretation_raises_error_path(
      self,
  ):
    pixeldata = np.zeros((10, 10, 3), dtype=np.uint8)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
    )
    dcm.Modality = 'CR'
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    dcm.save_as(dcm_path)
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*PhotometricInterpretation is required for DICOM.*',
    ):
      test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_load_dicom_file_unsupported_samples_per_pixel_raises_error(
      self,
  ):
    pixeldata = np.zeros((10, 10, 2), dtype=np.uint8)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
    )
    dcm.Modality = 'DX'
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    dcm.save_as(dcm_path)
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*unsupported number of samples per pixel.*',
    ):
      test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  @parameterized.parameters(
      ['BitsStored', 'PlanarConfiguration', 'PixelRepresentation']
  )
  def test_load_dicom_fails_if_missing_required_tag_raises_error_path(
      self, required_elements
  ):
    pixeldata = np.zeros((10, 10, 3), dtype=np.uint8)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
    )
    dcm.Modality = 'GM'
    dcm.PhotometricInterpretation = 'RGB'
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    del dcm[required_elements]
    dcm.save_as(dcm_path)
    with self.assertRaises(data_accessor_errors.DicomError):
      test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='MONOCHROME1',
          photometric_interpretation='MONOCHROME1',
          channels=1,
      ),
      dict(
          testcase_name='MONOCHROME2',
          photometric_interpretation='MONOCHROME2',
          channels=1,
      ),
      dict(testcase_name='RGB', photometric_interpretation='RGB', channels=3),
  )
  def test_load_dicom_path(self, photometric_interpretation, channels):
    pixeldata = np.zeros((10, 10, channels), dtype=np.uint8)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixeldata
    )
    dcm.Modality = 'XC'
    dcm.PhotometricInterpretation = photometric_interpretation
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    dcm.save_as(dcm_path)
    img = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator([dcm_path])
        )
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (10, 10, channels))

  def test_load_dicom_does_not_transform_if_missing_embedded_icc_profile(self):
    with PIL.Image.open(test_utils.testdata_path('image.jpeg')) as source_img:
      pixel_data = np.asarray(source_img)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixel_data
    )
    dcm.Modality = 'SM'
    dcm.PhotometricInterpretation = 'RGB'
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    dcm.save_as(dcm_path)
    img = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(img[0], pixel_data)

  def test_load_dicom_does_not_icc_transform_if_monochrome(self):
    with PIL.Image.open(
        test_utils.testdata_path('image_bw.jpeg')
    ) as source_img:
      pixel_data = np.asarray(source_img)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixel_data
    )
    dcm.Modality = 'CR'
    dcm.PhotometricInterpretation = 'MONOCHROME2'
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    dcm.save_as(dcm_path)
    img = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(img[0], np.expand_dims(pixel_data, axis=2))

  def test_load_dicom_transform_icc_if_profile_embedded_in_dicom(self):
    with PIL.Image.open(test_utils.testdata_path('image.jpeg')) as source_img:
      pixel_data = np.asarray(source_img)
    dcm = test_utils.create_test_dicom_instance(
        '1.2.840.10008.5.1.4.1.1.1.1', '1.1', '1.1.1', '1.1.1.1', pixel_data
    )
    dcm.Modality = 'DX'
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    dcm.PhotometricInterpretation = 'RGB'
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    dcm.save_as(dcm_path)
    img = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [],
            _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
            abstract_handler.InputFileIterator([dcm_path]),
        )
    )
    self.assertLen(img, 1)
    self.assertFalse(np.array_equal(img[0], pixel_data))

  def test_load_dicom_transform_icc_if_profile_embedded_in_image(self):
    with io.BytesIO() as img_data:
      with PIL.Image.open(test_utils.testdata_path('image.jpeg')) as source_img:
        pixel_data = np.asarray(source_img)
        source_img.save(
            img_data,
            icc_profile=dicom_slide.get_rommrgb_icc_profile_bytes(),
            format='JPEG',
        )
        width, height = source_img.width, source_img.height
      dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
      with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
        dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
        dcm.Columns = width
        dcm.Rows = height
        dcm.PhotometricInterpretation = 'RGB'
        dcm.SamplesPerPixel = 3
        dcm.save_as(dcm_path)
      img = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              abstract_handler.InputFileIterator([dcm_path]),
          )
      )
    self.assertLen(img, 1)
    self.assertFalse(np.array_equal(img[0], pixel_data))

  def test_load_dicom_does_not_transform_icc_if_profile_not_embedded(self):
    with io.BytesIO() as img_data:
      with PIL.Image.open(test_utils.testdata_path('image.jpeg')) as source_img:
        pixel_data = np.asarray(source_img)
        source_img.save(img_data, format='JPEG')
        width, height = source_img.width, source_img.height
      dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
      with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
        dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
        dcm.Columns = width
        dcm.Rows = height
        dcm.PhotometricInterpretation = 'RGB'
        dcm.SamplesPerPixel = 3
        dcm.save_as(dcm_path)
      img = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [],
              _MOCK_INSTANCE_METADATA_REQUEST_ICCPROFILE_NORM,
              abstract_handler.InputFileIterator([dcm_path]),
          )
      )
    self.assertLen(img, 1)
    self.assertFalse(np.array_equal(img[0], pixel_data))

  @parameterized.named_parameters(
      dict(
          testcase_name='downsample_2x',
          scale_factor=1 / 2,
          interpolation=cv2.INTER_AREA,
      ),
      dict(
          testcase_name='upsample_2x',
          scale_factor=2,
          interpolation=cv2.INTER_CUBIC,
      ),
  )
  def test_load_whole_dicom_resize(self, scale_factor, interpolation):
    with io.BytesIO() as img_data:
      with open(
          test_utils.testdata_path('image.jpeg'),
          'rb',
      ) as source_img:
        img_data.write(source_img.read())
      img_data.seek(0)
      with PIL.Image.open(img_data) as source_img:
        pixel_data = np.asarray(source_img)
        width, height = source_img.width, source_img.height
      img_data.seek(0)
      dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
      with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
        dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
        dcm.Columns = width
        dcm.Rows = height
        dcm.PhotometricInterpretation = 'RGB'
        dcm.SamplesPerPixel = 3
        dcm.save_as(dcm_path)
      img = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [],
              _mock_instance_extension_metadata({
                  _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                      'width': int(width * scale_factor),
                      'height': int(height * scale_factor),
                  }
              }),
              abstract_handler.InputFileIterator([dcm_path]),
          )
      )
    self.assertLen(img, 1)
    np.testing.assert_array_equal(
        img[0],
        cv2.resize(
            pixel_data,
            (int(width * scale_factor), int(height * scale_factor)),
            interpolation=interpolation,
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='downsample_2x',
          scale_factor=1 / 2,
          interpolation=cv2.INTER_AREA,
      ),
      dict(
          testcase_name='upsample_2x',
          scale_factor=2,
          interpolation=cv2.INTER_CUBIC,
      ),
  )
  def test_load_whole_dicom_resize_patchs(self, scale_factor, interpolation):
    patch_coordinates = [
        patch_coordinate.PatchCoordinate(0, 0, 10, 10),
        patch_coordinate.PatchCoordinate(10, 10, 10, 10),
    ]
    with io.BytesIO() as img_data:
      with open(
          test_utils.testdata_path('image.jpeg'),
          'rb',
      ) as source_img:
        img_data.write(source_img.read())
      img_data.seek(0)
      with PIL.Image.open(img_data) as source_img:
        pixel_data = np.asarray(source_img)
        width, height = source_img.width, source_img.height
      img_data.seek(0)
      dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
      with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
        dcm.PixelData = pydicom.encaps.encapsulate([img_data.getvalue()])
        dcm.Columns = width
        dcm.Rows = height
        dcm.PhotometricInterpretation = 'RGB'
        dcm.SamplesPerPixel = 3
        dcm.save_as(dcm_path)
      images = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              patch_coordinates,
              _mock_instance_extension_metadata({
                  _InstanceJsonKeys.IMAGE_DIMENSIONS: {
                      'width': int(width * scale_factor),
                      'height': int(height * scale_factor),
                  }
              }),
              abstract_handler.InputFileIterator([dcm_path]),
          )
      )
    expected_img = cv2.resize(
        pixel_data,
        (int(width * scale_factor), int(height * scale_factor)),
        interpolation=interpolation,
    )
    self.assertLen(images, 2)
    for pc, img in zip(patch_coordinates, images):
      np.testing.assert_array_equal(
          img,
          expected_img[
              pc.y_origin : pc.y_origin + pc.height,
              pc.x_origin : pc.x_origin + pc.width,
              ...,
          ],
      )

  def test_validate_transfer_syntax(self):
    dcm_path = os.path.join(self.create_tempdir(), 'test.dcm')
    with pydicom.dcmread(_encapsulated_dicom_path()) as dcm:
      dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.100'
      dcm.save_as(dcm_path)
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*DICOM instance encoded using unsupported transfer syntax.*',
    ):
      test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_path])
          )
      )

  def test_alternative_windowing(self):
    handler = generic_dicom_handler.GenericDicomHandler({
        'CT': generic_dicom_handler.ModalityDefaultImageTransform(
            get_image_transform_op=lambda dcm: generic_dicom_handler.TraditionalWindow(
                dcm.WindowCenter, dcm.WindowWidth
            ),
        )
    })
    dcm_path = _mock_ct_dicom(
        os.path.join(self.create_tempdir(), 'test.dcm'),
        'MONOCHROME2',
        0,
        1000,
    )
    result = test_utils.flatten_data_acquisition(
        handler.process_files(
            [], {}, abstract_handler.InputFileIterator([dcm_path])
        )
    )
    self.assertLen(result, 1)
    np.testing.assert_array_equal(
        np.squeeze(result[0], axis=-1),
        np.asarray(
            [
                [0, 0, 0],
                [0, 128, 255],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    )

  def test_default_ct_windowing_rgb_window(self):
    dcm_path = _mock_ct_dicom(
        os.path.join(self.create_tempdir(), 'test.dcm'),
        'MONOCHROME2',
        0,
        1000,
    )
    result = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator([dcm_path])
        )
    )
    self.assertLen(result, 1)
    np.testing.assert_array_equal(
        result[0],
        np.asarray(
            [
                [[0, 0, 0], [0, 0, 0], [3, 0, 0]],
                [[65, 0, 0], [128, 0, 0], [190, 255, 255]],
                [[252, 255, 255], [255, 255, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        ),
    )

  @parameterized.named_parameters(
      dict(testcase_name='uint16_data', dtype=np.uint16),
      dict(testcase_name='uint8_data', dtype=np.uint8),
  )
  def test_default_window(self, dtype):
    data = np.zeros((10, 10, 1), dtype=np.int16)
    for i in range(10):
      data[i, :] = -1000 + 200 * i
    window = generic_dicom_handler.TraditionalWindow(0, 500, dtype)
    expected = np.clip(data, -250, 250) + 250
    expected = (expected.astype(np.float64) / 500) * np.iinfo(dtype).max
    expected = np.round(expected, 0).astype(dtype)
    data = window.apply(data)
    np.testing.assert_array_equal(data, expected)

  def test_rgb_window(self):
    data = np.zeros((10, 10, 1), dtype=np.int16)
    for i in range(10):
      data[i, :] = -1000 + 200 * i
    window = generic_dicom_handler.RGBWindow(
        generic_dicom_handler.TraditionalWindow(0, 500, np.uint8),
        generic_dicom_handler.TraditionalWindow(175, 80, np.uint8),
        generic_dicom_handler.TraditionalWindow(40, 80, np.uint8),
    )
    red = np.clip(data, -250, 250) + 250
    red = (red.astype(np.float64) / 500) * np.iinfo(np.uint8).max
    red = np.round(red, 0).astype(np.uint8)
    green = np.clip(data, 135, 215) - 135
    green = (green.astype(np.float64) / 80) * np.iinfo(np.uint8).max
    green = np.round(green, 0).astype(np.uint8)
    blue = np.clip(data, 0, 80)
    blue = (blue.astype(np.float64) / 80) * np.iinfo(np.uint8).max
    blue = np.round(blue, 0).astype(np.uint8)
    data = window.apply(data)
    np.testing.assert_array_equal(
        data, np.concatenate([red, green, blue], axis=-1)
    )

  def test_default_window_raises_invalid_dtype(self):
    with self.assertRaisesRegex(
        ValueError, 'Output dtype must be either uint8 or uint16, .*'
    ):
      generic_dicom_handler.TraditionalWindow(4, 4, dtype=np.int32)

  def test_load_mri_dicom(self):
    temp_dir = self.create_tempdir()
    dicom_files_list = []
    for index in range(10):
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
      ) as dcm:
        dcm.Modality = 'MR'
      dcm_path = os.path.join(temp_dir, f'test{index}.dcm')
      dcm.save_as(dcm_path)
      dicom_files_list.append(dcm_path)
    mri_images = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator(dicom_files_list)
        )
    )
    volume = np.stack(mri_images)
    self.assertEqual(volume.dtype, np.uint8)
    self.assertEqual(volume.shape, (10, 512, 512, 1))
    self.assertEqual(np.min(volume), 0)
    self.assertEqual(np.max(volume), 255)

  def test_nop_image_transform(self):
    transform = generic_dicom_handler.NopImageTransform()
    test = np.zeros((10, 10, 1), dtype=np.uint8)
    self.assertIs(transform.apply(test), test)

  @parameterized.named_parameters([
      dict(
          testcase_name='explicit_little_endian',
          transfer_syntax_uid='1.2.840.10008.1.2.1',
      ),
      dict(
          testcase_name='jpeg_baseline',
          transfer_syntax_uid='1.2.840.10008.1.2.4.50',
      ),
  ])
  def test_decode_dicom_images_missing_pixeldata_raises(
      self, transfer_syntax_uid
  ):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        dcm.file_meta.TransferSyntaxUID = transfer_syntax_uid
        del dcm['PixelData']
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*DICOM instance missing PixelData.*',
      ):
        test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_data])
            )
        )

  @parameterized.named_parameters([
      dict(testcase_name='empty_pixel_data', pixel_data=b''),
      dict(testcase_name='none_pixel_data', pixel_data=None),
  ])
  def test_decode_dicom_images_empty_pixeldata_raises(self, pixel_data):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        dcm.file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
        dcm.PixelData = pixel_data
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*DICOM instance missing PixelData.*',
      ):
        test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_data])
            )
        )

  def test_decode_dicom_images_unknown_transfer_syntax_raises(self):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        del dcm.file_meta['TransferSyntaxUID']
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*DICOM missing TransferSyntaxUID.*',
      ):
        test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_data])
            )
        )

  @parameterized.parameters(list(dicom_source_utils._CT_SOP_CLASS_UIDS))
  def test_get_ct_modality_from_sop_class_uid(self, sop_class_uid):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      del dcm['Modality']
      dcm.SOPClassUID = sop_class_uid
      self.assertEqual(generic_dicom_handler._get_dicom_modality(dcm), 'CT')

  @parameterized.parameters(list(dicom_source_utils._MR_SOP_CLASS_UIDS))
  def test_get_mr_modality_from_sop_class_uid(self, sop_class_uid):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      del dcm['Modality']
      dcm.SOPClassUID = sop_class_uid
      self.assertEqual(generic_dicom_handler._get_dicom_modality(dcm), 'MR')

  @parameterized.parameters(list(dicom_source_utils._SM_SOP_CLASS_UIDS))
  def test_get_sm_modality_from_sop_class_uid(self, sop_class_uid):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      del dcm['Modality']
      dcm.SOPClassUID = sop_class_uid
      self.assertEqual(generic_dicom_handler._get_dicom_modality(dcm), 'SM')

  @parameterized.parameters(list(dicom_source_utils._DX_SOP_CLASS_UIDS))
  def test_get_dx_modality_from_sop_class_uid(self, sop_class_uid):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      del dcm['Modality']
      dcm.SOPClassUID = sop_class_uid
      self.assertEqual(generic_dicom_handler._get_dicom_modality(dcm), 'DX')

  def test_get_cr_modality_from_sop_class_uid(self):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      del dcm['Modality']
      dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.1'
      self.assertEqual(generic_dicom_handler._get_dicom_modality(dcm), 'CR')

  @parameterized.parameters(['CR', 'CT', 'MR', 'SM', 'DX'])
  def test_get_modality_from_modality_tag(self, modality):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      dcm.Modality = modality
      self.assertEqual(generic_dicom_handler._get_dicom_modality(dcm), modality)

  def test_get_unknown_modality_raises(self):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as dcm:
      dcm.SOPClassUID = '1.2.3'
      dcm.Modality = ''
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*DICOM missing a modality tag metadata with a defined value.*',
      ):
        generic_dicom_handler._get_dicom_modality(dcm)

  def test_decode_invalid_dicom_file_raises(self):
    with io.BytesIO(b'1342') as dcm_data:
      dicom_handler = generic_dicom_handler.GenericDicomHandler(
          raise_error_if_invalid_dicom=True
      )
      with self.assertRaises(pydicom.errors.InvalidDicomError):
        test_utils.flatten_data_acquisition(
            dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_data])
            )
        )

  def test_decode_invalid_dicom_returns_empty_iterator(self):
    with io.BytesIO(b'1342') as dcm_data:
      dicom_handler = generic_dicom_handler.GenericDicomHandler()
      self.assertEmpty(
          test_utils.flatten_data_acquisition(
              dicom_handler.process_files(
                  [], {}, abstract_handler.InputFileIterator([dcm_data])
              )
          )
      )

  def test_decode_encapsulated_with_invalid_number_of_frames_raises(self):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
      ) as dcm:
        dcm.NumberOfFrames = 0
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*DICOM instance missing PixelData.*',
      ):
        test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_data])
            )
        )

  def test_decode_encapsulated_dicom_missing_number_of_frames_assumes_one_frame(
      self,
  ):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
      ) as dcm:
        del dcm['NumberOfFrames']
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      results = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_data])
          )
      )
      self.assertLen(results, 1)

  def test_cannot_decode_encapsulated_image_raises(self):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
      ) as dcm:
        dcm.PixelData = pydicom.encaps.encapsulate([b'123'])
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          '.*Error decoding encapsulated DICOM PixelData.*',
      ):
        test_utils.flatten_data_acquisition(
            _generic_dicom_handler.process_files(
                [], {}, abstract_handler.InputFileIterator([dcm_data])
            )
        )

  def test_decode_unencapsulated_multi_frame_single_channel_dicom(self):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        dcm.SamplesPerPixel = 1
        dcm.Rows = 10
        dcm.Columns = 10
        dcm.NumberOfFrames = 2
        pixel_data = np.zeros((2, 10, 10), dtype=np.uint16)
        dcm.PixelData = pixel_data.tobytes()
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      results = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_data])
          )
      )
      self.assertEqual([r.shape for r in results], [(10, 10, 3), (10, 10, 3)])

  def test_decode_unencapsulated_multi_frame_multi_channel_dicom(self):
    with io.BytesIO() as dcm_data:
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        dcm.SamplesPerPixel = 3
        dcm.PlanarConfiguration = 0
        dcm.PhotometricInterpretation = 'RGB'
        dcm.Rows = 10
        dcm.Columns = 10
        dcm.NumberOfFrames = 2
        pixel_data = np.zeros((2, 10, 10, 3), dtype=np.uint16)
        dcm.PixelData = pixel_data.tobytes()
        dcm.save_as(dcm_data)
      dcm_data.seek(0)
      results = test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator([dcm_data])
          )
      )
      self.assertEqual([r.shape for r in results], [(10, 10, 3), (10, 10, 3)])

  def test_default_mri_volume_window_op(self):
    self.assertIsInstance(
        generic_dicom_handler._default_mri_volume_window_op(pydicom.Dataset()),
        generic_dicom_handler.NopImageTransform,
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='SOPClassUID',
          keyword='SOPClassUID',
          vr='UI',
          value='1.2.3',
      ),
      dict(
          testcase_name='StudyInstanceUID',
          keyword='StudyInstanceUID',
          vr='UI',
          value='1.2.3',
      ),
      dict(
          testcase_name='SeriesInstanceUID',
          keyword='SeriesInstanceUID',
          vr='UI',
          value='1.2.3',
      ),
      dict(
          testcase_name='ImageType',
          keyword='ImageType',
          vr='CS',
          value=['GOOD'],
      ),
      dict(
          testcase_name='AcquisitionUID',
          keyword='AcquisitionUID',
          vr='UI',
          value='1.2.3',
      ),
      dict(
          testcase_name='AcquisitionNumber',
          keyword='AcquisitionNumber',
          vr='IS',
          value=1,
      ),
      dict(
          testcase_name='AcquisitionDateTime',
          keyword='AcquisitionDateTime',
          vr='DT',
          value='19700101120000',
      ),
      dict(
          testcase_name='AcquisitionDate',
          keyword='AcquisitionDate',
          vr='DA',
          value='20060806',
      ),
      dict(
          testcase_name='AcquisitionTime',
          keyword='AcquisitionTime',
          vr='TM',
          value='121212',
      ),
  ])
  def test_not_same_acquisition_if_different_tags_dicom(
      self, keyword, vr, value
  ):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as test_dcm:
      test_dcm[keyword] = pydicom.DataElement(keyword, vr, value)
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        self.assertFalse(generic_dicom_handler._same_acquisition(test_dcm, dcm))

  def test_same_acquisition(self):
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as test_dcm:
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as dcm:
        self.assertTrue(generic_dicom_handler._same_acquisition(test_dcm, dcm))

  def test_process_buffered_mri_volume_empty_input(self):
    self.assertEmpty(
        list(generic_dicom_handler._process_buffered_mri_volume([]))
    )

  def test_process_buffered_mri_volume_single_value(self):
    np.testing.assert_array_equal(
        list(
            generic_dicom_handler._process_buffered_mri_volume(
                [np.zeros((10, 10), dtype=np.uint8)]
            )
        ),
        [np.full((10, 10), 255, dtype=np.uint8)],
    )

  def test_process_buffered_mri_volume_multiple_values(self):
    np.testing.assert_array_equal(
        list(
            generic_dicom_handler._process_buffered_mri_volume([
                np.full((10, 10), 1, dtype=np.uint8),
                np.full((10, 10), 2, dtype=np.uint8),
                np.full((10, 10), 3, dtype=np.uint8),
            ])
        ),
        [
            np.full((10, 10), 0, dtype=np.uint8),
            np.full((10, 10), 128, dtype=np.uint8),
            np.full((10, 10), 255, dtype=np.uint8),
        ],
    )

  def test_process_multiple_mri_volumes(self):
    temp_dir = self.create_tempdir()
    dicom_paths = []
    for index in range(4):
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as test_dcm:
        test_dcm.Modality = 'MR'
        test_dcm.PixelData = np.full(
            (512, 512), index, dtype=np.uint16
        ).tobytes()
        test_dcm.AcquisitionNumber = index
        dcm_path = os.path.join(temp_dir, f'test{index}.dcm')
        test_dcm.save_as(dcm_path)
        dicom_paths.append(dcm_path)
    results = test_utils.flatten_data_acquisition(
        _generic_dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator(dicom_paths)
        )
    )
    np.testing.assert_array_equal(
        results, [np.full((512, 512, 1), 255, dtype=np.uint8)] * 4
    )

  def test_process_single_mri_volume(self):
    temp_dir = self.create_tempdir()
    dicom_paths = []
    for index in range(1, 5):
      with pydicom.dcmread(
          test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
      ) as test_dcm:
        test_dcm.Modality = 'MR'
        test_dcm.PixelData = np.full(
            (512, 512), index, dtype=np.uint16
        ).tobytes()
        dcm_path = os.path.join(temp_dir, f'test{index}.dcm')
        test_dcm.save_as(dcm_path)
        dicom_paths.append(dcm_path)
    results = list(
        _generic_dicom_handler.process_files(
            [], {}, abstract_handler.InputFileIterator(dicom_paths)
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(
        results[0].acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
    )
    expected = [
        np.full((512, 512, 1), int(round(v / 3 * 255)), dtype=np.uint8)
        for v in range(4)
    ]
    np.testing.assert_array_equal(
        list(results[0].acquision_data_source_iterator), expected
    )

  def test_process_mri_ct_mri(self):
    temp_dir = self.create_tempdir()
    dicom_paths = []
    ct_dcm_path = test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    with pydicom.dcmread(ct_dcm_path) as test_dcm:
      test_dcm.Modality = 'MR'
      test_dcm.PixelData = np.full((512, 512), 3, dtype=np.uint16).tobytes()
      mri_path = os.path.join(temp_dir, 'mr.dcm')
      test_dcm.save_as(mri_path)
      dicom_paths.append(mri_path)
    dicom_paths.append(ct_dcm_path)
    dicom_paths.append(mri_path)
    results = _generic_dicom_handler.process_files(
        [], {}, abstract_handler.InputFileIterator(dicom_paths)
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
    )
    expected_mri = np.full((512, 512, 1), 255, dtype=np.uint8)
    np.testing.assert_array_equal(
        list(ds.acquision_data_source_iterator), [expected_mri]
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_CT_VOLUME,
    )
    ct_results = list(ds.acquision_data_source_iterator)
    self.assertLen(ct_results, 1)
    self.assertEqual(ct_results[0].shape, (512, 512, 3))
    np.testing.assert_array_equal(
        np.unique(ct_results[0]), [0, 1, 2, 3, 4, 5, 128]
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
    )
    np.testing.assert_array_equal(
        list(ds.acquision_data_source_iterator), [expected_mri]
    )

    with self.assertRaises(StopIteration):
      next(results)

  def test_process_mri_ct(self):
    temp_dir = self.create_tempdir()
    dicom_paths = []
    ct_dcm_path = test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    with pydicom.dcmread(ct_dcm_path) as test_dcm:
      test_dcm.Modality = 'MR'
      test_dcm.PixelData = np.full((512, 512), 3, dtype=np.uint16).tobytes()
      mri_path = os.path.join(temp_dir, 'mr.dcm')
      test_dcm.save_as(mri_path)
      dicom_paths.append(mri_path)
    dicom_paths.append(ct_dcm_path)
    results = _generic_dicom_handler.process_files(
        [], {}, abstract_handler.InputFileIterator(dicom_paths)
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
    )
    expected_mri = np.full((512, 512, 1), 255, dtype=np.uint8)
    np.testing.assert_array_equal(
        list(ds.acquision_data_source_iterator), [expected_mri]
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_CT_VOLUME,
    )
    ct_data = list(ds.acquision_data_source_iterator)
    self.assertLen(ct_data, 1)
    self.assertEqual(ct_data[0].shape, (512, 512, 3))
    np.testing.assert_array_equal(
        np.unique(ct_data[0]), [0, 1, 2, 3, 4, 5, 128]
    )
    with self.assertRaises(StopIteration):
      next(results)

  def test_process_ct_mri(self):
    temp_dir = self.create_tempdir()
    dicom_paths = []
    ct_dcm_path = test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    dicom_paths.append(ct_dcm_path)
    with pydicom.dcmread(ct_dcm_path) as test_dcm:
      test_dcm.Modality = 'MR'
      test_dcm.PixelData = np.full((512, 512), 3, dtype=np.uint16).tobytes()
      mri_path = os.path.join(temp_dir, 'mr.dcm')
      test_dcm.save_as(mri_path)
      dicom_paths.append(mri_path)
    results = _generic_dicom_handler.process_files(
        [], {}, abstract_handler.InputFileIterator(dicom_paths)
    )

    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_CT_VOLUME,
    )
    ct_data = list(ds.acquision_data_source_iterator)
    self.assertLen(ct_data, 1)
    self.assertEqual(ct_data[0].shape, (512, 512, 3))
    np.testing.assert_array_equal(
        np.unique(ct_data[0]), [0, 1, 2, 3, 4, 5, 128]
    )
    ds = next(results)
    self.assertEqual(
        ds.acquision_data_source,
        abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
    )
    expected_mri = np.full((512, 512, 1), 255, dtype=np.uint8)
    np.testing.assert_array_equal(
        list(ds.acquision_data_source_iterator), [expected_mri]
    )
    with self.assertRaises(StopIteration):
      next(results)

  def test_process_concatenated_dicom_raises(self):
    temp_dir = self.create_tempdir()
    dicom_paths = []
    path = os.path.join(temp_dir, 'test.dcm')
    with pydicom.dcmread(
        test_utils.testdata_path('ct', 'test_series', 'image0.dcm')
    ) as test_dcm:
      test_dcm.ConcatenationUID = '1.2.3'
      test_dcm.save_as(path)
      dicom_paths.append(path)
    with self.assertRaisesRegex(
        data_accessor_errors.DicomError,
        '.*Concatenated DICOM are not supported.*',
    ):
      test_utils.flatten_data_acquisition(
          _generic_dicom_handler.process_files(
              [], {}, abstract_handler.InputFileIterator(dicom_paths)
          )
      )

  def test_ct_rescale_slope_and_intercept(self):
    with pydicom.dcmread(
        _mock_ct_dicom(
            os.path.join(self.create_tempdir(), 'test.dcm'),
            'MONOCHROME2',
            0,
            1000,
        )
    ) as dcm:
      dcm.RescaleSlope = 66
      dcm.RescaleIntercept = -33000
      img = np.arange(1002, dtype=np.uint16)
      rescaled_img = generic_dicom_handler.rescale_ct_imaging(img, dcm)
      self.assertEqual(rescaled_img.dtype, np.int32)
      self.assertEqual(np.min(rescaled_img), -33000)
      self.assertEqual(np.max(rescaled_img), 33066)

  def test_ct_window(self):
    dcm_path = _mock_ct_dicom(
        os.path.join(self.create_tempdir(), 'test.dcm'),
        'MONOCHROME2',
        0,
        1000,
    )
    with pydicom.dcmread(dcm_path) as dcm:
      dcm.RescaleSlope = 2
      dcm.RescaleIntercept = -32767
      image_pixels = np.zeros((3, 3), dtype=np.uint16)
      for i in range(9):
        image_pixels[int(i // 3), int(i % 3)] = 32767 * i / 8
      dcm.PixelData = image_pixels.tobytes()
      dcm.save_as(dcm_path)
    handler = generic_dicom_handler.GenericDicomHandler({
        'CT': generic_dicom_handler.ModalityDefaultImageTransform(
            get_image_transform_op=lambda dcm: generic_dicom_handler.TraditionalWindow(
                4096, 57344
            ),
        )
    })
    result = test_utils.flatten_data_acquisition(
        handler.process_files(
            [], {}, abstract_handler.InputFileIterator([dcm_path])
        )
    )
    self.assertLen(result, 1)
    np.testing.assert_array_equal(
        np.squeeze(result[0], axis=-1),
        np.asarray(
            [
                [0, 0, 36],
                [73, 109, 146],
                [182, 219, 255],
            ],
            dtype=np.uint8,
        ),
    )


if __name__ == '__main__':
  absltest.main()
