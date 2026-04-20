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

"""Unit tests for pathology 2.0 endpoint predictor."""

import contextlib
import dataclasses
import json
import typing
from typing import Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_wsi import configuration
from data_accessors.dicom_wsi import data_accessor
from data_accessors.dicom_wsi import data_accessor_definition
from data_accessors.dicom_wsi import icc_profile_cache
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate
from data_accessors.utils import test_utils
from serving.serving_framework import model_runner
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys

_DEBUG_SETTINGS = configuration.ConfigurationSettings(
    224,
    224,
    None,
    configuration.IccProfileCacheConfiguration(testing=True),
)


def _dicom_instance_path(dcm: pydicom.Dataset) -> dicom_path.Path:
  return dicom_path.FromString(
      f'/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
  )


class MockModelRunner:
  """Mock embedding, return mean for each channel in patch."""

  def batch_model(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Compute and return mock embeddings."""
    return [np.mean(d, axis=(1, 2)) for d in data]


_mock_model_runner = typing.cast(model_runner.ModelRunner, MockModelRunner())


class DicomDigitalPathologyDataTest(parameterized.TestCase):

  def test_get_dicom_patches_instance_not_found(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    path = _dicom_instance_path(dcm)
    instance = data_accessor_definition.json_to_dicom_wsi_image(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        {
            _InstanceJsonKeys.PATCH_COORDINATES: [
                dataclasses.asdict(c) for c in coordinates
            ],
            _InstanceJsonKeys.DICOM_SOURCE: [
                str(dicom_path.FromPath(path, instance_uid='1.42'))
            ],
        },
        _DEBUG_SETTINGS,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomPathError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_patches_dicom_slide_not_found(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = dicom_path.FromString(
        '/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/1.42/series/1.42'
    )
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.json_to_dicom_wsi_image(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        {
            _InstanceJsonKeys.PATCH_COORDINATES: [
                dataclasses.asdict(c) for c in coordinates
            ],
            _InstanceJsonKeys.DICOM_SOURCE: [str(path)],
        },
        _DEBUG_SETTINGS,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomPathError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_patches_bad_path(self):
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        [dicom_path.FromString('https://bad_path')],
        {},
        coordinates,
        [],
    )
    with self.assertRaises(data_accessor_errors.DicomPathError):
      list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )

  @parameterized.parameters(['', 'mock_bearer_token'])
  def test_get_dicom_patches(self, bearer_token):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(bearer_token),
        [path],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(np.min(images[0]), 27)
    self.assertEqual(np.max(images[0]), 255)

  @parameterized.parameters(['', 'mock_bearer_token'])
  def test_get_dicom_whole_slide(self, bearer_token):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = []
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(bearer_token),
        [path],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEmpty(instance.patch_coordinates)
    self.assertEqual(images[0].shape, (700, 1152, 3))
    self.assertEqual(np.min(images[0]), 19)
    self.assertEqual(np.max(images[0]), 255)

  @parameterized.named_parameters([
      dict(testcase_name='no_extension', extension={}),
      dict(
          testcase_name='defines_icc_profile_transform',
          extension={
              _InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB',
          },
      ),
  ])
  @mock.patch.object(
      dicom_slide, 'create_icc_profile_transformation', autospec=True
  )
  def test_get_patches_from_dicom_with_out_icc_profile_not_create_transform(
      self, create_transform, extension
  ):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(
            'mock_bearer_token'
        ),
        [path],
        extension,
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(np.min(images[0]), 27)
    self.assertEqual(np.max(images[0]), 255)
    create_transform.assert_not_called()

  def test_get_dicom_patches_no_pixel_spacing(self):
    dcm = pydicom.dcmread(test_utils.testdata_path('wsi', 'test.dcm'))
    # remove pixel spacing
    del dcm['SharedFunctionalGroupsSequence']  # pylint: disable=invalid-delete
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(
            'mock_bearer_token'
        ),
        [path],
        {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(round(float(np.min(images[0])), 4), 0.0)
    self.assertEqual(np.max(images[0]), 255)

  @parameterized.named_parameters([
      dict(
          testcase_name='VL_SLIDE_COORDINATES_MIROSCOPIC_IMAGE_SOP_CLASS_UID',
          sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.3',
      ),
      dict(
          testcase_name='VL_MIROSCOPIC_IMAGE_SOP_CLASS_UID ',
          sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.2',
      ),
  ])
  def test_get_dicom_patches_from_non_tiled_dicom(self, sop_class_uid):
    dcm = pydicom.dcmread(test_utils.testdata_path('wsi', 'test.dcm'))
    # remove pixel spacing
    del dcm['SharedFunctionalGroupsSequence']  # pylint: disable=invalid-delete
    dcm.file_meta.MediaStorageSOPClassUID = sop_class_uid
    dcm.SOPClassUID = sop_class_uid
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(
            'mock_bearer_token'
        ),
        [path],
        {_InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
    self.assertLen(images, 1)
    self.assertEqual(images[0].shape, (224, 224, 3))
    self.assertEqual(round(float(np.min(images[0])), 4), 0.0)
    self.assertEqual(np.max(images[0]), 255)

  def test_get_dicom_patches_from_sparse_dicom_raises(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    del dcm['00209311']  # pylint: disable=invalid-delete
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('bearer_token'),
        [path],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomTiledFullError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_patches_from_missing_instance_raises(
      self,
  ):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('bearer_token'),
        [dicom_path.FromPath(path, instance_uid='1.42')],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.DicomPathError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  @parameterized.parameters('mock_bearer_token', '')
  def test_repeated_get_dicom_patches_does_not_re_int(self, bearer_token):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory(bearer_token),
        [path],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      # repeate prior call
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      # check patch results are as expected.
      self.assertLen(images, 1)
      self.assertEqual(images[0].shape, (224, 224, 3))
      self.assertEqual(np.min(images[0]), 27)
      self.assertEqual(np.max(images[0]), 255)

  @parameterized.parameters([1, 1.2, 'abc', ([],)])
  def test_get_ez_wsi_state_invalid_value_raises(self, val):
    with self.assertRaises(data_accessor_errors.EzWsiStateError):
      data_accessor._get_ez_wsi_state({_InstanceJsonKeys.EZ_WSI_STATE: val})

  def test_get_ez_wsi_state_default(self):
    self.assertEqual(data_accessor._get_ez_wsi_state({}), {})

  def test_get_ez_wsi_state_expected(self):
    expected = {'abc': 123}
    self.assertEqual(
        data_accessor._get_ez_wsi_state(
            {_InstanceJsonKeys.EZ_WSI_STATE: expected}
        ),
        expected,
    )

  def test_get_dicom_instances_with_different_transfer_syntax_raise(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    # define two concatenation instances with different transfer syntaxs.
    dcm.InConcatenationNumber = 1
    dcm.ConcatenationUID = '1.43'
    dcm.ConcatenationFrameOffsetNumber = 0
    dcm2 = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    dcm2.file_meta.MediaStorageSOPInstanceUID = '1.42'
    dcm2.ConcatenationFrameOffsetNumber = dcm.NumberOfFrames
    dcm2.SOPInstanceUID = '1.42'
    dcm2.InConcatenationNumber = 2
    dcm2.ConcatenationUID = '1.43'
    dcm2.file_meta.TransferSyntaxUID = '1.2.840.10008.1.​2.​1'
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        [path],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      mk_dicom_stores[store_path].add_instance(dcm2)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'All DICOM instances in a pyramid level are required to have same'
          ' TransferSyntaxUID.',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_get_dicom_instances_invalid_tags_raises(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    del dcm['00080008']  # pylint: disable=invalid-delete
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    instance = data_accessor_definition.DicomWSIImage(
        credential_factory.TokenPassthroughCredentialFactory('mock_token'),
        [path],
        {},
        coordinates,
        [],
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'DICOM instance missing required tags.',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_can_not_find_dicom_level_raises(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          path,
      )
      # error occures due to instance requesting predictions for an instance
      # which is not defined in the metadata.
      metadata = ds.json_metadata()
      metadata = metadata.replace(dcm.SOPInstanceUID, '1.42')
      metadata = json.loads(metadata)
      # modifying metadata to remove instance.
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {_InstanceJsonKeys.EZ_WSI_STATE: metadata},
          coordinates,
          [],
      )
      with self.assertRaises(data_accessor_errors.LevelNotFoundError):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_level_resize_greater_than_8x_raises(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {_InstanceJsonKeys.X_ORIGIN: 0, _InstanceJsonKeys.Y_ORIGIN: 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {
              _InstanceJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  image_dimension_utils.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 9),
                      int(dcm.TotalPixelMatrixRows // 9),
                  )
              )
          },
          coordinates,
          [],
      )
      with self.assertRaisesRegex(
          data_accessor_errors.DicomImageDownsamplingTooLargeError,
          'Image downsampling, 9.09091X exceeds 8X',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_level_resize(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {
              _InstanceJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  image_dimension_utils.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 3),
                      int(dcm.TotalPixelMatrixRows // 3),
                  )
              )
          },
          coordinates,
          [],
      )
      images = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )

      # check patch results are as expected.
      self.assertLen(images, 1)
      self.assertEqual(images[0].shape, (224, 224, 3))
      self.assertEqual(np.min(images[0]), 51)
      self.assertEqual(np.max(images[0]), 239)

  def test_dicom_patch_outside_level_dim(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {
              _InstanceJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  image_dimension_utils.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 7),
                      int(dcm.TotalPixelMatrixRows // 7),
                  )
              )
          },
          coordinates,
          [],
      )
      with self.assertRaisesRegex(
          data_accessor_errors.PatchOutsideOfImageDimensionsError,
          'Patch dimensions.*fall outside of DICOM level pyramid imaging'
          ' dimensions.*',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_bits_allocated_not_8_raises(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    dcm.BitsAllocated = 12
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {},
          coordinates,
          [],
      )
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'DICOM contains instances with imaging bits allocated != 8',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  def test_dicom_icc_profile_correction_changes_pixel_values(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB'},
          coordinates,
          [],
      )
      srgb_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB'},
          coordinates,
          [],
      )
      rommrgb_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      self.assertLen(srgb_result, 1)
      self.assertLen(rommrgb_result, 1)
      # color normalization changes pixel values
      self.assertGreater(np.max(np.abs(rommrgb_result[0] - srgb_result[0])), 0)

  def test_dicom_icc_profile_no_effect_of_correction_for_same_profile(self):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {},
          coordinates,
          [],
      )
      none_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB'},
          coordinates,
          [],
      )
      rommrgb_result = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      self.assertLen(none_result, 1)
      self.assertLen(rommrgb_result, 1)
      # no change in changes pixel values
      self.assertEqual(np.max(np.abs(rommrgb_result[0] - none_result[0])), 0)

  @parameterized.named_parameters([
      dict(testcase_name='no_profile_transform_defined', exensions={}),
      dict(
          testcase_name='none_transform',
          exensions={
              _InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'NONE'
          },
      ),
  ])
  @mock.patch.object(icc_profile_cache, 'get_dicom_icc_profile', autospec=True)
  def test_dicom_icc_profile_not_called(self, mock_get_profile, exensions):
    dcm = pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_instance_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.DicomWSIImage(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          [path],
          exensions,
          coordinates,
          [],
      )
      _ = list(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).data_iterator()
      )
      mock_get_profile.assert_not_called()

  @parameterized.named_parameters([
      dict(testcase_name='monochrome_1', shape=(224, 224)),
      dict(testcase_name='monochrome_2', shape=(224, 224, 1)),
      dict(testcase_name='rgb', shape=(224, 224, 3)),
      dict(testcase_name='rgba', shape=(224, 224, 4)),
  ])
  def test_fetch_patch_bytes_norms_monochrome_images_to_three_channels(
      self, shape
  ):
    mem = np.zeros(shape=shape, dtype=np.uint8)
    mock_patch = mock.create_autospec(dicom_slide.DicomPatch, instance=True)
    mock_patch.image_bytes.return_value = mem
    self.assertEqual(
        data_accessor._fetch_image_bytes(mock_patch, None).shape,
        (224, 224, 3),
    )

  def test_validate_dicom_image_accessor_raises(self):
    with self.assertRaises(data_accessor_errors.UnapprovedDicomStoreError):
      data_accessor._validate_dicom_image_accessor(
          'http://test_bucket/google.png',
          dataclasses.replace(
              _DEBUG_SETTINGS,
              approved_dicom_stores=['http://abc', 'http://123'],
          ),
      )

  @parameterized.parameters(['http://abc/studies', 'http://123/studies'])
  def test_validate_dicom_image_accessor_valid(self, source):
    self.assertIsNone(
        data_accessor._validate_dicom_image_accessor(
            source,
            dataclasses.replace(
                _DEBUG_SETTINGS,
                approved_dicom_stores=['http://abc', 'http://123'],
            ),
        )
    )

  def test_validate_default_dicom_image_accessor_valid(self):
    self.assertIsNone(
        data_accessor._validate_dicom_image_accessor(
            'http://test_bucket/studies', _DEBUG_SETTINGS
        )
    )

  def test_is_accessor_data_embedded_in_request(self):
    coordinates = [
        patch_coordinate.create_patch_coordinate(
            {'x_origin': 0, 'y_origin': 0},
            default_width=_DEBUG_SETTINGS.endpoint_input_width,
            default_height=_DEBUG_SETTINGS.endpoint_input_height,
        )
    ]
    with pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    ) as dcm:
      path = _dicom_instance_path(dcm)
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          credential_factory.TokenPassthroughCredentialFactory('mock_token'),
          {
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  dataclasses.asdict(c) for c in coordinates
              ],
              _InstanceJsonKeys.DICOM_SOURCE: str(path),
          },
          _DEBUG_SETTINGS,
          [],
      )
      self.assertFalse(
          data_accessor.DicomDigitalPathologyData(
              instance, _DEBUG_SETTINGS
          ).is_accessor_data_embedded_in_request()
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_patch_coordinates',
          coordinates=[],
          expected=1,
      ),
      dict(
          testcase_name='one_patch',
          coordinates=[
              patch_coordinate.create_patch_coordinate(
                  {'x_origin': 0, 'y_origin': 0},
                  default_width=_DEBUG_SETTINGS.endpoint_input_width,
                  default_height=_DEBUG_SETTINGS.endpoint_input_height,
              )
          ],
          expected=1,
      ),
      dict(
          testcase_name='two_patches',
          coordinates=[
              patch_coordinate.create_patch_coordinate(
                  {'x_origin': 0, 'y_origin': 0},
                  default_width=_DEBUG_SETTINGS.endpoint_input_width,
                  default_height=_DEBUG_SETTINGS.endpoint_input_height,
              ),
              patch_coordinate.create_patch_coordinate(
                  {'x_origin': 0, 'y_origin': 0},
                  default_width=_DEBUG_SETTINGS.endpoint_input_width,
                  default_height=_DEBUG_SETTINGS.endpoint_input_height,
              ),
          ],
          expected=2,
      ),
  )
  def test_accessor_length(self, coordinates, expected):
    with pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    ) as dcm:
      path = _dicom_instance_path(dcm)
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        mk_dicom_stores[store_path].add_instance(dcm)
        instance = data_accessor_definition.json_to_dicom_wsi_image(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.PATCH_COORDINATES: [
                    dataclasses.asdict(c) for c in coordinates
                ],
                _InstanceJsonKeys.DICOM_WEB_URI: str(path),
            },
            _DEBUG_SETTINGS,
            [],
        )
        self.assertLen(
            data_accessor.DicomDigitalPathologyData(instance, _DEBUG_SETTINGS),
            expected,
        )

  def test_parse_json_invalid_extensions_raises(self):
    with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
      data_accessor_definition.json_to_dicom_wsi_image(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.EXTENSIONS: {1: 1},
              _InstanceJsonKeys.DICOM_WEB_URI: (
                  'http://localhost:12345/studies/1.2.3/series/4.5.6/instances/7.8.9'
              ),
          },
          _DEBUG_SETTINGS,
          [],
      )

  def test_dicom_series_defines_wsi_and_microscope_image_raises(self):
    cf = credential_factory.NoAuthCredentialsFactory()
    with pydicom.dcmread(
        test_utils.testdata_path(
            'wsi', 'multiframe_camelyon_challenge_image.dcm'
        )
    ) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
      study_uid = dcm.StudyInstanceUID
      series_uid = dcm.SeriesInstanceUID
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      for dcm_path in (
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          ),
          test_utils.testdata_path(
              'slide_coordinates_microscopic_image',
              'test_slide_coordinates.dcm',
          ),
      ):
        with pydicom.dcmread(dcm_path) as dcm:
          dcm.StudyInstanceUID = study_uid
          dcm.SeriesInstanceUID = series_uid
          mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          cf,
          {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
          _DEBUG_SETTINGS,
          dicom_web_interface.DicomWebInterface(cf).get_instances(path),
      )
      with self.assertRaisesRegex(
          data_accessor_errors.DicomError,
          'Cannot return for DICOM images for  VL_Whole_Slide_Microscopy_Images'
          ' and DICOM Microscopic_Images in the same request.*',
      ):
        list(
            data_accessor.DicomDigitalPathologyData(
                instance, _DEBUG_SETTINGS
            ).data_iterator()
        )

  @parameterized.parameters([1, 2, 3])
  def test_dicom_series_with_microscope_images_returns_expected_number_of_images(
      self, expected
  ):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'slide_coordinates_microscopic_image',
        'test_slide_coordinates.dcm',
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      for index in range(expected):
        with pydicom.dcmread(dcm_path) as dcm:
          dcm.SOPInstanceUID = f'1.3.4.5{index}'
          mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          cf,
          {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
          _DEBUG_SETTINGS,
          dicom_web_interface.DicomWebInterface(cf).get_instances(path),
      )
      results = data_accessor.DicomDigitalPathologyData(
          instance, _DEBUG_SETTINGS
      ).data_acquisition_iterator()
      for _ in range(expected):
        ds = next(results)
        self.assertEqual(
            ds.acquision_data_source,
            abstract_data_accessor.AccessorDataSource.DICOM_MICROSCOPY_IMAGES,
        )
        self.assertLen(list(ds.acquision_data_source_iterator), 1)
      with self.assertRaises(StopIteration):
        next(results)

  @parameterized.parameters([1, 2, 3])
  def test_preload_microscope_images_returns_expected_number_of_images(
      self, expected
  ):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'slide_coordinates_microscopic_image',
        'test_slide_coordinates.dcm',
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      for index in range(expected):
        with pydicom.dcmread(dcm_path) as dcm:
          dcm.SOPInstanceUID = f'1.3.4.5{index}'
          mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          cf,
          {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
          _DEBUG_SETTINGS,
          dicom_web_interface.DicomWebInterface(cf).get_instances(path),
      )
      da = data_accessor.DicomDigitalPathologyData(instance, _DEBUG_SETTINGS)
      with contextlib.ExitStack() as stack:
        da.load_data(stack)
        self.assertLen(list(da.data_iterator()), expected)

  def test_dicom_series_with_wsi_returns_one_number_of_image(self):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        mk_dicom_stores[store_path].add_instance(dcm)
        instance = data_accessor_definition.json_to_dicom_wsi_image(
            cf,
            {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
            _DEBUG_SETTINGS,
            dicom_web_interface.DicomWebInterface(cf).get_instances(path),
        )
        self.assertLen(
            list(
                data_accessor.DicomDigitalPathologyData(
                    instance, _DEBUG_SETTINGS
                ).data_iterator()
            ),
            1,
        )

  def test_dicom_series_with_wsi_thumbnail_returns_one_number_of_image(self):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        dcm.NumberOfFrames = 1
        dcm.ImageType = ['THUMBNAIL']
        mk_dicom_stores[store_path].add_instance(dcm)
        instance = data_accessor_definition.json_to_dicom_wsi_image(
            cf,
            {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
            _DEBUG_SETTINGS,
            dicom_web_interface.DicomWebInterface(cf).get_instances(path),
        )
        self.assertLen(
            list(
                data_accessor.DicomDigitalPathologyData(
                    instance, _DEBUG_SETTINGS
                ).data_iterator()
            ),
            1,
        )

  @parameterized.parameters(['LABEL', 'OVERVIEW'])
  def test_dicom_series_with_only_invalid_wsi_instances_raises(
      self, image_type_str
  ):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        dcm.NumberOfFrames = 1
        dcm.ImageType = [image_type_str]
        mk_dicom_stores[store_path].add_instance(dcm)
        instance = data_accessor_definition.json_to_dicom_wsi_image(
            cf,
            {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
            _DEBUG_SETTINGS,
            dicom_web_interface.DicomWebInterface(cf).get_instances(path),
        )
        with self.assertRaisesRegex(
            data_accessor_errors.LevelNotFoundError,
            '.* is missing WSI pyramid and WSI thumbnail image.*',
        ):
          list(
              data_accessor.DicomDigitalPathologyData(
                  instance, _DEBUG_SETTINGS
              ).data_iterator()
          )

  def test_dicom_contains_invalid_metadata_raises(self):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        dcm.NumberOfFrames = 1
        dcm.ConcatenationFrameOffsetNumber = 2
        dcm.ImageType = ['THUMBNAIL']
        mk_dicom_stores[store_path].add_instance(dcm)
        instance = data_accessor_definition.json_to_dicom_wsi_image(
            cf,
            {_InstanceJsonKeys.DICOM_WEB_URI: str(path)},
            _DEBUG_SETTINGS,
            dicom_web_interface.DicomWebInterface(cf).get_instances(path),
        )
        with self.assertRaisesRegex(
            data_accessor_errors.DicomError,
            '.*DICOM metadata error.*',
        ):
          list(
              data_accessor.DicomDigitalPathologyData(
                  instance, _DEBUG_SETTINGS
              ).data_iterator()
          )

  def test_dicom_request_contains_invalid_ez_wsi_state_raises(self):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      path = _dicom_instance_path(dcm).GetSeriesPath()
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        mk_dicom_stores[store_path].add_instance(dcm)
        instance = data_accessor_definition.json_to_dicom_wsi_image(
            cf,
            {
                _InstanceJsonKeys.EXTENSIONS: {
                    _InstanceJsonKeys.EZ_WSI_STATE: {'1.2.3': {}}
                },
                _InstanceJsonKeys.DICOM_WEB_URI: str(path),
            },
            _DEBUG_SETTINGS,
            dicom_web_interface.DicomWebInterface(cf).get_instances(path),
        )
        with self.assertRaisesRegex(
            data_accessor_errors.EzWsiStateError,
            '.*Error decoding embedding request JSON metadata.*',
        ):
          list(
              data_accessor.DicomDigitalPathologyData(
                  instance, _DEBUG_SETTINGS
              ).data_iterator()
          )

  def test_dicom_request_from_two_concatenated_instances_returns_one_image(
      self,
  ):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      instance_path = _dicom_instance_path(dcm)
      series_path = instance_path.GetSeriesPath()
      store_path = str(instance_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      with pydicom.dcmread(dcm_path) as dcm:
        dcm.ConcatenationUID = '1.2'
        dcm.InConcatenationNumber = 1
        dcm.InConcatenationTotalNumber = 2
        dcm.NumberOfFrames = dcm.NumberOfFrames // 2
        dcm.ConcatenationFrameOffsetNumber = 0
        mk_dicom_stores[store_path].add_instance(dcm)
        path_1 = _dicom_instance_path(dcm)
      with pydicom.dcmread(dcm_path) as dcm:
        dcm.SOPInstanceUID = '1.2.3.4.5'
        dcm.ConcatenationUID = '1.2'
        dcm.InConcatenationNumber = 2
        dcm.InConcatenationTotalNumber = 2
        dcm.ConcatenationFrameOffsetNumber = dcm.NumberOfFrames // 2
        dcm.NumberOfFrames = dcm.NumberOfFrames - (dcm.NumberOfFrames // 2)
        mk_dicom_stores[store_path].add_instance(dcm)
        path_2 = _dicom_instance_path(dcm)
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          cf,
          {
              _InstanceJsonKeys.DICOM_SOURCE: [str(path_1), str(path_2)],
          },
          _DEBUG_SETTINGS,
          dicom_web_interface.DicomWebInterface(cf).get_instances(series_path),
      )
      self.assertLen(
          list(
              data_accessor.DicomDigitalPathologyData(
                  instance, _DEBUG_SETTINGS
              ).data_iterator()
          ),
          1,
      )

  def test_dicom_request_from_single_concatenated_instances_returns_one_image(
      self,
  ):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      instance_path = _dicom_instance_path(dcm)
      series_path = instance_path.GetSeriesPath()
      store_path = str(instance_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      with pydicom.dcmread(dcm_path) as dcm:
        dcm.ConcatenationUID = '1.2'
        dcm.InConcatenationNumber = 1
        dcm.InConcatenationTotalNumber = 2
        dcm.NumberOfFrames = dcm.NumberOfFrames // 2
        dcm.ConcatenationFrameOffsetNumber = 0
        mk_dicom_stores[store_path].add_instance(dcm)
        path_1 = _dicom_instance_path(dcm)
      with pydicom.dcmread(dcm_path) as dcm:
        dcm.SOPInstanceUID = '1.2.3.4.5'
        dcm.ConcatenationUID = '1.2'
        dcm.InConcatenationNumber = 2
        dcm.InConcatenationTotalNumber = 2
        dcm.ConcatenationFrameOffsetNumber = dcm.NumberOfFrames // 2
        dcm.NumberOfFrames = dcm.NumberOfFrames - (dcm.NumberOfFrames // 2)
        mk_dicom_stores[store_path].add_instance(dcm)
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          cf,
          {
              _InstanceJsonKeys.DICOM_SOURCE: [str(path_1)],
          },
          _DEBUG_SETTINGS,
          dicom_web_interface.DicomWebInterface(cf).get_instances(series_path),
      )
      self.assertLen(
          list(
              data_accessor.DicomDigitalPathologyData(
                  instance, _DEBUG_SETTINGS
              ).data_iterator()
          ),
          1,
      )

  @parameterized.parameters([0, 1, 2])
  def test_dicom_request_from_two_pyramids_levels_returns_two_images(
      self, max_parallel_download_workers
  ):
    cf = credential_factory.NoAuthCredentialsFactory()
    dcm_path = test_utils.testdata_path(
        'wsi', 'multiframe_camelyon_challenge_image.dcm'
    )
    with pydicom.dcmread(dcm_path) as dcm:
      instance_path = _dicom_instance_path(dcm)
      series_path = instance_path.GetSeriesPath()
      store_path = str(instance_path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      with pydicom.dcmread(dcm_path) as dcm:
        mk_dicom_stores[store_path].add_instance(dcm)
        path_1 = _dicom_instance_path(dcm)
      with pydicom.dcmread(dcm_path) as dcm:
        dcm.SOPInstanceUID = '1.2.3.4.5'
        mk_dicom_stores[store_path].add_instance(dcm)
        path_2 = _dicom_instance_path(dcm)
      settings = dataclasses.replace(
          _DEBUG_SETTINGS,
          max_parallel_download_workers=max_parallel_download_workers,
      )
      instance = data_accessor_definition.json_to_dicom_wsi_image(
          cf,
          {
              _InstanceJsonKeys.DICOM_SOURCE: [str(path_1), str(path_2)],
          },
          settings,
          dicom_web_interface.DicomWebInterface(cf).get_instances(series_path),
      )
      results = data_accessor.DicomDigitalPathologyData(
          instance, settings
      ).data_acquisition_iterator()
      for _ in range(2):
        ds = next(results)
        self.assertEqual(
            ds.acquision_data_source,
            abstract_data_accessor.AccessorDataSource.DICOM_WSI_MICROSCOPY_PYRAMID_LEVEL,
        )
        self.assertLen(list(ds.acquision_data_source_iterator), 1)
      with self.assertRaises(StopIteration):
        next(results)


if __name__ == '__main__':
  absltest.main()
