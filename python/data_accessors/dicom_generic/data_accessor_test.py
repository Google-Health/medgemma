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

"""Unit tests for dicom_generic data accessor."""

import contextlib
import copy
from typing import Any, Mapping, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
import pydicom
import requests_mock

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor
from data_accessors.dicom_generic import data_accessor_definition
from data_accessors.utils import test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys

_MOCK_DICOM_STORE_PATH = 'https://www.mock_dicom_store.com'


def _create_dicom_web_uri(dcm: pydicom.FileDataset) -> str:
  return f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'


def _test_load_generic_dicom(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    dcm: pydicom.FileDataset,
) -> Sequence[np.ndarray]:
  with dicom_store_mock.MockDicomStores(_MOCK_DICOM_STORE_PATH) as dicom_store:
    dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
    dwi = dicom_web_interface.DicomWebInterface(
        credential_factory_module.NoAuthCredentialsFactory()
    )
    instance_path = dicom_path.FromString(
        json_instance[_InstanceJsonKeys.DICOM_WEB_URI]
    )
    instance = data_accessor_definition.json_to_generic_dicom_image(
        credential_factory,
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
        dicom_instances_metadata=dwi.get_instances(instance_path),
    )
    return list(data_accessor.DicomGenericData(instance).data_iterator())


class DataAccessorTest(parameterized.TestCase):

  def test_dicom_missing_metadata_raises(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: [],
      }
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        instance = data_accessor_definition.json_to_generic_dicom_image(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            default_patch_width=256,
            default_patch_height=256,
            require_patch_dim_match_default_dim=False,
            dicom_instances_metadata=[],
        )
        with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
          list(data_accessor.DicomGenericData(instance).data_iterator())

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10}
          ],
          expected_shape=(10, 10, 1),
      ),
      dict(
          testcase_name='empty_patch_list',
          patch_coordinates=[],
          expected_shape=(1024, 1024, 1),
      ),
  )
  def test_dicom_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
      }
      img = _test_load_generic_dicom(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          dcm,
      )
      self.assertLen(img, 1)
      self.assertEqual(img[0].shape, expected_shape)

  def test_dicom_image_patch_coordinates_outside_of_image_raises(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: [
              {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
          ],
      }
      with self.assertRaises(
          data_accessor_errors.PatchOutsideOfImageDimensionsError
      ):
        _test_load_generic_dicom(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            dcm,
        )

  def test_is_accessor_data_embedded_in_request(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      json_instance = {
          _InstanceJsonKeys.DICOM_WEB_URI: _create_dicom_web_uri(dcm),
          _InstanceJsonKeys.PATCH_COORDINATES: [],
      }
      instance = data_accessor_definition.json_to_generic_dicom_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
          dicom_instances_metadata=[],
      )
      self.assertFalse(
          data_accessor.DicomGenericData(
              instance
          ).is_accessor_data_embedded_in_request()
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_patch_coordinates',
          metadata={},
          expected=1,
      ),
      dict(
          testcase_name='one_patch',
          metadata={
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
              ]
          },
          expected=1,
      ),
      dict(
          testcase_name='two_patches',
          metadata={
              _InstanceJsonKeys.PATCH_COORDINATES: [
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
                  {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10},
              ]
          },
          expected=2,
      ),
  )
  def test_accessor_length(self, metadata, expected):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        instance_path = _create_dicom_web_uri(dcm)
        json_instance = {
            _InstanceJsonKeys.DICOM_WEB_URI: instance_path,
        }
        json_instance.update(metadata)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory_module.NoAuthCredentialsFactory()
        )
        instance = data_accessor_definition.json_to_generic_dicom_image(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            default_patch_width=256,
            default_patch_height=256,
            require_patch_dim_match_default_dim=False,
            dicom_instances_metadata=dwi.get_instances(
                dicom_path.FromString(instance_path)
            ),
        )
        self.assertLen(data_accessor.DicomGenericData(instance), expected)

  @parameterized.parameters([0, 1, 2])
  def test_paralell_dicom_download(self, max_parallel_download_workers):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        instance_path = _create_dicom_web_uri(dcm)
        json_instance = {
            _InstanceJsonKeys.DICOM_SOURCE: [instance_path, instance_path],
        }
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory_module.NoAuthCredentialsFactory()
        )
        instance = data_accessor_definition.json_to_generic_dicom_image(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            default_patch_width=256,
            default_patch_height=256,
            require_patch_dim_match_default_dim=False,
            dicom_instances_metadata=dwi.get_instances(
                dicom_path.FromString(instance_path)
            ),
        )
        data_accessor_instance = data_accessor.DicomGenericData(
            instance,
            config=abstract_data_accessor.DataAccessorConfig(
                max_parallel_download_workers=max_parallel_download_workers,
            ),
        )
        with contextlib.ExitStack() as stack:
          data_accessor_instance.load_data(stack)
          data_accessor_instance.load_data(stack)
          self.assertLen(data_accessor_instance, 2)

  @parameterized.parameters(
      ['1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.1.99']
  )
  def test_can_decode_unencapsulated_transfer_syntax(self, uid):
    self.assertTrue(data_accessor._can_decode_transfer_syntax(uid))

  def test_raises_error_if_bad_dicom(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm:
      instance_path = _create_dicom_web_uri(dcm)

      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory_module.NoAuthCredentialsFactory()
        )
        metadata = dwi.get_instances(dicom_path.FromString(instance_path))
      with requests_mock.Mocker() as m:
        m.get(instance_path, content=b'1234')
        json_instance = {
            _InstanceJsonKeys.DICOM_SOURCE: [instance_path],
        }
        instance = data_accessor_definition.json_to_generic_dicom_image(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            default_patch_width=256,
            default_patch_height=256,
            require_patch_dim_match_default_dim=False,
            dicom_instances_metadata=metadata,
        )
        with self.assertRaises(data_accessor_errors.DicomError):
          list(data_accessor.DicomGenericData(instance).data_iterator())

  def test_download_multiple_dicom_instances_from_same_series(self):
    with pydicom.dcmread(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    ) as dcm1:
      dcm2 = copy.deepcopy(dcm1)
      dcm2.SOPInstanceUID = '1.2.3'
      dcm2.file_meta.MediaStorageSOPInstanceUID = '1.2.3'

      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm1)
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm2)

        instance_path1 = _create_dicom_web_uri(dcm1)
        instance_path2 = _create_dicom_web_uri(dcm2)

        json_instance = {
            _InstanceJsonKeys.DICOM_SOURCE: [instance_path1, instance_path2],
        }

        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory_module.NoAuthCredentialsFactory()
        )

        instance = data_accessor_definition.json_to_generic_dicom_image(
            credential_factory_module.NoAuthCredentialsFactory(),
            json_instance,
            default_patch_width=256,
            default_patch_height=256,
            require_patch_dim_match_default_dim=False,
            dicom_instances_metadata=dwi.get_instances(
                dicom_path.FromString(instance_path1).GetSeriesPath()
            ),
        )

        data_accessor_instance = data_accessor.DicomGenericData(instance)
        with contextlib.ExitStack() as stack:
          mock_download_series = stack.enter_context(
              mock.patch.object(
                  data_accessor,
                  '_download_dicom_series',
                  wraps=data_accessor._download_dicom_series,
              )
          )
          mock_download_instance = stack.enter_context(
              mock.patch.object(
                  data_accessor,
                  '_download_dicom_instance',
                  wraps=data_accessor._download_dicom_instance,
              )
          )
          data_accessor_instance.load_data(stack)
          self.assertLen(data_accessor_instance._local_file_paths, 2)
          mock_download_series.assert_called_once()
          mock_download_instance.assert_not_called()


if __name__ == '__main__':
  absltest.main()
