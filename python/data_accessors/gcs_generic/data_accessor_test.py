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
"""Tests for gcs_generic data accessor."""

import contextlib
import os
import shutil
import tempfile
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory as credential_factory_module
import google.auth.credentials
import numpy as np
import PIL.Image

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.gcs_generic import data_accessor
from data_accessors.gcs_generic import data_accessor_definition
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.utils import authentication_utils
from data_accessors.utils import test_utils
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_MOCK_TOKEN = 'MOCK_AUTH_TOKEN'


def _mock_apply_credentials(
    headers: MutableMapping[Any, Any], token: Optional[str] = None
) -> None:
  headers['authorization'] = 'Bearer {}'.format(token or _MOCK_TOKEN)


def _get_mocked_credentials(
    scopes: Optional[Sequence[str]],
) -> Tuple[google.auth.credentials.Credentials, str]:
  del scopes  # unused
  credentials_mock = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  type(credentials_mock).token = mock.PropertyMock(return_value=_MOCK_TOKEN)
  type(credentials_mock).valid = mock.PropertyMock(return_value='True')
  type(credentials_mock).expired = mock.PropertyMock(return_value='False')
  credentials_mock.apply.side_effect = _mock_apply_credentials
  return credentials_mock, 'fake_project'


def _test_load_from_gcs(
    file_handlers,
    blob_file_name,
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    worker_count: int,
    source_image_path: str,
) -> Sequence[np.ndarray]:
  with tempfile.TemporaryDirectory() as temp_dir:
    shutil.copyfile(
        source_image_path,
        os.path.join(temp_dir, blob_file_name),
    )
    with gcs_mock.GcsMock({'earth': temp_dir}):
      instance = data_accessor_definition.json_to_generic_gcs_image(
          credential_factory,
          json_instance,
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )
      gcs_data_accessor = data_accessor.GcsGenericData(
          instance,
          file_handlers=file_handlers,
          download_worker_count=worker_count,
      )
      return list(gcs_data_accessor.data_iterator())


def _test_load_image_from_gcs(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    worker_count: int,
    source_image_path: str,
) -> Sequence[np.ndarray]:
  return _test_load_from_gcs(
      [
          generic_dicom_handler.GenericDicomHandler(),
          traditional_image_handler.TraditionalImageHandler(),
      ],
      'image.jpeg',
      credential_factory,
      json_instance,
      worker_count,
      source_image_path,
  )


def _test_load_generic_dicom_from_gcs(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    json_instance: Mapping[str, Any],
    worker_count: int,
    source_image_path: str,
) -> Sequence[np.ndarray]:
  return _test_load_from_gcs(
      [
          traditional_image_handler.TraditionalImageHandler(),
          generic_dicom_handler.GenericDicomHandler(),
      ],
      'image.dcm',
      credential_factory,
      json_instance,
      worker_count,
      source_image_path,
  )


class DataAccessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single_worker',
          worker_count=1,
      ),
      dict(
          testcase_name='multiple_workers',
          worker_count=2,
      ),
  )
  def test_gcs_traditional_image_handler_color_image(self, worker_count):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image.jpeg')
    img = _test_load_image_from_gcs(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        worker_count,
        source_image_path,
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  def test_gcs_traditional_image_handler_bw_image(self):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image_bw.jpeg')
    img = _test_load_image_from_gcs(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        1,
        source_image_path,
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 1))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0][..., 0], expected_img)

  def test_gcs_traditional_image_credential_pass_through(self):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image.jpeg')
    img = _test_load_image_from_gcs(
        credential_factory_module.TokenPassthroughCredentialFactory('token'),
        json_instance,
        1,
        source_image_path,
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  def test_gcs_traditional_image_default_credential(self):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.jpeg',
    }
    source_image_path = test_utils.testdata_path('image.jpeg')
    credentials_mock = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )
    type(credentials_mock).token = mock.PropertyMock(return_value='TOKEN')
    type(credentials_mock).valid = mock.PropertyMock(return_value='True')
    type(credentials_mock).expired = mock.PropertyMock(return_value='False')
    with mock.patch(
        'google.auth.default', return_value=(credentials_mock, 'project')
    ):
      img = _test_load_image_from_gcs(
          credential_factory_module.DefaultCredentialFactory(),
          json_instance,
          1,
          source_image_path,
      )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, (67, 100, 3))
    with PIL.Image.open(source_image_path) as source_img:
      expected_img = np.asarray(source_img)
    np.testing.assert_array_equal(img[0], expected_img)

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 256, 'height': 256}
          ],
      ),
  )
  def test_gcs_traditional_image_patch_coordinates_outside_of_image_raises(
      self, patch_coordinates
  ):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.jpeg',
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      _test_load_image_from_gcs(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          1,
          test_utils.testdata_path('image.jpeg'),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='patch_list',
          patch_coordinates=[
              {'x_origin': 0, 'y_origin': 0, 'width': 10, 'height': 10}
          ],
          expected_shape=(10, 10, 3),
      ),
      dict(
          testcase_name='empty_patch_list',
          patch_coordinates=[],
          expected_shape=(67, 100, 3),
      ),
  )
  def test_gcs_traditional_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.jpeg',
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    img = _test_load_image_from_gcs(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        1,
        test_utils.testdata_path('image.jpeg'),
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, expected_shape)

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
  def test_gcs_dicom_image_patch_coordinates(
      self, patch_coordinates, expected_shape
  ):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.dcm',
        _InstanceJsonKeys.PATCH_COORDINATES: patch_coordinates,
    }
    img = _test_load_generic_dicom_from_gcs(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        1,
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
    )
    self.assertLen(img, 1)
    self.assertEqual(img[0].shape, expected_shape)

  def test_gcs_dicom_image_patch_coordinates_outside_of_image_raises(self):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.dcm',
        _InstanceJsonKeys.PATCH_COORDINATES: [
            {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
        ],
    }
    with self.assertRaises(
        data_accessor_errors.PatchOutsideOfImageDimensionsError
    ):
      _test_load_generic_dicom_from_gcs(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          1,
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
      )

  def test_is_accessor_data_embedded_in_request(self):
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.dcm',
        _InstanceJsonKeys.PATCH_COORDINATES: [
            {'x_origin': 0, 'y_origin': 0, 'width': 5000, 'height': 10}
        ],
    }
    instance = data_accessor_definition.json_to_generic_gcs_image(
        credential_factory_module.NoAuthCredentialsFactory(),
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    gcs_data_accessor = data_accessor.GcsGenericData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
        download_worker_count=1,
    )
    self.assertFalse(gcs_data_accessor.is_accessor_data_embedded_in_request())

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
    json_instance = {
        _InstanceJsonKeys.GCS_URI: 'gs://earth/image.dcm',
    }
    json_instance.update(metadata)
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
        os.path.join(temp_dir, 'image.dcm'),
    )
    with gcs_mock.GcsMock({'earth': temp_dir}):
      instance = data_accessor_definition.json_to_generic_gcs_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )
      gcs_data_accessor = data_accessor.GcsGenericData(
          instance,
          file_handlers=[
              traditional_image_handler.TraditionalImageHandler(),
              generic_dicom_handler.GenericDicomHandler(),
          ],
          download_worker_count=1,
      )
      self.assertLen(gcs_data_accessor, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='application_default',
          access_credential={'access_credential': 'application_default'},
          expected_token=_MOCK_TOKEN,
      ),
      dict(
          testcase_name='token_passthrough',
          access_credential={'access_credential': 'ABC'},
          expected_token='ABC',
      ),
  )
  @mock.patch.object(data_accessor, '_download_to_file', autospec=True)
  @mock.patch('google.auth.default', side_effect=_get_mocked_credentials)
  @mock.patch('google.cloud.storage.Client', autospec=True)
  @mock.patch(
      'google.cloud.storage.Client.create_anonymous_client', autospec=True
  )
  def test_gcs_dicom_image_credentials(
      self,
      anonymous_client_mock,
      client_mock,
      *_,
      access_credential,
      expected_token
  ):
    json_instance = {_InstanceJsonKeys.GCS_URI: 'gs://earth/image.dcm'}
    json_instance.update(access_credential)
    auth = authentication_utils.create_auth_from_instance(
        json_instance.get('access_credential', '')
    )
    instance = data_accessor_definition.json_to_generic_gcs_image(
        auth,
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    gcs_data_accessor = data_accessor.GcsGenericData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
        download_worker_count=1,
    )
    with contextlib.ExitStack() as stack:
      gcs_data_accessor.load_data(stack)
      client_mock.assert_called_once()
      self.assertEqual(
          client_mock.call_args.kwargs['credentials'].token, expected_token
      )
      anonymous_client_mock.assert_not_called()

  @mock.patch.object(data_accessor, '_download_to_file', autospec=True)
  @mock.patch(
      'google.cloud.storage.Client.create_anonymous_client', autospec=True
  )
  def test_gcs_dicom_image_no_auth(self, anonymous_client_mock, *_):
    json_instance = {_InstanceJsonKeys.GCS_URI: 'gs://earth/image.dcm'}
    auth = authentication_utils.create_auth_from_instance(
        json_instance.get('access_credential', '')
    )
    instance = data_accessor_definition.json_to_generic_gcs_image(
        auth,
        json_instance,
        default_patch_width=256,
        default_patch_height=256,
        require_patch_dim_match_default_dim=False,
    )
    gcs_data_accessor = data_accessor.GcsGenericData(
        instance,
        file_handlers=[
            traditional_image_handler.TraditionalImageHandler(),
            generic_dicom_handler.GenericDicomHandler(),
        ],
        download_worker_count=1,
    )
    with contextlib.ExitStack() as stack:
      gcs_data_accessor.load_data(stack)
      anonymous_client_mock.assert_called_once()

  @parameterized.parameters([0, 1, 2])
  def test_download_multiple_files(self, max_parallel_download_workers):
    json_instance = {
        _InstanceJsonKeys.GCS_SOURCE: [
            'gs://earth/image.dcm',
            'gs://earth/image.dcm',
        ],
    }
    temp_dir = self.create_tempdir()
    shutil.copyfile(
        test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
        os.path.join(temp_dir, 'image.dcm'),
    )
    with gcs_mock.GcsMock({'earth': temp_dir}):
      instance = data_accessor_definition.json_to_generic_gcs_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          json_instance,
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )
      gcs_data_accessor = data_accessor.GcsGenericData(
          instance,
          file_handlers=[
              traditional_image_handler.TraditionalImageHandler(),
              generic_dicom_handler.GenericDicomHandler(),
          ],
          download_worker_count=1,
          config=abstract_data_accessor.DataAccessorConfig(
              max_parallel_download_workers=max_parallel_download_workers,
          ),
      )
      with contextlib.ExitStack() as stack:
        gcs_data_accessor.load_data(stack)
        gcs_data_accessor.load_data(stack)
        self.assertLen(gcs_data_accessor, 2)

  def test_missing_gcs_uri_raises(self):
    with self.assertRaisesRegex(
        data_accessor_errors.InvalidRequestFieldError, '.*GCS URI not defined.*'
    ):
      data_accessor_definition.json_to_generic_gcs_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          {},
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )

  def test_missing_gcs_source_empty_list_raises(self):
    with self.assertRaisesRegex(
        data_accessor_errors.InvalidRequestFieldError,
        '.*gcs_source is an empty list.*',
    ):
      data_accessor_definition.json_to_generic_gcs_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          {'gcs_source': []},
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )

  @parameterized.parameters([1, '', 'gs://foo'])
  def test_missing_gcs_source_contains_invalid_value_string_raises(self, value):
    with self.assertRaisesRegex(
        data_accessor_errors.InvalidRequestFieldError, '.*invalid GCS URI;.*'
    ):
      data_accessor_definition.json_to_generic_gcs_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          {'gcs_source': [value]},
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )

  @parameterized.parameters([0, 1, 2])
  def test_decode_dicom_and_traditional_images(
      self, max_parallel_download_workers
  ):
    temp_dir = self.create_tempdir()
    with gcs_mock.GcsMock({'earth': temp_dir}):
      instance = data_accessor_definition.json_to_generic_gcs_image(
          credential_factory_module.NoAuthCredentialsFactory(),
          {
              'gcs_source': [
                  'gs://earth/image.dcm',
                  'gs://earth/image.jpeg',
                  'gs://earth/image.dcm',
                  'gs://earth/image.jpeg',
              ]
          },
          default_patch_width=256,
          default_patch_height=256,
          require_patch_dim_match_default_dim=False,
      )
      gcs_data_accessor = data_accessor.GcsGenericData(
          instance,
          file_handlers=[
              traditional_image_handler.TraditionalImageHandler(),
              generic_dicom_handler.GenericDicomHandler(),
          ],
          download_worker_count=1,
          config=abstract_data_accessor.DataAccessorConfig(
              max_parallel_download_workers=max_parallel_download_workers,
          ),
      )
      shutil.copyfile(
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
          os.path.join(temp_dir, 'image.dcm'),
      )
      shutil.copyfile(
          test_utils.testdata_path('image.jpeg'),
          os.path.join(temp_dir, 'image.jpeg'),
      )
      with contextlib.ExitStack() as stack:
        gcs_data_accessor.load_data(stack)
        self.assertLen(gcs_data_accessor, 4)


if __name__ == '__main__':
  absltest.main()
