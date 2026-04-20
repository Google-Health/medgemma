# Copyright 2023 Google LLC
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
# ==============================================================================
"""Tests for secret_flag_utils."""
import os
from typing import Any, List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import google.api_core
from google.cloud import secretmanager

from serving.logging_lib.flags import secret_flag_utils

_MOCK_SECRET = 'projects/test-project/secrets/test-secret'


def _create_mock_secret_value_response(
    data: Any,
) -> secretmanager.AccessSecretVersionResponse:
  payload = mock.create_autospec(secretmanager.SecretPayload, instance=True)
  payload.data = data
  sec_val = mock.create_autospec(
      secretmanager.AccessSecretVersionResponse, instance=True
  )
  sec_val.payload = payload
  return sec_val


def _create_mock_secret_version_response(
    secret_version_prefix: str,
    version_numbers: List[str],
) -> list[secretmanager.ListSecretVersionsResponse]:
  version_list = []
  for version_number in version_numbers:
    version = mock.create_autospec(
        secretmanager.ListSecretVersionsResponse, instance=True
    )
    version.name = f'{secret_version_prefix}{version_number}'
    version_list.append(version)
  return version_list


class SecretFlagUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    secret_flag_utils._cache.clear()
    self.enter_context(
        mock.patch(
            'google.auth.default', autospec=True, return_value=('mock', 'mock')
        )
    )

  def tearDown(self):
    super().tearDown()
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = False

  def test_init_fork_module_state(self):
    secret_flag_utils._cache = None
    secret_flag_utils._cache_lock = None
    secret_flag_utils._init_fork_module_state()
    self.assertIsNotNone(secret_flag_utils._cache)
    self.assertIsNotNone(secret_flag_utils._cache_lock)

  @parameterized.named_parameters(
      dict(
          testcase_name='expected',
          version_numbers=['1', '2', '3'],
          expected_version='3',
      ),
      dict(
          testcase_name='ignore_unexpected_version',
          version_numbers=['A', '1'],
          expected_version='1',
      ),
  )
  def test_get_secret_version(self, version_numbers, expected_version):
    client = mock.create_autospec(
        secretmanager.SecretManagerServiceClient, instance=True
    )
    client.list_secret_versions.return_value = (
        _create_mock_secret_version_response(
            f'{_MOCK_SECRET}/versions/', version_numbers
        )
    )
    version = secret_flag_utils._get_secret_version(client, _MOCK_SECRET)
    self.assertEqual(version, expected_version)

  def test_get_secret_version_cannot_find_version(self):
    client = mock.create_autospec(
        secretmanager.SecretManagerServiceClient, instance=True
    )
    client.list_secret_versions.return_value = (
        _create_mock_secret_version_response('', ['NO_MATCH', 'NO_MATCH'])
    )
    with self.assertRaises(secret_flag_utils.SecretDecodeError):
      secret_flag_utils._get_secret_version(client, _MOCK_SECRET)

  def test_get_secret_version_ignore_bad_response(self):
    client = mock.create_autospec(
        secretmanager.SecretManagerServiceClient, instance=True
    )
    client.list_secret_versions.return_value = []
    with self.assertRaises(secret_flag_utils.SecretDecodeError):
      secret_flag_utils._get_secret_version(client, _MOCK_SECRET)

  def test_read_secrets_path_to_version_empty(self):
    self.assertEqual(secret_flag_utils._read_secrets(''), {})

  def test_read_secrets_path_invalid_raises(self):
    with self.assertRaises(secret_flag_utils.SecretDecodeError):
      secret_flag_utils._read_secrets('no_match')

  def test_read_secrets_path_disabled_returns_empty_result(self):
    self.assertEqual(
        secret_flag_utils._read_secrets('projects/prj/secrets/sec'), {}
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='none',
          data=None,
          expected_value={},
      ),
      dict(
          testcase_name='empty',
          data='',
          expected_value={},
      ),
      dict(
          testcase_name='empty_dict',
          data='{}',
          expected_value={},
      ),
      dict(
          testcase_name='defined_dict_str',
          data='{"abc": "123"}',
          expected_value={'abc': '123'},
      ),
      dict(
          testcase_name='defined_dict_bytes',
          data='{"abc": "123"}'.encode('utf-8'),
          expected_value={'abc': '123'},
      ),
  ])
  @mock.patch.object(
      secret_flag_utils, '_get_secret_version', autospec=True, return_value='1'
  )
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_read_secrets_value(
      self, mk_access_secret, mk_get_secret_version, data, expected_value
  ):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec'
    secret_version = '1'
    mk_get_secret_version.return_value = secret_version
    mk_access_secret.return_value = _create_mock_secret_value_response(data)
    self.assertEqual(
        secret_flag_utils._read_secrets(project_secret),
        expected_value,
    )
    mk_get_secret_version.assert_called_once_with(mock.ANY, project_secret)
    mk_access_secret.assert_called_once_with(
        mock.ANY,
        request={'name': f'{project_secret}/versions/{secret_version}'},
    )

  @mock.patch.object(secret_flag_utils, '_get_secret_version', autospec=True)
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_read_secrets_value_does_not_call_version_if_in_path(
      self, mk_access_secret, mk_get_secret_version
  ):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/2'
    mk_access_secret.return_value = _create_mock_secret_value_response('')
    self.assertEqual(
        secret_flag_utils._read_secrets(project_secret),
        {},
    )
    mk_get_secret_version.assert_not_called()
    mk_access_secret.assert_called_once_with(
        mock.ANY, request={'name': project_secret}
    )

  def test_read_secrets_returns_cache_value(self):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/3'
    expected = 'EXPECTED_VALUE'
    secret_flag_utils._cache[project_secret] = expected
    self.assertEqual(secret_flag_utils._read_secrets(project_secret), expected)

  @parameterized.named_parameters([
      dict(testcase_name='invalid_json', response='{'),
      dict(testcase_name='not_dict', response='[]'),
  ])
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_read_secrets_returns_invalid_value_raises(
      self, mk_access_secret, response
  ):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/2'
    mk_access_secret.return_value = _create_mock_secret_value_response(response)
    with self.assertRaises(secret_flag_utils.SecretDecodeError):
      secret_flag_utils._read_secrets(project_secret)

  @parameterized.parameters([
      google.api_core.exceptions.NotFound('mock'),
      google.api_core.exceptions.PermissionDenied('mock'),
  ])
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_access_secret_version_raises(self, exp, mk_access_secret):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/2'
    mk_access_secret.side_effect = exp
    with self.assertRaises(secret_flag_utils.SecretDecodeError):
      secret_flag_utils._read_secrets(project_secret)

  @parameterized.parameters([
      google.api_core.exceptions.NotFound('mock'),
      google.api_core.exceptions.PermissionDenied('mock'),
  ])
  @mock.patch.object(secret_flag_utils, '_get_secret_version', autospec=True)
  def test_get_secret_version_raises(self, exp, mk_get_secret_version):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec'
    mk_get_secret_version.side_effect = exp
    with self.assertRaises(secret_flag_utils.SecretDecodeError):
      secret_flag_utils._read_secrets(project_secret)

  @parameterized.named_parameters([
      dict(
          testcase_name='defined_env',
          env_name='MOCK_ENV',
          expected_value='DEFINED_VALUE',
      ),
      dict(
          testcase_name='not_defined_env',
          env_name='UNDEFINED_ENV',
          expected_value='MOCK_DEFAULT',
      ),
  ])
  def test_get_secret_or_env_no_config_defined_returns_env_value_or_default(
      self, env_name, expected_value
  ):
    with mock.patch.dict(os.environ, {'MOCK_ENV': 'DEFINED_VALUE'}):
      self.assertEqual(
          secret_flag_utils.get_secret_or_env(env_name, 'MOCK_DEFAULT'),
          expected_value,
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='secret_env',
          search_env='SECRET_ENV',
          expected='SECRET_VALUE',
      ),
      dict(
          testcase_name='not_in_sec',
          search_env='NOT_IN_SEC',
          expected='ENV_VALUE',
      ),
      dict(
          testcase_name='not_in_env',
          search_env='NOT_IN_ENV',
          expected='MOCK_DEFAULT',
      ),
  ])
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_get_secret_or_env_with_config(
      self, mk_access_secret, search_env, expected
  ):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/1'
    mk_access_secret.return_value = _create_mock_secret_value_response(
        '{"SECRET_ENV": "SECRET_VALUE"}'
    )
    with mock.patch.dict(
        os.environ,
        {
            secret_flag_utils._SECRET_MANAGER_ENV_CONFIG: project_secret,
            'NOT_IN_SEC': 'ENV_VALUE',
        },
    ):
      self.assertEqual(
          secret_flag_utils.get_secret_or_env(search_env, 'MOCK_DEFAULT'),
          expected,
      )
      mk_access_secret.assert_called_once_with(
          mock.ANY,
          request={'name': project_secret},
      )

  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_get_secret_or_env_default_return_value(self, mk_access_secret):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/1'
    mk_access_secret.return_value = _create_mock_secret_value_response(
        '{"MOCK_ENV": "SECRET_VALUE"}'
    )
    with mock.patch.dict(
        os.environ,
        {
            secret_flag_utils._SECRET_MANAGER_ENV_CONFIG: project_secret,
        },
    ):
      self.assertIsNone(
          secret_flag_utils.get_secret_or_env('NOT_IN_SECRET_OR_ENV', None)
      )
      mk_access_secret.assert_called_once_with(
          mock.ANY,
          request={'name': project_secret},
      )

  @parameterized.named_parameters([
      dict(testcase_name='FOUND', env_name='MOCK_ENV', expected_value=True),
      dict(
          testcase_name='DEFAULT_MISSING',
          env_name='NOT_FOUND_ENV',
          expected_value=False,
      ),
  ])
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_get_bool_secret_or_env(
      self, mk_access_secret, env_name, expected_value
  ):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/1'
    mk_access_secret.return_value = _create_mock_secret_value_response(
        '{"MOCK_ENV": "TRUE"}'
    )
    with mock.patch.dict(
        os.environ,
        {
            secret_flag_utils._SECRET_MANAGER_ENV_CONFIG: project_secret,
        },
    ):
      self.assertEqual(
          secret_flag_utils.get_bool_secret_or_env(env_name), expected_value
      )
      mk_access_secret.assert_called_once_with(
          mock.ANY,
          request={'name': project_secret},
      )

  @parameterized.named_parameters([
      dict(testcase_name='FOUND', env_name='ENV_FOUND', expected_value=False),
      dict(
          testcase_name='DEFAULT_MISSING',
          env_name='NOT_FOUND_ENV',
          expected_value=True,
      ),
  ])
  @mock.patch.object(
      secretmanager.SecretManagerServiceClient,
      'access_secret_version',
      autospec=True,
  )
  def test_get_bool_secret_or_env_trys_env(
      self, mk_access_secret, env_name, expected_value
  ):
    secret_flag_utils._ENABLE_ENV_SECRET_MANAGER = True
    project_secret = 'projects/prj/secrets/sec/versions/1'
    mk_access_secret.return_value = _create_mock_secret_value_response('{}')
    with mock.patch.dict(
        os.environ,
        {
            secret_flag_utils._SECRET_MANAGER_ENV_CONFIG: project_secret,
            'ENV_FOUND': 'False',
        },
    ):
      self.assertEqual(
          secret_flag_utils.get_bool_secret_or_env(env_name, True),
          expected_value,
      )
      mk_access_secret.assert_called_once_with(
          mock.ANY,
          request={'name': project_secret},
      )


if __name__ == '__main__':
  absltest.main()
