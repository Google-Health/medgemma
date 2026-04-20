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

"""Tests for authentication utils for  utils."""
from typing import Any, MutableMapping, Optional, Sequence, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
import google.auth.credentials

from data_accessors.utils import authentication_utils

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


class AuthenticationUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='application_default',
          bearer_token='application_default',
          expected_class=credential_factory.DefaultCredentialFactory,
      ),
      dict(
          testcase_name='token_passthrough',
          bearer_token='ABC',
          expected_class=credential_factory.TokenPassthroughCredentialFactory,
      ),
      dict(
          testcase_name='no_auth',
          bearer_token='',
          expected_class=credential_factory.NoAuthCredentialsFactory,
      ),
  )
  def test_bearer_token_to_credential_factory_instance(
      self, bearer_token, expected_class
  ):
    self.assertIsInstance(
        authentication_utils.create_auth_from_instance(bearer_token),
        expected_class,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='application_default',
          bearer_token='application_default',
          expected_token=_MOCK_TOKEN,
      ),
      dict(
          testcase_name='token_passthrough',
          bearer_token='ABC',
          expected_token='ABC',
      ),
      dict(
          testcase_name='no_auth',
          bearer_token='',
          expected_token=None,
      ),
  )
  @mock.patch('google.auth.default', side_effect=_get_mocked_credentials)
  def test_bearer_token_to_credential_token(
      self, _, bearer_token, expected_token
  ):
    auth = authentication_utils.create_auth_from_instance(bearer_token)
    self.assertEqual(auth.get_credentials().token, expected_token)


if __name__ == '__main__':
  absltest.main()
