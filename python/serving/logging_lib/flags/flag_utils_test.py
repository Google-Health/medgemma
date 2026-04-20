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
"""Tests for flag utils."""
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from serving.logging_lib.flags import flag_utils

# const
_UNDEFINED_ENV_VAR_NAME = 'UNDEFINED'


class FlagUtilsTest(parameterized.TestCase):

  @parameterized.parameters(['y', ' YES ', 't', 'tRue', 'on', '1'])
  def test_str_to_bool_true(self, val):
    self.assertTrue(flag_utils.str_to_bool(val))

  @parameterized.parameters(['n', 'no', 'f', 'FALSE', ' oFf ', '0'])
  def test_strtobool_false(self, val):
    self.assertFalse(flag_utils.str_to_bool(val))

  def test_str_to_bool_raises(self):
    with self.assertRaises(ValueError):
      flag_utils.str_to_bool('ABCD')

  def test_undefined_env_default_true(self):
    self.assertNotIn(_UNDEFINED_ENV_VAR_NAME, os.environ)
    self.assertTrue(flag_utils.env_value_to_bool(_UNDEFINED_ENV_VAR_NAME, True))

  def test_undefined_env_no_default(self):
    self.assertNotIn(_UNDEFINED_ENV_VAR_NAME, os.environ)
    self.assertFalse(flag_utils.env_value_to_bool(_UNDEFINED_ENV_VAR_NAME))

  @mock.patch.dict(os.environ, {'FOO': ' True '})
  def test_initialized_env(self):
    self.assertTrue(flag_utils.env_value_to_bool('FOO'))

  @mock.patch.dict(os.environ, {'BAD_VALUE': 'TrueABSCDEF'})
  def test_bad_initialized_env(self):
    with self.assertRaises(ValueError):
      flag_utils.env_value_to_bool('BAD_VALUE')


if __name__ == '__main__':
  absltest.main()
