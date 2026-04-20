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

from absl.testing import absltest
from absl.testing import parameterized

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.inline_text import data_accessor
from data_accessors.inline_text import data_accessor_definition


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class DataAccessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="test_empty_text",
          msg={},
          expected="",
      ),
      dict(
          testcase_name="test_text",
          msg={_InstanceJsonKeys.TEXT: "test_text"},
          expected="test_text",
      ),
  )
  def test_data_accessor_definition(self, msg, expected):
    result = data_accessor_definition.json_to_text(msg)
    self.assertEqual(result.text, expected)
    self.assertEqual(result.base_request, msg)

  def test_data_accessor_definition_raises_if_value_is_not_string(self):
    with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
      data_accessor_definition.json_to_text({_InstanceJsonKeys.TEXT: 1})

  def test_data_accessor(self):
    result = data_accessor.InlineText(
        data_accessor_definition.json_to_text(
            {_InstanceJsonKeys.TEXT: "test_text"}
        )
    )
    self.assertEqual(list(result.data_iterator()), ["test_text"])

  def test_is_accessor_data_embedded_in_request(self):
    json_instance = {
        _InstanceJsonKeys.TEXT: "test_text",
    }
    instance = data_accessor_definition.json_to_text(json_instance)
    local_data_accessor = data_accessor.InlineText(instance)
    self.assertTrue(local_data_accessor.is_accessor_data_embedded_in_request())

  def test_accessor_length(self):
    json_instance = {
        _InstanceJsonKeys.TEXT: "test_text",
    }
    instance = data_accessor_definition.json_to_text(json_instance)
    local_data_accessor = data_accessor.InlineText(instance)
    self.assertLen(local_data_accessor, 1)


if __name__ == "__main__":
  absltest.main()
