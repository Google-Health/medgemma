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

"""Tests for dicom source utils."""


from absl.testing import absltest
from absl.testing import parameterized
from data_accessors import data_accessor_errors
from data_accessors.utils import data_accessor_definition_utils


class DataAccessorDefinitionUtilTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(testcase_name='empty', instance={}),
      dict(testcase_name='dicomweb_uri_empty', instance={'dicomweb_uri': ''}),
      dict(testcase_name='dicomweb_uri_not_str', instance={'dicomweb_uri': 12}),
      dict(testcase_name='dicomweb_uri_is_list', instance={'dicomweb_uri': []}),
      dict(
          testcase_name='dicom_source_empty_str', instance={'dicom_source': ''}
      ),
      dict(
          testcase_name='dicom_source_empty_list', instance={'dicom_source': []}
      ),
      dict(
          testcase_name='dicom_source_contains_empty_str',
          instance={'dicom_source': ['']},
      ),
      dict(
          testcase_name='dicom_source_contains_non_str',
          instance={'dicom_source': [1]},
      ),
      dict(testcase_name='dicom_source_non_str', instance={'dicom_source': 1}),
      dict(
          testcase_name='multiple_instances_in_series',
          instance={
              'dicom_source': [
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6',
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6/instances/7.8.9',
              ]
          },
      ),
      dict(
          testcase_name='study_path',
          instance={
              'dicom_source': [
                  'http://test.com/dicomweb/studies/1.2.3',
              ]
          },
      ),
      dict(
          testcase_name='multiple_series',
          instance={
              'dicom_source': [
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.7/instances/4.8.9',
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6/instances/7.8.9',
              ]
          },
      ),
  ])
  def test_invalid_dicom_source_raises_error(self, instance):
    with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
      data_accessor_definition_utils.parse_dicom_source(instance)

  @parameterized.named_parameters([
      dict(
          testcase_name='dicomweb_uri',
          instance={
              'dicomweb_uri': (
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6'
              )
          },
          expected=['http://test.com/dicomweb/studies/1.2.3/series/4.5.6'],
      ),
      dict(
          testcase_name='dicom_source_str',
          instance={
              'dicom_source': (
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6'
              )
          },
          expected=['http://test.com/dicomweb/studies/1.2.3/series/4.5.6'],
      ),
      dict(
          testcase_name='dicom_source_str_series_list',
          instance={
              'dicom_source': [
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6',
              ]
          },
          expected=[
              'http://test.com/dicomweb/studies/1.2.3/series/4.5.6',
          ],
      ),
      dict(
          testcase_name='dicom_source_str_instance_list',
          instance={
              'dicom_source': [
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6/instances/7.8.9',
                  'http://test.com/dicomweb/studies/1.2.3/series/4.5.6/instances/9.10',
              ]
          },
          expected=[
              'http://test.com/dicomweb/studies/1.2.3/series/4.5.6/instances/7.8.9',
              'http://test.com/dicomweb/studies/1.2.3/series/4.5.6/instances/9.10',
          ],
      ),
  ])
  def test_dicom_source_success(self, instance, expected):
    self.assertEqual(
        data_accessor_definition_utils.parse_dicom_source(instance), expected
    )


if __name__ == '__main__':
  absltest.main()
