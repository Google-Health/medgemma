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
import io

from absl.testing import absltest
from absl.testing import parameterized

from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.utils import test_utils


class AbstractHandlerTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='sequence_of_strings',
          source=['a', 'b', 'c'],
      ),
      dict(testcase_name='iterator_of_strings', source=iter(['a', 'b', 'c'])),
  ])
  def test_input_file_iterator_strings(self, source):
    input_file_iterator = abstract_handler.InputFileIterator(source)
    test_iter = iter(input_file_iterator)
    self.assertEqual(next(test_iter), 'a')
    self.assertFalse(input_file_iterator.iteration_completed)
    self.assertEqual(next(test_iter), 'a')
    self.assertFalse(input_file_iterator.iteration_completed)
    input_file_iterator.processed_file()
    self.assertEqual(next(test_iter), 'b')
    self.assertFalse(input_file_iterator.iteration_completed)
    input_file_iterator.processed_file()
    self.assertEqual(next(test_iter), 'c')
    self.assertFalse(input_file_iterator.iteration_completed)
    self.assertEqual(next(test_iter), 'c')
    self.assertFalse(input_file_iterator.iteration_completed)
    self.assertEqual(next(test_iter), 'c')
    self.assertFalse(input_file_iterator.iteration_completed)
    input_file_iterator.processed_file()
    with self.assertRaises(StopIteration):
      next(test_iter)
    self.assertTrue(input_file_iterator.iteration_completed)
    with self.assertRaises(StopIteration):
      next(test_iter)
    self.assertTrue(input_file_iterator.iteration_completed)

  @parameterized.named_parameters([
      dict(
          testcase_name='sequence_of_iobytes',
          source=[io.BytesIO(b'1'), io.BytesIO(b'2'), io.BytesIO(b'3')],
      ),
      dict(
          testcase_name='iterator_of_iobytes',
          source=iter([io.BytesIO(b'1'), io.BytesIO(b'2'), io.BytesIO(b'3')]),
      ),
  ])
  def test_input_file_iterator_iobytes(self, source):
    input_file_iterator = abstract_handler.InputFileIterator(source)
    test_iter = iter(input_file_iterator)
    self.assertEqual(next(test_iter).read(), b'1')  # pytype: disable=attribute-error
    self.assertFalse(input_file_iterator.iteration_completed)
    self.assertEqual(next(test_iter).read(), b'1')  # pytype: disable=attribute-error
    self.assertFalse(input_file_iterator.iteration_completed)
    input_file_iterator.processed_file()
    self.assertEqual(next(test_iter).read(), b'2')  # pytype: disable=attribute-error
    self.assertFalse(input_file_iterator.iteration_completed)
    input_file_iterator.processed_file()
    self.assertEqual(next(test_iter).read(), b'3')  # pytype: disable=attribute-error
    self.assertFalse(input_file_iterator.iteration_completed)
    self.assertEqual(next(test_iter).read(), b'3')  # pytype: disable=attribute-error
    self.assertFalse(input_file_iterator.iteration_completed)
    self.assertEqual(next(test_iter).read(), b'3')  # pytype: disable=attribute-error
    self.assertFalse(input_file_iterator.iteration_completed)
    input_file_iterator.processed_file()
    with self.assertRaises(StopIteration):
      next(test_iter)
    self.assertTrue(input_file_iterator.iteration_completed)
    with self.assertRaises(StopIteration):
      next(test_iter)
    self.assertTrue(input_file_iterator.iteration_completed)

  def test_get_base_request_extensions_valid(self):
    base_request = {
        'extensions': {
            'image_dimensions': {'width': 100, 'height': 100},
            'require_patches_fully_in_source_image': False,
        }
    }
    self.assertEqual(
        abstract_handler.get_base_request_extensions(base_request),
        {
            'image_dimensions': {'width': 100, 'height': 100},
            'require_patches_fully_in_source_image': False,
        },
    )

  def test_get_base_request_extensions_raises(self):
    base_request = {'extensions': {1: 2}}
    with self.assertRaises(
        data_accessor_errors.InvalidRequestFieldError,
    ):
      abstract_handler.get_base_request_extensions(base_request)

  def test_process_files_with_handlers_raises_no_handler(self):
    test_dicom = test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    test_image = test_utils.testdata_path('image.jpeg')
    test_openslide = test_utils.testdata_path('openslide', 'ndpi_test.ndpi')
    process_files = abstract_handler.process_files_with_handlers(
        [
            generic_dicom_handler.GenericDicomHandler(),
            traditional_image_handler.TraditionalImageHandler(),
        ],
        [],
        {},
        [test_dicom, test_image, test_dicom, test_image, test_openslide],
    )
    with self.assertRaisesRegex(
        data_accessor_errors.DataAccessorError,
        'No file handler identified for input.',
    ):
      test_utils.flatten_data_acquisition(process_files)

  def test_process_files_with_handler_success(self):
    test_dicom = test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    test_image = test_utils.testdata_path('image.jpeg')
    self.assertLen(
        test_utils.flatten_data_acquisition(
            abstract_handler.process_files_with_handlers(
                [
                    generic_dicom_handler.GenericDicomHandler(),
                    traditional_image_handler.TraditionalImageHandler(),
                ],
                [],
                {},
                [test_dicom, test_image, test_dicom, test_image],
            )
        ),
        4,
    )

  def test_raises_if_no_handlers(self):
    test_dicom = test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm')
    test_image = test_utils.testdata_path('image.jpeg')
    with self.assertRaisesRegex(
        data_accessor_errors.InternalError, 'No configured file handers.*'
    ):
      test_utils.flatten_data_acquisition(
          abstract_handler.process_files_with_handlers(
              [],
              [],
              {},
              [test_dicom, test_image, test_dicom, test_image],
          )
      )


if __name__ == '__main__':
  absltest.main()
