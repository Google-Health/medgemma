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

"""Tests for MedSigLIP predictor."""

import base64
import io
import json
import os
from typing import Any, Mapping
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import PIL.Image
import pydicom
import requests_mock

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.dicom_wsi import configuration as dicom_wsi_configuration
from data_accessors.utils import test_utils
from serving.serving_framework import model_runner
from serving import predictor
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


def _pc(
    x_origin: int, y_origin: int, width: int = 224, height: int = 224
) -> Mapping[str, int]:
  """Return dictionary defining of patch coordinate."""
  return {
      'x_origin': x_origin,
      'y_origin': y_origin,
      'width': width,
      'height': height,
  }


_MOCK_STORE_PATH = 'https://test_store'
_MOCK_MODEL_RUNNER = mock.create_autospec(
    model_runner.ModelRunner, instance=True
)
_MOCK_MODEL_RUNNER.run_model_multiple_output.return_value = {
    'text_output': np.array([b'test_output']),
    'num_input_tokens': np.array(42),
    'num_output_tokens': np.array(3),
}


def _mock_prompt_converter(
    conversation: list[dict[str, Any]], params: dict[str, Any]
) -> str:
  """Mock prompt converter."""
  del params
  return json.dumps(conversation)


def _read_test_path_dcm() -> pydicom.FileDataset:
  path = os.path.join(
      os.path.dirname(__file__),
      'testdata',
      'multiframe_camelyon_challenge_image.dcm',
  )
  return pydicom.dcmread(path)


def _read_test_cxr_dcm() -> pydicom.FileDataset:
  path = os.path.join(
      os.path.dirname(__file__), 'testdata', 'encapsulated_cxr.dcm'
  )
  return pydicom.dcmread(path)


def _read_test_jpeg() -> bytes:
  path = os.path.join(os.path.dirname(__file__), 'testdata', 'image.jpeg')
  with open(path, 'rb') as infile:
    return infile.read()


def _mock_base64_encoder(image_bytes: bytes) -> str:
  image_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
  avg = round(np.mean(image_bytes), 2)
  length = image_bytes.shape[0]
  return f'average_byte_value: {avg}, byte_length: {length}'


class DicomDigitalPathologyDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Mock base 64 encoder
    self.enter_context(
        mock.patch.object(
            predictor,
            '_base64_encode_image_bytes',
            autospec=True,
            side_effect=_mock_base64_encoder,
        )
    )
    self.enter_context(
        flagsaver.flagsaver(worker_download_parallelism='THREAD')
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='one_pixel',
          input_image_shape=(1, 1, 3),
          expected_image_input='average_byte_value: 255.0, byte_length: 3',
      ),
      dict(
          testcase_name='square',
          input_image_shape=(4, 4, 3),
          expected_image_input='average_byte_value: 255.0, byte_length: 48',
      ),
      dict(
          testcase_name='large_square',
          input_image_shape=(100, 100, 3),
          expected_image_input='average_byte_value: 255.0, byte_length: 30000',
      ),
      dict(
          testcase_name='tall',
          input_image_shape=(100, 1, 3),
          expected_image_input='average_byte_value: 2.55, byte_length: 30000',
      ),
      dict(
          testcase_name='wide',
          input_image_shape=(1, 50, 3),
          expected_image_input='average_byte_value: 5.1, byte_length: 7500',
      ),
  )
  # make image compression a pass through to improve testing transparency.
  @mock.patch.object(
      predictor, '_compress_image', side_effect=lambda x: x.tobytes()
  )
  def test_non_square_images_are_padded(
      self,
      _,
      input_image_shape: tuple[int, int, int],
      expected_image_input: bytes,
  ):

    mock_prediction_input = {
        'messages': [
            {
                'role': 'user',
                'content': [{
                    'type': 'image_gcs',
                    'image_gcs': {'gcs_uri': 'gs://earth/test.png'},
                }],
            },
        ],
        'max_tokens': 500,
        'temperature': 0,
    }
    temp_dir = self.create_tempdir().full_path
    non_square_image = np.full(input_image_shape, 255, dtype=np.uint8)
    with PIL.Image.fromarray(non_square_image) as img:
      img.save(os.path.join(temp_dir, 'test.png'))
    with gcs_mock.GcsMock({'earth': temp_dir}):
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      image_input = mock_model_runner.run_model_multiple_output.call_args[1][
          'model_input'
      ]['image'][0]
      self.assertEqual(image_input, expected_image_input)

  @parameterized.named_parameters(
      dict(
          testcase_name='png',
          image_input_compression_format='png',
          expected_image_input='average_byte_value: 61.83, byte_length: 69',
      ),
      dict(
          testcase_name='jpeg',
          image_input_compression_format='jpeg',
          expected_image_input='average_byte_value: 77.83, byte_length: 631',
      ),
      dict(
          testcase_name='jpg',
          image_input_compression_format='jpg',
          expected_image_input='average_byte_value: 77.83, byte_length: 631',
      ),
  )
  def test_image_compression(
      self,
      image_input_compression_format: str,
      expected_image_input: bytes,
  ):

    mock_prediction_input = {
        'messages': [
            {
                'role': 'user',
                'content': [{
                    'type': 'image_gcs',
                    'image_gcs': {'gcs_uri': 'gs://earth/test.png'},
                }],
            },
        ],
        'max_tokens': 500,
        'temperature': 0,
    }
    temp_dir = self.create_tempdir().full_path
    non_square_image = np.full((1, 1, 3), 255, dtype=np.uint8)
    with PIL.Image.fromarray(non_square_image) as img:
      img.save(os.path.join(temp_dir, 'test.png'))
    with gcs_mock.GcsMock({'earth': temp_dir}):
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      with flagsaver.flagsaver(
          image_input_compression_format=image_input_compression_format
      ):
        pred.predict(mock_prediction_input, mock_model_runner)
      image_input = mock_model_runner.run_model_multiple_output.call_args[1][
          'model_input'
      ]['image'][0]
      self.assertEqual(image_input, expected_image_input)

  @flagsaver.flagsaver(image_input_compression_format='invalid')
  def test_invalid_image_compression_format_raises_error(self):
    with self.assertRaises(data_accessor_errors.InternalError):
      predictor._compress_image(np.zeros((1, 1, 3), dtype=np.uint8))

  def test_path_dicom_prediction(self):
    dcm = _read_test_path_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    mock_prediction_input = {
        'messages': [
            {
                'role': 'system',
                'content': [{
                    'type': 'image_dicom',
                    'image_dicom': {
                        'dicomweb_uri': instance_path,
                        'patch_coordinates_list': [_pc(2, 2), _pc(3, 3)],
                    },
                }],
            },
        ],
        'max_tokens': 500,
        'temperature': 0,
    }
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
      # validate and clear fields that are not deterministic.
      self.assertIsInstance(result['created'], int)
      self.assertIsInstance(result['id'], str)
      result['created'] = 0
      result['id'] = ''
      self.assertEqual(
          result,
          {
              'choices': [{
                  'index': 0,
                  'message': {'content': 'test_output', 'role': 'assistant'},
              }],
              'created': 0,
              'id': '',
              'model': 'placeholder',
              'object': 'chat.completion',
              'usage': {
                  'prompt_tokens': 42,
                  'completion_tokens': 3,
                  'total_tokens': 45,
              },
          },
      )

  def test_http_image_prediction(self):
    mock_prediction_input = {
        'messages': [
            {
                'role': 'user',
                'content': [{
                    'type': 'image_url',
                    'image_url': {
                        'url': 'http://earth.com/image.jpeg',
                        'patch_coordinates_list': [_pc(0, 0, 10, 10)],
                    },
                }],
            },
        ],
        'max_tokens': 500,
        'temperature': 0,
    }
    with requests_mock.Mocker() as m:
      m.get('http://earth.com/image.jpeg', content=_read_test_jpeg())
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
      # validate and clear fields that are not deterministic.
      self.assertIsInstance(result['created'], int)
      self.assertIsInstance(result['id'], str)
      result['created'] = 0
      result['id'] = ''
      self.assertEqual(
          result,
          {
              'choices': [{
                  'index': 0,
                  'message': {'content': 'test_output', 'role': 'assistant'},
              }],
              'created': 0,
              'id': '',
              'model': 'placeholder',
              'object': 'chat.completion',
              'usage': {
                  'prompt_tokens': 42,
                  'completion_tokens': 3,
                  'total_tokens': 45,
              },
          },
      )

  def test_jpeg_image_prediction(self):
    base64_jpeg = base64.b64encode(_read_test_jpeg()).decode('utf-8')
    mock_prediction_input = {
        'max_tokens': 500,
        'temperature': 0,
        'messages': [{
            'role': 'user',
            'content': [
                {
                    'type': 'image_bytes',
                    'image_bytes': {
                        'input_bytes': base64_jpeg,
                    },
                },
                {
                    'type': 'image_bytes',
                    'image_bytes': {
                        'input_bytes': base64_jpeg,
                        'patch_coordinates_list': [
                            _pc(2, 2, 10, 10),
                            _pc(3, 3, 10, 10),
                        ],
                    },
                },
            ],
        }],
    }
    pred = predictor.MedGemmaPredictor(prompt_converter=_mock_prompt_converter)
    result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
    # validate and clear fields that are not deterministic.
    self.assertIsInstance(result['created'], int)
    self.assertIsInstance(result['id'], str)
    result['created'] = 0
    result['id'] = ''
    self.assertEqual(
        result,
        {
            'choices': [{
                'index': 0,
                'message': {'content': 'test_output', 'role': 'assistant'},
            }],
            'created': 0,
            'id': '',
            'model': 'placeholder',
            'object': 'chat.completion',
            'usage': {
                'prompt_tokens': 42,
                'completion_tokens': 3,
                'total_tokens': 45,
            },
        },
    )

  def test_cxr_dicom(self):
    dcm = _read_test_cxr_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    mock_prediction_input = {
        'max_tokens': 500,
        'temperature': 0,
        'messages': [
            {
                'role': 'user',
                'content': [{
                    'type': 'image_dicom',
                    'image_dicom': {
                        'dicomweb_uri': instance_path,
                        'patch_coordinates_list': [_pc(0, 0), _pc(1, 1)],
                    },
                }],
            },
            {
                'role': 'user',
                'content': [{
                    'type': 'image_dicom',
                    'image_dicom': {
                        'dicomweb_uri': instance_path,
                        'patch_coordinates_list': [_pc(2, 2), _pc(3, 3)],
                    },
                }],
            },
        ],
    }
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
      # validate and clear fields that are not deterministic.
      self.assertIsInstance(result['created'], int)
      self.assertIsInstance(result['id'], str)
      result['created'] = 0
      result['id'] = ''
      self.assertEqual(
          result,
          {
              'choices': [{
                  'index': 0,
                  'message': {'content': 'test_output', 'role': 'assistant'},
              }],
              'created': 0,
              'id': '',
              'model': 'placeholder',
              'object': 'chat.completion',
              'usage': {
                  'prompt_tokens': 42,
                  'completion_tokens': 3,
                  'total_tokens': 45,
              },
          },
      )

  def test_text_prediction(self):
    mock_prediction_input = {
        'max_tokens': 500,
        'temperature': 0,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'test_text_1'},
                {'type': 'text', 'text': 'test_text_2'},
            ],
        }],
    }
    pred = predictor.MedGemmaPredictor(prompt_converter=_mock_prompt_converter)
    result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
    # validate and clear fields that are not deterministic.
    self.assertIsInstance(result['created'], int)
    self.assertIsInstance(result['id'], str)
    result['created'] = 0
    result['id'] = ''
    self.assertEqual(
        result,
        {
            'choices': [{
                'index': 0,
                'message': {'content': 'test_output', 'role': 'assistant'},
            }],
            'created': 0,
            'id': '',
            'model': 'placeholder',
            'object': 'chat.completion',
            'usage': {
                'prompt_tokens': 42,
                'completion_tokens': 3,
                'total_tokens': 45,
            },
        },
    )

  def test_image_and_text_prediction(self):
    dcm = _read_test_path_dcm()
    instance_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
      mock_prediction_input = {
          'max_tokens': 500,
          'temperature': 0,
          'messages': [{
              'role': 'user',
              'content': [
                  {'type': 'text', 'text': 'test_text_1'},
                  {
                      'type': 'image_dicom',
                      'image_dicom': {
                          'dicomweb_uri': instance_path,
                          'patch_coordinates_list': [_pc(0, 0), _pc(1, 1)],
                      },
                  },
                  {'type': 'text', 'text': 'test_text_2'},
                  {
                      'type': 'image_dicom',
                      'image_dicom': {
                          'dicomweb_uri': instance_path,
                      },
                  },
              ],
          }],
      }
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
      # validate and clear fields that are not deterministic.
      self.assertIsInstance(result['created'], int)
      self.assertIsInstance(result['id'], str)
      result['created'] = 0
      result['id'] = ''
      self.assertEqual(
          result,
          {
              'choices': [{
                  'index': 0,
                  'message': {'content': 'test_output', 'role': 'assistant'},
              }],
              'created': 0,
              'id': '',
              'model': 'placeholder',
              'object': 'chat.completion',
              'usage': {
                  'prompt_tokens': 42,
                  'completion_tokens': 3,
                  'total_tokens': 45,
              },
          },
      )

  def test_data_accessor_error(self):
    mock_prediction_input = {
        'max_tokens': 500,
        'temperature': 0,
        'messages': [{
            'role': 'user',
            'content': [{
                'type': 'image_gcs',
                'image_gcs': {
                    'gcs_uri': 'gs://earth/test.dcm',
                },
            }],
        }],
    }
    with gcs_mock.GcsMock():
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
      result['error']['message'] = ''
      self.assertEqual(
          result,
          {'error': {'object': 'error', 'message': ''}},
      )

  def test_read_32bit_png_from_gcs(self):
    with io.BytesIO(_read_test_jpeg()) as f:
      with PIL.Image.open(f) as img:
        img_bytes = np.asarray(img)
      img_shape = list(img_bytes.shape)
      img_shape[-1] = 4
      png_bytes = np.zeros(img_shape, dtype=np.uint8)
      png_bytes[:, :, :3] = img_bytes[...]
      png_bytes[:, :, 3] = 125
    temp_dir = self.create_tempdir()
    temp_png_path = os.path.join(temp_dir.full_path, 'test.png')
    mock_prediction_input = {
        'max_tokens': 500,
        'temperature': 0,
        'messages': [{
            'role': 'user',
            'content': [{
                'type': 'image_gcs',
                'image_gcs': {
                    'gcs_uri': 'gs://earth/test.png',
                },
            }],
        }],
    }
    with PIL.Image.fromarray(png_bytes) as img:
      img.save(temp_png_path)
    with gcs_mock.GcsMock({'earth': temp_dir}):
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
      # validate and clear fields that are not deterministic.
      self.assertIsInstance(result['created'], int)
      self.assertIsInstance(result['id'], str)
      result['created'] = 0
      result['id'] = ''
      self.assertEqual(
          result,
          {
              'choices': [{
                  'index': 0,
                  'message': {'content': 'test_output', 'role': 'assistant'},
              }],
              'created': 0,
              'id': '',
              'model': 'placeholder',
              'object': 'chat.completion',
              'usage': {
                  'prompt_tokens': 42,
                  'completion_tokens': 3,
                  'total_tokens': 45,
              },
          },
      )

  def test_read_32bit_png_inline(self):
    with io.BytesIO(_read_test_jpeg()) as f:
      with PIL.Image.open(f) as img:
        img_bytes = np.asarray(img)
      img_shape = list(img_bytes.shape)
      img_shape[-1] = 4
      png_bytes = np.zeros(img_shape, dtype=np.uint8)
      png_bytes[:, :, :3] = img_bytes[...]
      png_bytes[:, :, 3] = 125
    with io.BytesIO() as outfile:
      with PIL.Image.fromarray(png_bytes) as img:
        img.save(outfile, format='PNG')
      png_bytes = outfile.getvalue()
    mock_prediction_input = {
        'max_tokens': 500,
        'temperature': 0,
        'messages': [{
            'role': 'user',
            'content': [{
                'type': 'image_bytes',
                'image_bytes': {
                    'input_bytes': base64.b64encode(png_bytes).decode('utf-8'),
                },
            }],
        }],
    }
    pred = predictor.MedGemmaPredictor(prompt_converter=_mock_prompt_converter)
    result = pred.predict(mock_prediction_input, _MOCK_MODEL_RUNNER)
    # validate and clear fields that are not deterministic.
    self.assertIsInstance(result['created'], int)
    self.assertIsInstance(result['id'], str)
    result['created'] = 0
    result['id'] = ''
    self.assertEqual(
        result,
        {
            'choices': [{
                'index': 0,
                'message': {'content': 'test_output', 'role': 'assistant'},
            }],
            'created': 0,
            'id': '',
            'model': 'placeholder',
            'object': 'chat.completion',
            'usage': {
                'prompt_tokens': 42,
                'completion_tokens': 3,
                'total_tokens': 45,
            },
        },
    )


class ImageEncoderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='expanded_2d',
          input_shape=(3, 3, 1),
      ),
      dict(
          testcase_name='compact_2d',
          input_shape=(3, 3),
      ),
  )
  @flagsaver.flagsaver(image_input_compression_format='png')
  def test_16bit_image_encoder_png(self, input_shape):
    input_image = np.zeros(input_shape, dtype=np.uint16)
    input_image[1, 1, ...] = 65535
    encoded_image = predictor._encode_image_bytes(input_image)
    image_bytes = base64.b64decode(encoded_image)
    with PIL.Image.open(io.BytesIO(image_bytes)) as img:
      decoded_image = np.asarray(img)
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[1, 1, :] = 255
    self.assertEqual(decoded_image.tolist(), expected.tolist())

  @flagsaver.flagsaver(image_input_compression_format='png')
  def test_8bit_image_encoder_png(self):
    input_image = np.zeros((3, 3, 1), dtype=np.uint8)
    input_image[1, 1, 0] = 255
    encoded_image = predictor._encode_image_bytes(input_image)
    image_bytes = base64.b64decode(encoded_image)
    with PIL.Image.open(io.BytesIO(image_bytes)) as img:
      decoded_image = np.asarray(img)
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[1, 1, :] = 255
    self.assertEqual(decoded_image.tolist(), expected.tolist())

  @flagsaver.flagsaver(image_input_compression_format='jpeg')
  def test_16bit_image_encoder_jpeg(self):
    input_image = np.zeros((3, 3, 1), dtype=np.uint16)
    input_image[1, 1, 0] = 65535
    encoded_image = predictor._encode_image_bytes(input_image)
    image_bytes = base64.b64decode(encoded_image)
    with PIL.Image.open(io.BytesIO(image_bytes)) as img:
      decoded_image = np.asarray(img)
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[0, 1, :] = 1
    expected[1, 1, :] = 254
    expected[1, 2, :] = 1
    expected[2, 2, :] = 3
    self.assertEqual(decoded_image.tolist(), expected.tolist())

  @flagsaver.flagsaver(image_input_compression_format='jpeg')
  def test_8bit_image_encoder_jpeg(self):
    input_image = np.zeros((3, 3, 1), dtype=np.uint8)
    input_image[1, 1, 0] = 255
    encoded_image = predictor._encode_image_bytes(input_image)
    image_bytes = base64.b64decode(encoded_image)
    with PIL.Image.open(io.BytesIO(image_bytes)) as img:
      decoded_image = np.asarray(img)
    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[0, 1, :] = 1
    expected[1, 1, :] = 254
    expected[1, 2, :] = 1
    expected[2, 2, :] = 3
    self.assertEqual(decoded_image.tolist(), expected.tolist())

  def test_get_local_file_handler_initalizes_once(self):
    self.assertIs(
        predictor._get_local_file_handlers(),
        predictor._get_local_file_handlers(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='ct',
          mock_modality='CT',
      ),
      dict(
          testcase_name='mri',
          mock_modality='MR',
      ),
  )
  def test_radiology_volume_images_with_more_than_one_slice_include_slice_prompt(
      self, mock_modality
  ):
    temp_dir = self.create_tempdir().full_path
    dcm_files = []
    for i in range(2):
      path = test_utils.testdata_path('ct', 'test_series', f'image{i}.dcm')
      filename = os.path.basename(path)
      with pydicom.dcmread(path) as dcm:
        dcm.Modality = mock_modality
        dcm.save_as(os.path.join(temp_dir, filename))
      dcm_files.append(f'gs://earth/{filename}')
    with gcs_mock.GcsMock({'earth': temp_dir}):
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      mock_prediction_input = {
          'messages': [
              {
                  'role': 'user',
                  'content': [
                      {
                          'type': 'text',
                          'text': 'Describe the following scan volume.',
                      },
                      {
                          'type': 'image_gcs',
                          'image_gcs': {'gcs_source': dcm_files},
                      },
                      {
                          'type': 'text',
                          'text': 'Be clear and concise.',
                      },
                  ],
              },
          ],
          'max_tokens': 500,
          'temperature': 0,
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      model_text_input = mock_model_runner.run_model_multiple_output.call_args[
          1
      ]['model_input']['text_input']
      self.assertEqual(
          model_text_input,
          [
              b'[{"role": "user", "content": [{"type": "text", "text":'
              b' "Describe the following scan volume."}, {"type": "image",'
              b' "image_gcs": {"gcs_source": ["gs://earth/image0.dcm",'
              b' "gs://earth/image1.dcm"]}}, {"type": "text", "text": "SLICE'
              b' 1"}, {"type": "image", "image_gcs": {"gcs_source":'
              b' ["gs://earth/image0.dcm", "gs://earth/image1.dcm"]}}, {"type":'
              b' "text", "text": "SLICE 2"}, {"type": "text", "text": "Be clear'
              b' and concise."}]}]'
          ],
      )
      model_input_images = (
          mock_model_runner.run_model_multiple_output.call_args[1][
              'model_input'
          ]['image']
      )
      self.assertLen(model_input_images, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name='ct',
          mock_modality='CT',
      ),
      dict(
          testcase_name='mri',
          mock_modality='MR',
      ),
  )
  def test_radiology_volume_images_define_single_excludes_index_prompt(
      self, mock_modality
  ):
    temp_dir = self.create_tempdir().full_path
    dcm_files = []
    path = test_utils.testdata_path('ct', 'test_series', f'image{0}.dcm')
    filename = os.path.basename(path)
    with pydicom.dcmread(path) as dcm:
      dcm.Modality = mock_modality
      dcm.save_as(os.path.join(temp_dir, filename))
    dcm_files.append(f'gs://earth/{filename}')
    with gcs_mock.GcsMock({'earth': temp_dir}):
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      mock_prediction_input = {
          'messages': [
              {
                  'role': 'user',
                  'content': [
                      {
                          'type': 'text',
                          'text': 'Describe the following scan volume.',
                      },
                      {
                          'type': 'image_gcs',
                          'image_gcs': {'gcs_source': dcm_files},
                      },
                      {
                          'type': 'text',
                          'text': 'Be clear and concise.',
                      },
                  ],
              },
          ],
          'max_tokens': 500,
          'temperature': 0,
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      model_text_input = mock_model_runner.run_model_multiple_output.call_args[
          1
      ]['model_input']['text_input']
      self.assertEqual(
          model_text_input,
          [
              b'[{"role": "user", "content": [{"type": "text", "text":'
              b' "Describe the following scan volume."}, {"type": "image",'
              b' "image_gcs": {"gcs_source": ["gs://earth/image0.dcm"]}},'
              b' {"type": "text", "text": "Be clear and concise."}]}]'
          ],
      )
      model_input_images = (
          mock_model_runner.run_model_multiple_output.call_args[1][
              'model_input'
          ]['image']
      )
      self.assertLen(model_input_images, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name='ct',
          mock_modality='CT',
      ),
      dict(
          testcase_name='mri',
          mock_modality='MR',
      ),
  )
  def test_radiology_series_volume_prediction_with_multiple_slices_includes_slice_prompt(
      self, mock_modality
  ):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      for index in range(2):
        path = test_utils.testdata_path(
            'ct', 'test_series', f'image{index}.dcm'
        )
        with pydicom.dcmread(path) as dcm:
          dcm.Modality = mock_modality
          dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
          series_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
      mock_prediction_input = {
          'messages': [{
              'role': 'system',
              'content': [
                  {
                      'type': 'text',
                      'text': 'Describe the following scan volume.',
                  },
                  {
                      'type': 'image_dicom',
                      'image_dicom': {
                          'dicom_source': series_path,
                      },
                  },
                  {
                      'type': 'text',
                      'text': 'Be clear and concise.',
                  },
              ],
          }],
          'max_tokens': 500,
          'temperature': 0,
      }
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      model_text_input = mock_model_runner.run_model_multiple_output.call_args[
          1
      ]['model_input']['text_input']
      self.assertEqual(
          model_text_input,
          [
              b'[{"role": "system", "content": [{"type": "text", "text":'
              b' "Describe the following scan volume."}, {"type": "image",'
              b' "image_dicom": {"dicom_source":'
              b' "https://test_store/studies/1.2.3.4.5.6/series/1.2.3.4.5.7"}},'
              b' {"type": "text", "text": "SLICE 1"}, {"type": "image",'
              b' "image_dicom": {"dicom_source":'
              b' "https://test_store/studies/1.2.3.4.5.6/series/1.2.3.4.5.7"}},'
              b' {"type": "text", "text": "SLICE 2"}, {"type": "text", "text":'
              b' "Be clear and concise."}]}]'
          ],
      )
      model_input_images = (
          mock_model_runner.run_model_multiple_output.call_args[1][
              'model_input'
          ]['image']
      )
      self.assertLen(model_input_images, 2)

  def test_dicom_pathology_prompt_via_dicom_series(
      self,
  ):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      path = test_utils.testdata_path(
          'wsi', 'multiframe_camelyon_challenge_image.dcm'
      )
      with pydicom.dcmread(path) as dcm:
        dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
        series_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
      mock_prediction_input = {
          'messages': [{
              'role': 'system',
              'content': [
                  {
                      'type': 'text',
                      'text': 'Describe the following pathology slide.',
                  },
                  {
                      'type': 'image_dicom',
                      'image_dicom': {
                          'dicom_source': series_path,
                      },
                  },
                  {
                      'type': 'text',
                      'text': 'Be clear and concise.',
                  },
              ],
          }],
          'max_tokens': 500,
          'temperature': 0,
      }
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      model_text_input = mock_model_runner.run_model_multiple_output.call_args[
          1
      ]['model_input']['text_input']
      model_input_images = (
          mock_model_runner.run_model_multiple_output.call_args[1][
              'model_input'
          ]['image']
      )
      self.assertEqual(
          model_text_input,
          [
              b'[{"role": "system", "content": [{"type": "text", "text":'
              b' "Describe the following pathology slide."}, {"type": "image",'
              b' "image_dicom": {"dicom_source":'
              b' "https://test_store/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634"}},'
              b' {"type": "text", "text": "Be clear and concise."}]}]'
          ],
      )
      self.assertLen(model_input_images, 1)

  def test_dicom_pathology_prompt_via_dicom_series_has_multiple_patches(
      self,
  ):
    with dicom_store_mock.MockDicomStores(_MOCK_STORE_PATH) as dicom_store:
      path = test_utils.testdata_path(
          'wsi', 'multiframe_camelyon_challenge_image.dcm'
      )
      with pydicom.dcmread(path) as dcm:
        dicom_store[_MOCK_STORE_PATH].add_instance(dcm)
        series_path = f'{_MOCK_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
      mock_prediction_input = {
          'messages': [{
              'role': 'system',
              'content': [
                  {
                      'type': 'text',
                      'text': 'Describe the following pathology slide.',
                  },
                  {
                      'type': 'image_dicom',
                      'image_dicom': {
                          'dicom_source': series_path,
                          'patch_coordinates_list': [_pc(2, 2), _pc(3, 3)],
                      },
                  },
                  {
                      'type': 'text',
                      'text': 'Be clear and concise.',
                  },
              ],
          }],
          'max_tokens': 500,
          'temperature': 0,
      }
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      model_text_input = mock_model_runner.run_model_multiple_output.call_args[
          1
      ]['model_input']['text_input']
      model_input_images = (
          mock_model_runner.run_model_multiple_output.call_args[1][
              'model_input'
          ]['image']
      )
      self.assertEqual(
          model_text_input,
          [
              b'[{"role": "system", "content": [{"type": "text", "text":'
              b' "Describe the following pathology slide."}, {"type": "image",'
              b' "image_dicom": {"dicom_source":'
              b' "https://test_store/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634",'
              b' "patch_coordinates_list": [{"x_origin": 2, "y_origin": 2,'
              b' "width": 224, "height": 224}, {"x_origin": 3, "y_origin": 3,'
              b' "width": 224, "height": 224}]}}, {"type": "image",'
              b' "image_dicom": {"dicom_source":'
              b' "https://test_store/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634",'
              b' "patch_coordinates_list": [{"x_origin": 2, "y_origin": 2,'
              b' "width": 224, "height": 224}, {"x_origin": 3, "y_origin": 3,'
              b' "width": 224, "height": 224}]}}, {"type": "text", "text": "Be'
              b' clear and concise."}]}]'
          ],
      )
      self.assertLen(model_input_images, 2)

  def test_http_image_prediction_legacy_image_syntax(self):
    mock_prediction_input = {
        'messages': [
            {
                'role': 'user',
                'content': [{
                    'type': 'image',
                    'image': 'http://earth.com/image.jpeg',
                    'patch_coordinates_list': [_pc(0, 0, 10, 10)],
                }],
            },
        ],
        'max_tokens': 500,
        'temperature': 0,
    }
    with requests_mock.Mocker() as m:
      m.get('http://earth.com/image.jpeg', content=_read_test_jpeg())
      pred = predictor.MedGemmaPredictor(
          prompt_converter=_mock_prompt_converter
      )
      mock_model_runner = mock.create_autospec(
          model_runner.ModelRunner, instance=True
      )
      mock_model_runner.run_model_multiple_output.return_value = {
          'text_output': np.array([b'test_output']),
          'num_input_tokens': np.array(42),
          'num_output_tokens': np.array(3),
      }
      pred.predict(mock_prediction_input, mock_model_runner)
      model_text_input = mock_model_runner.run_model_multiple_output.call_args[
          1
      ]['model_input']['text_input']
      model_input_images = (
          mock_model_runner.run_model_multiple_output.call_args[1][
              'model_input'
          ]['image']
      )
      self.assertEqual(
          model_text_input,
          [
              b'[{"role": "user", "content": [{"type": "image", "image":'
              b' "http://earth.com/image.jpeg", "patch_coordinates_list":'
              b' [{"x_origin": 0, "y_origin": 0, "width": 10, "height":'
              b' 10}]}]}]'
          ],
      )
      self.assertLen(model_input_images, 1)

  @parameterized.named_parameters([
      dict(
          testcase_name='content_is_not_a_dict',
          content=[],
          msg='Request image content is not a dict.*',
      ),
      dict(
          testcase_name='missing_type',
          content={},
          msg='Request image content does not define input "type" attribute.*',
      ),
      dict(
          testcase_name='missing_image_bytes',
          content={'type': 'image_bytes'},
          msg='Request image content does not define "image_bytes" record.*',
      ),
      dict(
          testcase_name='missing_image_url',
          content={'type': 'image_url'},
          msg='Request image content does not define "image_url" record.*',
      ),
      dict(
          testcase_name='missing_image_gcs',
          content={'type': 'image_gcs'},
          msg='Request image content does not define "image_gcs" record.*',
      ),
      dict(
          testcase_name='missing_image_dicom',
          content={'type': 'image_dicom'},
          msg='Request image content does not define "image_dicom" record.*',
      ),
      dict(
          testcase_name='unsupported_image_type',
          content={'type': 'unknown'},
          msg='Request image content defines unsupported image type.*',
      ),
  ])
  def test_parse_invalid_image_content_raises(self, content, msg):
    config = mock.create_autospec(
        dicom_wsi_configuration.ConfigurationSettings, instance=True
    )
    with self.assertRaisesRegex(
        data_accessor_errors.InvalidRequestFieldError, msg
    ):
      predictor._parse_image_content(config, content)

  @parameterized.named_parameters([
      dict(
          testcase_name='thread',
          method='THREAD',
          expected=abstract_data_accessor.AccessorWorkerParallelismMethod.THREAD,
      ),
      dict(
          testcase_name='process',
          method='PROCESS',
          expected=abstract_data_accessor.AccessorWorkerParallelismMethod.PROCESS,
      ),
  ])
  def test_get_worker_parallelism_method(self, method, expected):
    with flagsaver.flagsaver(worker_download_parallelism=method):
      self.assertEqual(predictor._get_worker_parallelism_method(), expected)

  def test_get_worker_parallelism_method_unsupported_type_raises_error(self):
    with flagsaver.flagsaver(worker_download_parallelism='FOO'):
      with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
        predictor._get_worker_parallelism_method()


if __name__ == '__main__':
  absltest.main()
