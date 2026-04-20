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

"""Callable responsible for running Inference on provided patches."""

from __future__ import annotations

import base64
import concurrent.futures
import contextlib
import copy
import dataclasses
import functools
import time
import typing
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import uuid

import cv2
import jsonschema
import numpy as np

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.dicom_generic import data_accessor as dicom_generic_data_accessor
from data_accessors.dicom_generic import data_accessor_definition as dicom_generic_data_accessor_definition
from data_accessors.dicom_wsi import configuration as dicom_wsi_configuration
from data_accessors.dicom_wsi import data_accessor as dicom_wsi_data_accessor
from data_accessors.dicom_wsi import data_accessor_definition as dicom_wsi_data_accessor_definition
from data_accessors.gcs_generic import data_accessor as gcs_generic_data_accessor
from data_accessors.gcs_generic import data_accessor_definition as gcs_generic_data_accessor_definition
from data_accessors.http_image import data_accessor as http_image_data_accessor
from data_accessors.http_image import data_accessor_definition as http_image_data_accessor_definition
from data_accessors.inline_bytes import data_accessor as inline_bytes_data_accessor
from data_accessors.inline_bytes import data_accessor_definition as inline_bytes_data_accessor_definition
from data_accessors.inline_text import data_accessor as inline_text_data_accessor
from data_accessors.inline_text import data_accessor_definition as inline_text_data_accessor_definition
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.local_file_handlers import openslide_handler
from data_accessors.local_file_handlers import traditional_image_handler
from data_accessors.local_file_handlers import wsi_dicom_handler
from data_accessors.utils import authentication_utils
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import json_validation_utils
from serving.serving_framework import model_runner
from serving import flags
from serving import predictor_const
from serving.logging_lib import cloud_logging_client


INSTANCES_KEY = 'instances'
PREDICTIONS_KEY = 'predictions'

# Number of images at which it is faster to encode PNG images in parallel using
# a process pool instead of serially on single cpu.
_PNG_PROCESS_POOL_THRESHOLD = 20

_DICOM_CT_OR_MRI_VOLUME_DATA_SOURCES = (
    abstract_data_accessor.AccessorDataSource.DICOM_CT_VOLUME,
    abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
)
_MESSAGE_CONTENT_ENTRY_TYPE_REMAP = {
    img_type: 'image' for img_type in predictor_const.IMAGE_INPUT_TYPES
} | {
    predictor_const.TEXT_INPUT_TYPE: 'text',
}


class _InlineTextInstance(inline_text_data_accessor_definition.InlineText):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # text does not have patch coordinates but adding to enable text accessor
    # to be used identically with image accessors.
    self.patch_coordinates = []


_EmbeddingInstance = Union[
    dicom_wsi_data_accessor_definition.DicomWSIImage,
    dicom_generic_data_accessor_definition.DicomGenericImage,
    gcs_generic_data_accessor_definition.GcsGenericBlob,
    inline_bytes_data_accessor_definition.InlineBytes,
    _InlineTextInstance,
    http_image_data_accessor_definition.HttpImage,
]

_local_file_handlers: Optional[Sequence[abstract_handler.AbstractHandler]] = (
    None
)


def _get_local_file_handlers() -> Sequence[abstract_handler.AbstractHandler]:
  """Returns local file handlers."""
  global _local_file_handlers
  if _local_file_handlers is None:
    _local_file_handlers = [
        generic_dicom_handler.GenericDicomHandler(),
        traditional_image_handler.TraditionalImageHandler(),
        openslide_handler.OpenSlideHandler(
            openslide_handler.EndpointInputDimensions(
                width_px=flags.MODEL_INPUT_WIDTH_FLAG.value,
                height_px=flags.MODEL_INPUT_HEIGHT_FLAG.value,
            )
        ),
        wsi_dicom_handler.WsiDicomHandler(),
    ]
  return _local_file_handlers


# endpoint configurations.
_GCS_DOWNLOAD_THREAD_COUNT = int(2)
_MAX_ERROR_DESCRIPTION_LENGTH = 1024

# Default value for max_tokens.
_DEFAULT_MAX_TOKENS = 500


def _cast_and_validate_bool(val: Any) -> bool:
  """Converts a value to bool or returns val as is."""
  if isinstance(val, bool):
    return val
  if isinstance(val, int) or isinstance(val, float):
    return bool(val)
  if not isinstance(val, str):
    raise json_validation_utils.ValidationError('not bool')
  val = val.strip().lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  if val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  raise json_validation_utils.ValidationError('not bool')


def _get_worker_parallelism_method() -> (
    abstract_data_accessor.AccessorWorkerParallelismMethod
):
  """Returns worker type."""
  match flags.WORKER_DOWNLOAD_PARALLELISM.value:
    case 'THREAD':
      return abstract_data_accessor.AccessorWorkerParallelismMethod.THREAD
    case 'PROCESS':
      return abstract_data_accessor.AccessorWorkerParallelismMethod.PROCESS
    case _:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Unsupported worker type.'
      )


def _parse_image_content(
    config: dicom_wsi_configuration.ConfigurationSettings,
    instance: Mapping[str, Any],
) -> abstract_data_accessor.AbstractDataAccessor[
    _EmbeddingInstance, np.ndarray
]:
  """Parses image instance request from json."""
  if not isinstance(instance, dict):
    raise data_accessor_errors.InvalidRequestFieldError(
        'Request image content is not a dict.'
    )
  try:
    input_type = instance[predictor_const.INPUT_TYPE]
  except KeyError as e:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Request image content does not define input "type" attribute.'
    ) from e
  match (input_type):
    case predictor_const.IMAGE_TYPE_BYTES:
      try:
        image_bytes = instance[predictor_const.IMAGE_TYPE_BYTES]
      except KeyError as e:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Request image content does not define "image_bytes" record.'
        ) from e
      parsed_instance = (
          inline_bytes_data_accessor_definition.json_to_generic_bytes(
              image_bytes,
              config.endpoint_input_width,
              config.endpoint_input_height,
              False,  # Require patch dim match default dim.
          )
      )
      return inline_bytes_data_accessor.InlineBytesData(
          parsed_instance, _get_local_file_handlers()
      )
    # support HTTP data source.
    case predictor_const.IMAGE_TYPE_URL:
      try:
        http_record = instance[predictor_const.IMAGE_TYPE_URL]
      except KeyError as e:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Request image content does not define "image_url" record.'
        ) from e
      parsed_instance = http_image_data_accessor_definition.json_to_http_image(
          authentication_utils.create_auth_from_instance(
              http_record.get(predictor_const.BEARER_TOKEN, '')
          ),
          http_record,
          config.endpoint_input_width,
          config.endpoint_input_height,
          False,  # Require patch dim match default dim.
      )
      return http_image_data_accessor.HttpImageData(
          parsed_instance,
          _get_local_file_handlers(),
          config=abstract_data_accessor.DataAccessorConfig(
              max_parallel_download_workers=config.max_parallel_download_workers,
              worker_parallelism_method=_get_worker_parallelism_method(),
          ),
      )
    # support MedGemma internal chat syntax.
    case predictor_const.IMAGE_TYPE_MEDGEMMA_INTERNAL:
      parsed_instance = http_image_data_accessor_definition.json_to_http_image(
          authentication_utils.create_auth_from_instance(
              instance.get(predictor_const.BEARER_TOKEN, '')
          ),
          instance,
          config.endpoint_input_width,
          config.endpoint_input_height,
          False,  # Require patch dim match default dim.
      )
      return http_image_data_accessor.HttpImageData(
          parsed_instance,
          _get_local_file_handlers(),
          config=abstract_data_accessor.DataAccessorConfig(
              max_parallel_download_workers=config.max_parallel_download_workers,
          ),
      )
    # support GCS.
    case predictor_const.IMAGE_TYPE_GCS:
      try:
        gcs_record = instance[predictor_const.IMAGE_TYPE_GCS]
      except KeyError as e:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Request image content does not define "image_gcs" record.'
        ) from e
      parsed_instance = (
          gcs_generic_data_accessor_definition.json_to_generic_gcs_image(
              authentication_utils.create_auth_from_instance(
                  gcs_record.get(predictor_const.BEARER_TOKEN, '')
              ),
              gcs_record,
              config.endpoint_input_width,
              config.endpoint_input_height,
              False,  # Require patch dim match default dim.
          )
      )
      return gcs_generic_data_accessor.GcsGenericData(
          parsed_instance,
          _get_local_file_handlers(),
          _GCS_DOWNLOAD_THREAD_COUNT,
          config=abstract_data_accessor.DataAccessorConfig(
              max_parallel_download_workers=config.max_parallel_download_workers,
              worker_parallelism_method=_get_worker_parallelism_method(),
          ),
      )
    # support DICOM.
    case predictor_const.IMAGE_TYPE_DICOM:
      # decode dicom path
      # determine dicom source type may query dicom store for series
      # instance metadata.
      try:
        dicom_record = instance[predictor_const.IMAGE_TYPE_DICOM]
      except KeyError as e:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Request image content does not define "image_dicom" record.'
        ) from e
      auth = authentication_utils.create_auth_from_instance(
          dicom_record.get(predictor_const.BEARER_TOKEN, '')
      )
      result = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          auth, dicom_record
      )
      # if slide microscope image
      if (
          result.dicom_source_type
          == dicom_source_utils.DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE
      ):
        # Define pathology DICOM input.
        parsed_instance = (
            dicom_wsi_data_accessor_definition.json_to_dicom_wsi_image(
                auth,
                dicom_record,
                config,
                result.dicom_instances_metadata,
            )
        )
        return dicom_wsi_data_accessor.DicomDigitalPathologyData(
            parsed_instance, config
        )
      parsed_instance = (
          dicom_generic_data_accessor_definition.json_to_generic_dicom_image(
              auth,
              dicom_record,
              config.endpoint_input_width,
              config.endpoint_input_height,
              False,  # Require patch dim match default dim.
              result.dicom_instances_metadata,
          )
      )
      return dicom_generic_data_accessor.DicomGenericData(
          parsed_instance,
          config=abstract_data_accessor.DataAccessorConfig(
              max_parallel_download_workers=config.max_parallel_download_workers,
              worker_parallelism_method=_get_worker_parallelism_method(),
          ),
      )
    case _:
      raise data_accessor_errors.InvalidRequestFieldError(
          'Request image content defines unsupported image type.'
      )


def _base64_encode_image_bytes(image_bytes: bytes) -> bytes:
  """Mockable target for testing."""
  return base64.b64encode(image_bytes)


def _zero_pad_image_to_square(norm_img: np.ndarray) -> np.ndarray:
  """Pads image with zeros to be dimensionally square."""
  height, width = norm_img.shape[:2]
  if height < width:
    dh = width - height
    half_dh = dh // 2
    return np.pad(norm_img, ((half_dh, dh - half_dh), (0, 0), (0, 0)))
  elif width < height:
    dw = height - width
    half_dw = dw // 2
    return np.pad(norm_img, ((0, 0), (half_dw, dw - half_dw), (0, 0)))
  return norm_img


def _compress_image(image_bytes: np.ndarray) -> bytes:
  """Convert image bytes to compressed image and return compressed bytes."""
  compression_format = flags.IMAGE_INPUT_COMPRESSION_FORMAT_FLAG.value.lower()
  if 'jpeg' in compression_format or 'jpg' in compression_format:
    quality = max(
        1, min(100, flags.IMAGE_INPUT_JPEG_COMPRESSION_QUALITY_FLAG.value)
    )
    return cv2.imencode(
        '.jpeg', image_bytes, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )[1].tobytes()
  if 'png' in compression_format:
    return cv2.imencode('.png', image_bytes)[1].tobytes()
  raise data_accessor_errors.InternalError(
      f'Unsupported compression flag: {compression_format}'
  )


def _encode_image_bytes(image_bytes: np.ndarray) -> bytes:
  """Encodes uncompressed image bytes as a base64 bytes."""
  # convert to 8 bit image
  if image_bytes.dtype != np.uint8:
    sf = np.iinfo(image_bytes.dtype).max / np.iinfo(np.uint8).max
    image_bytes = image_bytes.astype(np.float64) / sf
    image_bytes = np.round(image_bytes, 0).astype(np.uint8)
  if image_bytes.shape[-1] == 1 and image_bytes.ndim == 3:
    # if single channel monochrome image, represent as 3 channel image.
    image_bytes = np.concatenate(
        [image_bytes, image_bytes, image_bytes], axis=-1
    )
  elif image_bytes.shape[-1] == 4 and image_bytes.ndim == 3:
    # if RGBA image remove alpha channel.
    image_bytes = image_bytes[..., :3]
  elif image_bytes.ndim == 2:
    image_bytes = np.stack([image_bytes, image_bytes, image_bytes], axis=-1)
  image_shape = image_bytes.shape[:2]
  if flags.IMAGE_SIZE_OPTIMIZEATION_FLAG.value and (
      image_shape[0] > flags.MODEL_INPUT_HEIGHT_FLAG.value
      or image_shape[1] > flags.MODEL_INPUT_WIDTH_FLAG.value
  ):
    # Optional optimization if image size does exceed model input size then
    # shrink dim of the image to match the model encoder input size. The
    # optimization reduces the size of the image sent to the model encoder.
    # For imaging that exceeds the models input size, the optimization will
    # reduce memory and compute costs associated with execution.
    width = min(image_shape[1], flags.MODEL_INPUT_WIDTH_FLAG.value)
    height = min(image_shape[0], flags.MODEL_INPUT_HEIGHT_FLAG.value)
    # Interarea interpolation is done for decimation. Algorithm is fast
    # and resistant to high freq noise.
    image_bytes = cv2.resize(
        image_bytes, (width, height), interpolation=cv2.INTER_AREA
    )

  # pad image with zeros to be dimensionally square
  # method performed in training
  image_bytes = _zero_pad_image_to_square(image_bytes)

  # opencv assumes BGR ordering.
  image_bytes = cv2.cvtColor(image_bytes, cv2.COLOR_RGB2BGR)
  image_bytes = _compress_image(image_bytes)
  return _base64_encode_image_bytes(image_bytes)


@dataclasses.dataclass(frozen=True)
class _MedGemmaContent:
  """MedGemma content."""

  input_type: str
  content: abstract_data_accessor.AbstractDataAccessor[
      _EmbeddingInstance, Union[np.ndarray, str]
  ]

  def __post_init__(self):
    if self.content is None:
      raise ValueError('MedGemmaContent content is None.')


def _parse_text_content(
    content: Mapping[str, Any],
) -> abstract_data_accessor.AbstractDataAccessor[_InlineTextInstance, str]:
  inline_text = inline_text_data_accessor_definition.json_to_text(content)
  parsed_instance = _InlineTextInstance(
      text=inline_text.text, base_request=inline_text.base_request
  )
  return typing.cast(
      abstract_data_accessor.AbstractDataAccessor[_InlineTextInstance, str],
      inline_text_data_accessor.InlineText(parsed_instance),
  )


def _parse_content(
    config: dicom_wsi_configuration.ConfigurationSettings,
    content_list: Sequence[Mapping[str, Any]],
) -> Sequence[_MedGemmaContent]:
  """Parses content from json and sets up data accessors."""
  parsed_content = []
  for content in content_list:
    json_validation_utils.validate_str_key_dict(content)
    input_type = json_validation_utils.validate_str(
        content.get(predictor_const.TYPE)
    )
    match input_type:
      case input_type if input_type in predictor_const.IMAGE_INPUT_TYPES:
        parsed_content.append(
            _MedGemmaContent(input_type, _parse_image_content(config, content))
        )
      case predictor_const.TEXT_INPUT_TYPE:
        pass  # no handling needed.
      case _:
        raise data_accessor_errors.InvalidRequestFieldError(
            f'Invalid content input type: {input_type}'
        )
  return parsed_content


def _parse_all_content(
    config: dicom_wsi_configuration.ConfigurationSettings,
    json_messages: Sequence[Mapping[str, Any]],
) -> Sequence[_MedGemmaContent]:
  """Parses medgemma messages from json."""
  contents = []
  for message in json_messages:
    json_validation_utils.validate_str_key_dict(message)
    if isinstance(message.get(predictor_const.CONTENT), str):
      continue
    content = json_validation_utils.validate_not_empty_list(
        message.get(predictor_const.CONTENT)
    )
    contents.extend(_parse_content(config, content))
  return contents


@dataclasses.dataclass(frozen=True)
class _MedGemmaPredictionParameters:
  """Model request parameters."""

  max_tokens: Optional[int] = None
  temperature: Optional[int] = None
  frequency_penalty: Optional[float] = None
  n: Optional[int] = None
  presence_penalty: Optional[float] = None
  seed: Optional[int] = None
  stop: Optional[str] = None
  top_p: Optional[float] = None
  best_of: Optional[int] = None
  top_k: Optional[int] = None
  min_p: Optional[float] = None
  repetition_penalty: Optional[float] = None
  include_stop_str_in_output: Optional[bool] = None
  ignore_eos: Optional[bool] = None
  min_tokens: Optional[int] = None
  skip_special_tokens: Optional[bool] = None
  truncate_prompt_tokens: Optional[bool] = None
  spaces_between_special_tokens: Optional[bool] = None

  @classmethod
  def from_json(
      cls, json_parameters: Mapping[str, Any]
  ) -> '_MedGemmaPredictionParameters':
    """Extracts parameters from json and incorporates defaults."""
    name_map = {
        'max_tokens': 'max_tokens',
        'max_completion_tokens': 'max_tokens',
        'temperature': 'temperature',
        'frequency_penalty': 'frequency_penalty',
        'n': 'n',
        'presence_penalty': 'presence_penalty',
        'seed': 'seed',
        'stop': 'stop',
        'top_p': 'top_p',
        'best_of': 'best_of',
        'top_k': 'top_k',
        'min_p': 'min_p',
        'repetition_penalty': 'repetition_penalty',
        'include_stop_str_in_output': 'include_stop_str_in_output',
        'ignore_eos': 'ignore_eos',
        'min_tokens': 'min_tokens',
        'skip_special_tokens': 'skip_special_tokens',
        'truncate_prompt_tokens': 'truncate_prompt_tokens',
        'spaces_between_special_tokens': 'spaces_between_special_tokens',
    }
    default_parameters = {
        'max_tokens': _DEFAULT_MAX_TOKENS,
    }
    mapped_parameters = {
        name_map[k]: v for k, v in json_parameters.items() if k in name_map
    }
    return cls(**(default_parameters | mapped_parameters))

  def to_dict(self) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in dataclasses.asdict(self).items()
        if value is not None
    }


def _dicom_ct_or_mri_volume_slice_index_text_entry(
    slice_index: int,
) -> Mapping[str, Any]:
  return {
      predictor_const.INPUT_TYPE: 'text',
      'text': f'SLICE {slice_index}',
  }


@dataclasses.dataclass(frozen=True)
class _MedGemmaPredictionRequest:
  """MedGemma model input."""

  messages: Sequence[dict[str, Any]]
  content: Sequence[_MedGemmaContent]
  parameters: _MedGemmaPredictionParameters
  add_generation_prompt: bool

  def model_input(
      self,
      prompt_converter: Callable[[list[dict[str, Any]], dict[str, Any]], str],
  ) -> Mapping[str, np.ndarray]:
    """Model input for prediction."""
    images = []
    revised_msgs = []
    image_content_index = 0
    for message in self.messages:
      if 'content' not in message or isinstance(message['content'], str):
        revised_msgs.append(message)
        continue
      revised_content = []
      for entry in message['content']:
        entry_type = entry[predictor_const.INPUT_TYPE]
        if entry_type not in predictor_const.IMAGE_INPUT_TYPES:
          revised_content.append(entry)
          continue
        entry = copy.copy(entry)
        entry[predictor_const.INPUT_TYPE] = (
            _MESSAGE_CONTENT_ENTRY_TYPE_REMAP.get(entry_type, entry_type)
        )
        image_content = self.content[image_content_index]
        image_content_index += 1
        for data_source in image_content.content.data_acquisition_iterator():
          acquision_data_source = data_source.acquision_data_source
          if acquision_data_source not in _DICOM_CT_OR_MRI_VOLUME_DATA_SOURCES:
            for img in data_source.acquision_data_source_iterator:
              images.append(img)
              revised_content.append(entry)
            continue
          for slice_index, img in enumerate(
              data_source.acquision_data_source_iterator, 1
          ):
            if slice_index == 2:
              # add indicator for 1st slice if there are at least 2 slices.
              revised_content.append(
                  _dicom_ct_or_mri_volume_slice_index_text_entry(1)
              )
            images.append(img)
            revised_content.append(entry)
            if slice_index == 1:
              continue
            # automatically inject slice index prompts for DICOM CT and MRI
            # volume data sources if there is more than 1 slice.
            revised_content.append(
                _dicom_ct_or_mri_volume_slice_index_text_entry(slice_index)
            )
      message = copy.copy(message)
      message['content'] = revised_content
      revised_msgs.append(message)
    prompt = prompt_converter(
        revised_msgs,
        {'add_generation_prompt': self.add_generation_prompt},
    )
    input_map = {
        'text_input': np.array([prompt.encode('utf-8')], dtype=np.object_),
        'exclude_input_in_output': np.array([1], dtype=np.bool_),
        'return_num_input_tokens': np.array([1], dtype=np.bool_),
        'return_num_output_tokens': np.array([1], dtype=np.bool_),
    }
    if images:
      if (
          len(images) > _PNG_PROCESS_POOL_THRESHOLD
          and flags.WORKER_DOWNLOAD_PARALLELISM.value == 'PROCESS'
          and 'png' in flags.IMAGE_INPUT_COMPRESSION_FORMAT_FLAG.value.lower()
      ):
        # For PNG compression using a process pool is faster when
        # number of images > 20 at ~100 pool is ~1 sec faster
        # Thresholds emperically determined.
        # For JPEG compression, process pool is slower in most cases and
        # offers negligible speed gains (~40 ms) when number of images is large.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=5
        ) as thread_pool:
          images = list(thread_pool.map(_encode_image_bytes, images))
      else:
        images = list(map(_encode_image_bytes, images))
      input_map['image'] = np.array(images, dtype=np.object_)
    return input_map


def prediction_input_json_to_embedding_request(
    config: dicom_wsi_configuration.ConfigurationSettings,
    json_metadata: Mapping[str, Any],
) -> _MedGemmaPredictionRequest:
  """Converts json to embedding request.

  Args:
    config: The configuration settings for the data source.
    json_metadata: The value of the JSON payload provided to the API.

  Returns:
    Structured EmbeddingRequest object.

  Raises:
    InvalidRequestFieldError: If the provided fields are invalid.
  """
  json_validation_utils.validate_str_key_dict(json_metadata)
  try:
    parameters = _MedGemmaPredictionParameters.from_json(json_metadata)
    messages = json_validation_utils.validate_list(
        json_metadata.get('messages', None)
    )
    content = _parse_all_content(config, messages)
  except json_validation_utils.ValidationError as e:
    raise data_accessor_errors.InvalidRequestFieldError(
        f'Invalid request field: {e}'
    ) from e
  add_generation_prompt = json_metadata.get('add_generation_prompt', True)
  return _MedGemmaPredictionRequest(
      messages, content, parameters, add_generation_prompt
  )


def _validate_instance_list(json_metadata: Mapping[str, Any]) -> Sequence[Any]:
  if not isinstance(json_metadata, dict):
    raise data_accessor_errors.InvalidRequestFieldError(
        'Request is not a dict.'
    )
  val = json_metadata.get(predictor_const.INSTANCES)
  if isinstance(val, list):
    return val
  raise data_accessor_errors.InvalidRequestFieldError(
      'Invalid input, missing expected'
      f' key: {predictor_const.INSTANCES} and associated list of values.'
  )


def _get_inst_data_map_func(
    context: contextlib.ExitStack,
    med_gemma_content: _MedGemmaContent,
) -> Optional[data_accessor_errors.DataAccessorError]:
  """Returns retrieved data or exception."""
  try:
    med_gemma_content.content.load_data(context)
    return None
  except data_accessor_errors.DataAccessorError as exp:
    return exp


@dataclasses.dataclass(frozen=True)
class _MedGemmaPredictionResults:
  """Model response."""

  text_output: tuple[str, ...]
  num_input_tokens: int
  num_output_tokens: tuple[int, ...]


class _ModelPredictor:
  """Retrieves data and runs data across model predictor."""

  def __init__(
      self,
      *,
      prompt_converter: Callable[[list[dict[str, Any]], dict[str, Any]], str],
  ):
    self._threadpool_max_workers = max(
        flags.THREAD_POOL_MAX_WORKERS_FLAG.value, 1
    )
    self._thread_pool_timeout = flags.THREAD_POOL_TIMEOUT_FLAG.value
    self._prompt_converter = prompt_converter

  def _prefetch_instance_data_async(
      self,
      stack: contextlib.ExitStack,
      med_gemma_content: Sequence[_MedGemmaContent],
  ) -> Sequence[data_accessor_errors.DataAccessorError]:
    """Calls function in parallel to init each instance."""
    if not med_gemma_content:
      return []
    if len(med_gemma_content) == 1:
      result = _get_inst_data_map_func(stack, med_gemma_content[0])
      return [] if result is None else [result]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self._threadpool_max_workers,
    ) as thread_pool:
      results = thread_pool.map(
          functools.partial(_get_inst_data_map_func, stack),
          med_gemma_content,
          timeout=self._thread_pool_timeout,
      )
    return [r for r in results if r is not None]

  def predict_request(
      self,
      model: model_runner.ModelRunner,
      request: _MedGemmaPredictionRequest,
  ) -> Union[
      _MedGemmaPredictionResults,
      Sequence[
          data_accessor_errors.DataAccessorError | model_runner.ModelError
      ],
  ]:
    """Get imaging in parallel and then run ML model once."""
    start_time = time.time()
    # create a list of all content required for inference request.
    content_list = request.content
    with contextlib.ExitStack() as stack:
      errors_loading_data = self._prefetch_instance_data_async(
          stack, content_list
      )
      if errors_loading_data:
        return errors_loading_data
      # generate text input from med gemma prediction.
      # model execution parameters
      # call medgemma and return response.
      try:
        result = model.run_model_multiple_output(
            model_input=request.model_input(self._prompt_converter),
            parameters=request.parameters.to_dict(),
            model_output_keys={
                'text_output',
                'num_input_tokens',
                'num_output_tokens',
            },
        )
      except model_runner.ModelError as e:
        return [e]

    cloud_logging_client.info(
        f'Called embedding model; {time.time() - start_time} (sec).'
    )
    # text_output is a ndarray containing byte-encoded strings.
    return _MedGemmaPredictionResults(
        tuple([
            text_output.decode('utf-8') for text_output in result['text_output']
        ]),
        int(result['num_input_tokens']),
        tuple(result['num_output_tokens'].ravel().tolist()),
    )


def _instance_response(
    result: _MedGemmaPredictionResults,
) -> Mapping[str, Any]:
  """Returns a JSON-serializable embedding instance responses."""
  choices = []
  for index, text_output in enumerate(result.text_output):
    choices.append({
        'index': index,
        'message': {'content': text_output, 'role': 'assistant'},
    })
  completion_tokens = sum(result.num_output_tokens)
  return {
      'id': str(uuid.uuid4()),
      'object': 'chat.completion',
      'created': int(time.time()),
      'choices': choices,
      'usage': {
          'prompt_tokens': result.num_input_tokens,
          'completion_tokens': completion_tokens,
          'total_tokens': result.num_input_tokens + completion_tokens,
      },
      'model': 'placeholder',
  }


def _prediction_error_response(
    ds_error: data_accessor_errors.DataAccessorError | model_runner.ModelError,
) -> Mapping[str, Any]:
  """Returns a JSON-serializable error response for prediction exceptions."""
  if isinstance(ds_error, model_runner.ModelError):
    message = str(ds_error)
  elif isinstance(ds_error, data_accessor_errors.DataAccessorError):
    message = ds_error.api_description
  else:
    # Internal error should never be raised.
    raise ValueError(f'Unsupported error type: {type(ds_error)}')
  error = {
      'message': message[:_MAX_ERROR_DESCRIPTION_LENGTH],
      'object': predictor_const.ERROR,
      # 'type': ds_error.error_code.category,
  }
  return {predictor_const.ERROR: error}


class MedGemmaPredictor:
  """Callable responsible for generating embeddings."""

  def __init__(
      self,
      *,
      prompt_converter: Callable[[list[dict[str, Any]], dict[str, Any]], str],
      instance_validator: jsonschema.Draft202012Validator | None = None,
  ):
    self._prompt_converter = prompt_converter
    self._instance_validator = instance_validator

  def _single_predict(
      self,
      prediction_input: Mapping[str, Any],
      model: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs chat completion on the provided conversation.

    Args:
      prediction_input: JSON formatted input for embedding prediction.
      model: ModelRunner to handle model step.

    Returns:
      JSON formatted output.

    Raises:
      ERROR_LOADING_DICOM: If the provided patches are not concated.
    """
    config = dicom_wsi_configuration.ConfigurationSettings(
        endpoint_input_width=flags.MODEL_INPUT_WIDTH_FLAG.value,
        endpoint_input_height=flags.MODEL_INPUT_HEIGHT_FLAG.value,
        approved_dicom_stores=flags.APPROVED_DICOM_STORE_SOURCE_LIST_FLAG.value,
        icc_profile_cache_configuration=dicom_wsi_configuration.IccProfileCacheConfiguration(
            gcs_bucket=flags.ICC_PROFILE_CACHE_GCS_BUCKET_FLAG.value,
            redis_ip=flags.ICC_PROFILE_CACHE_REDIS_IP_FLAG.value,
            redis_port=flags.ICC_PROFILE_CACHE_REDIS_PORT_FLAG.value,
            store_icc_profile_bytes_in_redis=flags.STORE_ICC_PROFILE_BYTES_IN_REDIS_FLAG.value,
            testing=flags.IS_DEBUGGING_FLAG.value,
        ),
        max_parallel_download_workers=max(
            1, flags.MAX_PARALLEL_DOWNLOAD_WORKERS_FLAG.value
        ),
    )
    try:
      med_gemma_predictionrequest = prediction_input_json_to_embedding_request(
          config, prediction_input
      )
    except data_accessor_errors.DataAccessorError as exp:
      return dict(_prediction_error_response(exp))

    predictor = _ModelPredictor(prompt_converter=self._prompt_converter)
    try:
      result = predictor.predict_request(model, med_gemma_predictionrequest)
      if isinstance(result, _MedGemmaPredictionResults):
        return dict(_instance_response(result))
      # Error handling only from here on.
    except data_accessor_errors.DataAccessorError as exp:
      return dict(_prediction_error_response(exp))
    try:
      result = result[0]
    except IndexError:
      result = data_accessor_errors.InternalError('Invalided model response')
    cloud_logging_client.info('Returning embeddings.')
    return dict(_prediction_error_response(result))

  def predict(
      self,
      prediction_input: Mapping[str, Any],
      model: model_runner.ModelRunner,
  ) -> dict[str, Any]:
    """Runs chat completion on a request."""

    # Sort out whether this is a singular request or multiple instances.
    if INSTANCES_KEY in prediction_input:
      if len(prediction_input[INSTANCES_KEY]) == 1:
        try:
          if self._instance_validator is not None:
            self._instance_validator.validate(
                prediction_input[INSTANCES_KEY][0]
            )
        except jsonschema.exceptions.ValidationError as e:
          cloud_logging_client.warning('Input validation failed')
          return {'error': str(e)}
        return {
            PREDICTIONS_KEY: self._single_predict(
                prediction_input[INSTANCES_KEY][0], model
            )
        }
      try:
        if self._instance_validator is not None:
          for instance in prediction_input[INSTANCES_KEY]:
            self._instance_validator.validate(instance)
      except jsonschema.exceptions.ValidationError as e:
        cloud_logging_client.warning('Input validation failed')
        return {'error': str(e)}
      predictions = [
          self._single_predict(instance, model)
          for instance in prediction_input[INSTANCES_KEY]
      ]
      return {PREDICTIONS_KEY: predictions}

    try:
      if self._instance_validator is not None:
        self._instance_validator.validate(prediction_input)
    except jsonschema.exceptions.ValidationError as e:
      cloud_logging_client.warning('Input validation failed')
      return {'error': str(e)}
    return self._single_predict(prediction_input, model)
