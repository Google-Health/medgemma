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
"""Defines abstract representation of Error."""

import abc
import enum


class ErrorCode(enum.Enum):
  """Error codes for DataAccessors."""

  INTERNAL_ERROR = 'INTERNAL_ERROR'

  # Request errors
  INVALID_REQUEST_FIELD_ERROR = 'INVALID_REQUEST_FIELD_ERROR'
  INVALID_CREDENTIALS_ERROR = 'INVALID_CREDENTIALS_ERROR'

  # Network errors
  HTTP_ERROR = 'HTTP_ERROR'

  # DICOM errors
  DICOM_PATH_ERROR = 'DICOM_PATH_ERROR'
  UNAPPROVED_DICOM_STORE_ERROR = 'UNAPPROVED_DICOM_STORE_ERROR'

  # DICOM WSI specific errors.
  TOO_MANY_PATCHES_ERROR = 'TOO_MANY_PATCHES_ERROR'
  LEVEL_NOT_FOUND_ERROR = 'LEVEL_NOT_FOUND_ERROR'
  EZ_WSI_STATE_ERROR = 'EZ_WSI_STATE_ERROR'
  INVALID_ICC_PROFILE_TRANSFORM_ERROR = 'INVALID_ICC_PROFILE_TRANSFORM_ERROR'
  IMAGE_DIMENSION_ERROR = 'IMAGE_DIMENSION_ERROR'
  DICOM_TILED_FULL_ERROR = 'DICOM_TILED_FULL_ERROR'
  DICOM_ERROR = 'DICOM_ERROR'
  DICOM_IMAGE_DOWNSAMPLING_TOO_LARGE_ERROR = (
      'DICOM_IMAGE_DOWNSAMPLING_TOO_LARGE_ERROR'
  )

  # GCS ERRORS
  UNHANDLED_GCS_FILE_ERROR = 'UNHANDLED_GCS_FILE_ERROR'

  # HTTP ERRORS
  UNHANDLED_HTTP_FILE_ERROR = 'UNHANDLED_HTTP_FILE_ERROR'

  # User provided bytes not processed by any file handler.
  UNHANDLED_GENERIC_BYTES_ERROR = 'UNHANDLED_GENERIC_BYTES_ERROR'

  # Patch specific errors.
  PATCH_OUTSIDE_OF_IMAGE_DIMENSIONS_ERROR = (
      'PATCH_OUTSIDE_OF_IMAGE_DIMENSIONS_ERROR'
  )
  PATCH_DIMENSIONS_DO_NOT_MATCH_ENDPOINT_INPUT_DIMENSIONS_ERROR = (
      'PATCH_DIMENSIONS_DO_NOT_MATCH_ENDPOINT_INPUT_DIMENSIONS_ERROR'
  )


class DataAccessorError(Exception, metaclass=abc.ABCMeta):
  """Base error class for Pete Errors."""

  def __init__(self, message: str = '', api_description: str = ''):
    """Errors with optional alternative descriptions for API echoing."""
    super().__init__(message if message else api_description)
    self._api_description = api_description

  @property
  def api_description(self) -> str:
    """Returns the API description of the error or the default message."""
    return self._api_description if self._api_description else str(self)

  @property
  @abc.abstractmethod
  def error_code(self) -> ErrorCode:
    """Returns the error code of the error."""


class UnhandledGenericBytesError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.UNHANDLED_GENERIC_BYTES_ERROR


class UnhandledGcsFileError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.UNHANDLED_GCS_FILE_ERROR


class UnhandledHttpFileError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.UNHANDLED_HTTP_FILE_ERROR


class InvalidRequestFieldError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.INVALID_REQUEST_FIELD_ERROR


class InvalidCredentialsError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.INVALID_CREDENTIALS_ERROR


class LevelNotFoundError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.LEVEL_NOT_FOUND_ERROR


class EzWsiStateError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.EZ_WSI_STATE_ERROR


class PatchOutsideOfImageDimensionsError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.PATCH_OUTSIDE_OF_IMAGE_DIMENSIONS_ERROR


class PatchCoordinateError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return (
        ErrorCode.PATCH_DIMENSIONS_DO_NOT_MATCH_ENDPOINT_INPUT_DIMENSIONS_ERROR
    )


class HttpError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.HTTP_ERROR


class InvalidIccProfileTransformError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.INVALID_ICC_PROFILE_TRANSFORM_ERROR


class ImageDimensionError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.IMAGE_DIMENSION_ERROR


class DicomTiledFullError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.DICOM_TILED_FULL_ERROR


class DicomPathError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.DICOM_PATH_ERROR


class DicomError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.DICOM_ERROR


class DicomImageDownsamplingTooLargeError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.DICOM_IMAGE_DOWNSAMPLING_TOO_LARGE_ERROR


class TooManyPatchesError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.TOO_MANY_PATCHES_ERROR


class UnapprovedDicomStoreError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.UNAPPROVED_DICOM_STORE_ERROR


class InternalError(DataAccessorError):

  @property
  def error_code(self) -> ErrorCode:
    return ErrorCode.INTERNAL_ERROR
