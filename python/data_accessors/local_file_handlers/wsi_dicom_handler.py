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
"""local handler for handling traditional image files."""

import io
import math
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

from ez_wsi_dicomweb import dicom_frame_decoder
import numpy as np
from PIL import ImageCms
import pydicom

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.local_file_handlers import generic_dicom_handler
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module


_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])


def _get_uncompressed_dicom_frame_bytes(dcm: pydicom.FileDataset) -> np.ndarray:
  """Loads DICOM instance frames into memory as a list of frames(bytes)."""
  try:
    return dcm.pixel_array
  except (AttributeError, ValueError) as exp:
    raise data_accessor_errors.DicomError(
        f'Cannot decode pixel data: {exp}.'
    ) from exp


def _get_compressed_dicom_frame_bytes(
    dcm: pydicom.FileDataset,
) -> Sequence[bytes]:
  """Loads DICOM instance frames into memory as a list of frames(bytes).

  Args:
    dcm: Pydicom instance.

  Returns:
    List of bytes encoded in DICOM instance frames.
  """
  if 'PixelData' not in dcm or not dcm.PixelData:
    raise data_accessor_errors.DicomError('DICOM missing PixelData.')
  try:
    number_of_frames = int(dcm.NumberOfFrames)
  except (TypeError, ValueError, AttributeError):
    number_of_frames = 1
  if number_of_frames < 1:
    raise data_accessor_errors.DicomError('Invalid number of frames in DICOM.')
  if _PYDICOM_MAJOR_VERSION <= 2:
    # pytype: disable=module-attr
    frame_bytes_generator = pydicom.encaps.generate_pixel_data_frame(
        dcm.PixelData, number_of_frames
    )
    # pytype: enable=module-attr
  else:
    # pytype: disable=module-attr
    frame_bytes_generator = pydicom.encaps.generate_frames(
        dcm.PixelData, number_of_frames=number_of_frames
    )
    # pytype: enable=module-attr
  return [frame_bytes for frame_bytes in frame_bytes_generator]


def _crop_target_frame(
    frame_pixels_start: int, frame_dim: int, patch_origin: int, patch_dim: int
) -> Tuple[int, int]:
  frame_pixels_start -= patch_origin
  start_pos = max(frame_pixels_start, 0)
  end_pos = min(frame_pixels_start + frame_dim, patch_dim)
  return start_pos, end_pos


def _pad_channel(frame: np.ndarray) -> np.ndarray:
  """Pads frame channelwith zero dim."""
  if frame.ndim == 2:
    return np.expand_dims(frame, 2)
  return frame


def _get_frame(
    dcm: pydicom.FileDataset,
    dicom_frames: Union[Sequence[bytes], np.ndarray],
    frame_index: int,
    frame_x_offset: int,
    frame_y_offset: int,
    frames_per_row: int,
    frames_per_column: int,
) -> np.ndarray:
  """Returns a frame from a DICOM instance."""
  try:
    number_of_frames = int(dcm.NumberOfFrames)
  except (ValueError, AttributeError, TypeError):
    number_of_frames = 1
  if (
      frame_x_offset >= frames_per_row
      or frame_y_offset >= frames_per_column
      or frame_index >= number_of_frames
  ):
    frame_data = _get_frame(
        dcm, dicom_frames, 0, 0, 0, frames_per_row, frames_per_column
    )
    return _pad_channel(np.zeros_like(frame_data))
  if isinstance(dicom_frames, np.ndarray):
    if dcm.SamplesPerPixel == 1:
      if dicom_frames.ndim == 2:
        return _pad_channel(dicom_frames)
      else:
        return _pad_channel(dicom_frames[frame_index, ...])
    else:
      if dicom_frames.ndim == 3:
        return _pad_channel(dicom_frames)
      else:
        return _pad_channel(dicom_frames[frame_index, ...])
  result = dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
      dicom_frames[frame_index], dcm.file_meta.TransferSyntaxUID
  )
  if result is None:
    raise data_accessor_errors.DicomError('DICOM cannot decode pixel data.')
  return _pad_channel(result)


def _get_patch(
    dcm: pydicom.FileDataset,
    dicom_frames: Union[Sequence[bytes], np.ndarray],
    resize_image_dimensions: Optional[image_dimension_utils.ImageDimensions],
    pc: patch_coordinate_module.PatchCoordinate,
    validate_patch_in_dim: bool,
) -> np.ndarray:
  """Returns a patch from a DICOM instance."""
  frame_width = dcm.Columns
  frame_height = dcm.Rows
  if (
      resize_image_dimensions is not None
      and resize_image_dimensions.width == dcm.TotalPixelMatrixColumns
      and resize_image_dimensions.height == dcm.TotalPixelMatrixRows
  ):
    resize_image_dimensions = None
  if validate_patch_in_dim:
    if resize_image_dimensions is None:
      pc.validate_patch_in_dim(
          image_dimension_utils.ImageDimensions(
              dcm.TotalPixelMatrixColumns, dcm.TotalPixelMatrixRows
          )
      )
    else:
      pc.validate_patch_in_dim(resize_image_dimensions)

  projected_patch = image_dimension_utils.get_projected_patch(
      pc,
      dcm.TotalPixelMatrixColumns,
      dcm.TotalPixelMatrixRows,
      resize_image_dimensions,
  )
  start_x = projected_patch.start_x
  start_y = projected_patch.start_y
  end_x = start_x + projected_patch.projected_read_width
  end_y = start_y + projected_patch.projected_read_height
  frames_per_row = int(math.ceil(dcm.TotalPixelMatrixColumns / frame_width))
  frames_per_column = int(math.ceil(dcm.TotalPixelMatrixRows / frame_height))
  if frames_per_row <= 0 or frames_per_column <= 0:
    raise data_accessor_errors.DicomError('Frame per row or column is zero.')

  start_frame_index_y = int(start_y / frame_height)
  end_frame_index_y = int((end_y - 1) / frame_height) + 1

  start_frame_index_x = int(start_x / frame_width)
  end_frame_index_x = int((end_x - 1) / frame_width) + 1

  frame = _get_frame(
      dcm, dicom_frames, 0, 0, 0, frames_per_row, frames_per_column
  )
  memory = np.zeros(
      (
          projected_patch.projected_read_height,
          projected_patch.projected_read_width,
          dcm.SamplesPerPixel,
      ),
      dtype=frame.dtype,
  )
  frame_y = start_frame_index_y * frame_height
  x_frame_start = start_frame_index_x * frame_width
  source_y = start_y - frame_y
  y_frame_offset = start_frame_index_y * frames_per_row
  for y_frame_index in range(start_frame_index_y, end_frame_index_y):
    cy_s, cy_e = _crop_target_frame(
        frame_y,
        frame_height,
        projected_patch.start_y,
        projected_patch.projected_read_height,
    )
    copy_height = cy_e - cy_s
    frame_x = x_frame_start
    source_x = start_x - frame_x
    for x_frame_index in range(start_frame_index_x, end_frame_index_x):
      frame = _get_frame(
          dcm,
          dicom_frames,
          x_frame_index + y_frame_offset,
          x_frame_index,
          y_frame_index,
          frames_per_row,
          frames_per_column,
      )
      cx_s, cx_e = _crop_target_frame(
          frame_x,
          frame_width,
          projected_patch.start_x,
          projected_patch.projected_read_width,
      )
      copy_width = cx_e - cx_s
      memory[cy_s:cy_e, cx_s:cx_e, ...] = frame[
          source_y : source_y + copy_height,
          source_x : source_x + copy_width,
          ...,
      ]
      frame_x += frame_width
      source_x = 0
    frame_y += frame_height
    y_frame_offset += frames_per_row
    source_y = 0
  if memory.shape[0] != pc.height or memory.shape[1] != pc.width:
    memory = image_dimension_utils.resize_projected_patch(
        pc, projected_patch, memory
    )
  return memory


def _create_icc_profile_image_transformation(
    dcm: pydicom.FileDataset,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
) -> Optional[ImageCms.ImageCmsTransform]:
  """Transforms image to target ICC profile."""
  if target_icc_profile is None:
    return None
  if dcm.SamplesPerPixel != 3:
    return None
  if dcm.BitsAllocated != 8:
    return None
  icc_profile_bytes = icc_profile_utils.get_dicom_icc_profile_bytes(dcm)
  if not icc_profile_bytes:
    return None
  return icc_profile_utils.create_icc_profile_transformation(
      icc_profile_bytes, target_icc_profile
  )


def _decode_dicom_image(
    dcm: pydicom.FileDataset,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
    patch_coordinates: Sequence[patch_coordinate_module.PatchCoordinate],
    resize_image_dimensions: Optional[image_dimension_utils.ImageDimensions],
    patch_required_to_be_fully_in_source_image: bool,
) -> Iterator[np.ndarray]:
  """Decode DICOM image and return decoded image bytes."""
  generic_dicom_handler.validate_transfer_syntax(dcm)
  generic_dicom_handler.validate_samples_per_pixel(dcm)
  generic_dicom_handler.validate_samples_per_pixel_and_photometric_interpretation_match(
      dcm
  )
  try:
    encapsulated_dicom = (
        dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
            dcm.file_meta.TransferSyntaxUID
        )
    )
  except (AttributeError, ValueError) as exp:
    raise data_accessor_errors.DicomError(
        'DICOM missing TransferSyntaxUID.'
    ) from exp
  if encapsulated_dicom:
    dicom_frames = _get_compressed_dicom_frame_bytes(dcm)
    if dicom_frames is None:
      raise data_accessor_errors.DicomError('DICOM cannot decode pixel data.')
  else:
    generic_dicom_handler.validate_unencapsulated_photometric_interpretation(
        dcm
    )
    dicom_frames = _get_uncompressed_dicom_frame_bytes(dcm)

  if not patch_coordinates:
    if resize_image_dimensions is None:
      patch_coordinates = [
          patch_coordinate_module.PatchCoordinate(
              0, 0, dcm.TotalPixelMatrixColumns, dcm.TotalPixelMatrixRows
          )
      ]
    else:
      patch_coordinates = [
          patch_coordinate_module.PatchCoordinate(
              0,
              0,
              resize_image_dimensions.width,
              resize_image_dimensions.height,
          )
      ]
  icc_profile_image_transformation = _create_icc_profile_image_transformation(
      dcm, target_icc_profile
  )
  try:
    monochorome1_transformation = (
        dcm.SamplesPerPixel == 1
        and dcm.PhotometricInterpretation == generic_dicom_handler.MONOCHROME1
    )
  except (AttributeError, ValueError) as exp:
    raise data_accessor_errors.DicomError(
        'DICOM missing PhotometricInterpretation or SamplesPerPixel.'
    ) from exp
  for pc in patch_coordinates:
    decoded_image_bytes = _get_patch(
        dcm,
        dicom_frames,
        resize_image_dimensions,
        pc,
        patch_required_to_be_fully_in_source_image,
    )
    if icc_profile_image_transformation is not None:
      decoded_image_bytes = (
          icc_profile_utils.transform_image_bytes_to_target_icc_profile(
              decoded_image_bytes, icc_profile_image_transformation
          )
      )
    if monochorome1_transformation:
      decoded_image_bytes = (
          np.iinfo(decoded_image_bytes.dtype).max - decoded_image_bytes
      )
    yield decoded_image_bytes


class WsiDicomHandler(abstract_handler.AbstractHandler):
  """Reads a traditional image from local file system."""

  def process_files(
      self,
      instance_patch_coordinates: Sequence[
          patch_coordinate_module.PatchCoordinate
      ],
      base_request: Mapping[str, Any],
      file_paths: abstract_handler.InputFileIterator,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    instance_extensions = abstract_handler.get_base_request_extensions(
        base_request
    )
    for file_path in file_paths:
      try:
        with pydicom.dcmread(file_path, specific_tags=['SOPClassUID']) as dcm:
          if (
              dcm.SOPClassUID
              != dicom_source_utils.VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
          ):
            return
        if isinstance(file_path, io.BytesIO):
          file_path.seek(0)  # pytype: disable=attribute-error
        with pydicom.dcmread(file_path) as dcm:
          try:
            number_of_frames = int(dcm.NumberOfFrames)
          except (ValueError, AttributeError, TypeError):
            number_of_frames = 1
          if (
              number_of_frames > 1
              and dcm.DimensionOrganizationType != 'TILED_FULL'
          ):
            raise data_accessor_errors.DicomTiledFullError(
                'DICOM DimensionOrganizationType is not TILED_FULL.'
            )
          if 'ConcatenationUID' in dcm and dcm.ConcatenationUID:
            raise data_accessor_errors.DicomError(
                'Reading concatenated WSI DICOM from sources other than a DICOM'
                ' store is not supported.'
            )
          target_icc_profile = icc_profile_utils.get_target_icc_profile(
              instance_extensions
          )
          patch_required_to_be_fully_in_source_image = patch_coordinate_module.patch_required_to_be_fully_in_source_image(
              instance_extensions
          )
          resize_image_dimensions = (
              image_dimension_utils.get_resize_image_dimensions(
                  instance_extensions
              )
          )
          yield abstract_data_accessor.DataAcquisition(
              abstract_data_accessor.AccessorDataSource.DICOM_WSI_MICROSCOPY_PYRAMID_LEVEL,
              _decode_dicom_image(
                  dcm,
                  target_icc_profile,
                  instance_patch_coordinates,
                  resize_image_dimensions,
                  patch_required_to_be_fully_in_source_image,
              ),
          )
          # mark file as being processed so custom iterator will now return next
          # file in sequence.
          file_paths.processed_file()
      except pydicom.errors.InvalidDicomError:
        return
