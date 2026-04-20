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
"""local handler for handling generic DICOM files."""

import abc
import dataclasses
import io
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

from ez_wsi_dicomweb import dicom_frame_decoder
import numpy as np
from PIL import ImageCms
import pydicom
import pydicom.errors

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_errors
from data_accessors.local_file_handlers import abstract_handler
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module
from data_processing import image_utils


_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])

# DICOM Tag Keywords
_PIXEL_DATA = 'PixelData'
_WINDOW_CENTER = 'WindowCenter'
_WINDOW_WIDTH = 'WindowWidth'

# Default CT Window (-1400 to 600 HU)
_DEFAULT_CT_WINDOW_CENTER = 100
_DEFAULT_CT_WINDOW_WIDTH = 2500

# DICOM Transfer Syntax UIDs
_IMPLICIT_VR_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2'
_EXPLICIT_VR_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2.1'
_DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN_TRANSFER_SYNTAX = '1.2.840.10008.1.2.1.99'

VALID_UNENCAPSULATED_DICOM_TRANSFER_SYNTAXES = frozenset([
    _IMPLICIT_VR_ENDIAN_TRANSFER_SYNTAX,
    _EXPLICIT_VR_ENDIAN_TRANSFER_SYNTAX,
    _DEFLATED_EXPLICIT_VR_LITTLE_ENDIAN_TRANSFER_SYNTAX,
])

# PhotometricInterpretation Coded Values
MONOCHROME1 = 'MONOCHROME1'
_MONOCHROME2 = 'MONOCHROME2'
_RGB = 'RGB'
_SINGLE_CHANNEL_PHOTOMETRIC_INTERPRETATION = frozenset(
    [MONOCHROME1, _MONOCHROME2]
)
_SUPPORTED_UNENCAPSULATED_PHOTOMETRIC_INTERPRETATIONS = frozenset(
    [MONOCHROME1, _MONOCHROME2, _RGB]
)

_SUPPORTED_SAMPLES_PER_PIXEL = frozenset([1, 3])

_MODALITY = dicom_source_utils.MODALITY
_CXR_MODALITIES = dicom_source_utils.CXR_MODALITIES
_WINDOWED_MODALITIES = tuple([_MODALITY.CT])

_DICOM_TAG_KEYWORDS_EQUAL_VALUE_FOR_INSTANCES_IN_SAME_ACQUISITION = (
    'SOPClassUID',
    'StudyInstanceUID',
    'SeriesInstanceUID',
    'ImageType',
    'AcquisitionUID',
    'AcquisitionNumber',
    'AcquisitionDateTime',
    'AcquisitionDate',
    'AcquisitionTime',
    'PyramidUID',
    'Modality',
)


class ImageTransform(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def apply(self, image: np.ndarray) -> np.ndarray:
    """Returns the window operation on an dicom image."""


class ModalityDefaultImageTransform:
  """Default window operation."""

  def __init__(
      self,
      get_image_transform_op: Callable[
          [Union[pydicom.Dataset, pydicom.FileDataset]],
          Optional[ImageTransform],
      ],
  ):
    self._get_image_transform_op = get_image_transform_op

  def get_image_image_transform(
      self, dcm: pydicom.FileDataset
  ) -> Optional[ImageTransform]:
    """Returns the default window operation for the dicom modality."""
    return self._get_image_transform_op(dcm)


_UINT8_TYPE = type(np.uint8)
_UINT16_TYPE = type(np.uint16)


class TraditionalWindow(ImageTransform):
  """Traditional window operation used in CXR and traditional for CT images.

  Background: The voxels encoded within CT imaging are typically expressed as
  signed 16-bit hounsfield units (HU), one value per-voxel. CT images are
  commonly visualized as a grayscale image. The imaging is typically windowed
  for human reading tasks to increase the contrast across a task specific
  diagnostic range.
  """

  def __init__(
      self,
      center: int,
      width: int,
      dtype: Optional[Union[_UINT8_TYPE, _UINT16_TYPE]] = None,
  ):
    self._center = center
    self._width = width
    self._default_type = dtype if dtype is not None else np.uint8
    if self._default_type not in (np.uint8, np.uint16):
      raise ValueError(
          'Output dtype must be either uint8 or uint16, got'
          f' {self._default_type}.'
      )

  @property
  def center(self) -> int:
    return self._center

  @property
  def width(self) -> int:
    return self._width

  def apply(self, image: np.ndarray) -> np.ndarray:
    """Applies the Window operation and returns the windowed image.

    Based on CXR embedding implementation in image_utils. implemented inline to
    address bug in CXR implementation. * actual window range should be center +-
    half width. * rounds interpolated value to nearest integer for better
    precision.

    Args:
      image: An image to be windowed.containing signed integer pixels.

    Returns:
      Windowed image as numpy array.
    """
    iinfo = np.iinfo(self._default_type)
    # Actual range is center - half width to center + half width.
    # Actual number of pixels is width + 1.
    # See https://radiopaedia.org/articles/windowing-ct?lang=us
    half_window_width = self.width // 2
    center = self.center
    top_clip = center + half_window_width
    bottom_clip = center - half_window_width
    # Round prior to cast to minimize precision loss.
    return np.round(
        np.interp(
            image.clip(bottom_clip, top_clip),
            (bottom_clip, top_clip),
            (0, iinfo.max),
        ),
        0,
    ).astype(iinfo)


class RGBWindow(ImageTransform):
  """RGB channel-wise imaging to pack information into windowed image.

  The image encoder used to transform CT imaging into MedGemma performed a
  traditional windowing operation to encode multiple image windows into a
  MedGemma input.

  Background: The voxels encoded within CT imaging are typically expressed as
  signed 16-bit hounsfield units (HU), one value per-voxel. CT images are
  commonly visualized as a grayscale image. The imaging is typically windowed
  for human reading tasks to increase the contrast across a task specific
  diagnostic range.

  MedGemma has was been trained to interpret CT imaging where the RGB channels
  of input imaging correspond to a novel windowing representation to enable it
  to interpret multiple representations of CT imaging simultaneously.
  Specifically, MedGemma 1.5 has been trained with the components defined as
  follows:

  Red (component 0): Wide window; range: -1024 HU (air) to 1024 HU (above bone)
  Green (component 1): Soft tissue window; range: 135 HU (fat) to 215 HU (start
  of bone)
  Blue (component 2): Brain window; range: 0 HU (water) to 80 HU (brain)
  Because, each of the RGB channels in the prompt imaging correspond to a
  windowing prompt imaging prepared using this method will visually appear
  color.
  """

  def __init__(
      self,
      red_window: TraditionalWindow,
      green_window: TraditionalWindow,
      blue_window: TraditionalWindow,
  ):
    self._red_window = red_window
    self._green_window = green_window
    self._blue_window = blue_window

  def apply(self, image: np.ndarray) -> np.ndarray:
    red = self._red_window.apply(image)
    green = self._green_window.apply(image)
    blue = self._blue_window.apply(image)
    return np.concatenate([red, green, blue], axis=-1)


class NopImageTransform(ImageTransform):
  """Default operation used to transform MRI imaging for model input.

  MRI imaging is commonly expressed 16-bit values. Unlike CT the imaging is
  typically not captured in a calibrated acquision, i.e. the voxel values have
  reliative meaning only. MRI imaging is commonly transformed for visualization
  by scaling the values across the dynamic range.
  """

  def apply(self, image: np.ndarray) -> np.ndarray:
    return image


# Default CT Window for MedGemma 1.0
_MEDGEMMA_1_CT_DEFAULT_WINDOW = RGBWindow(
    TraditionalWindow(0, 2048),
    TraditionalWindow(175, 80),
    TraditionalWindow(40, 80),
)


def validate_transfer_syntax(dcm: pydicom.FileDataset) -> None:
  """Validates the transfer syntax of the DICOM instance."""
  try:
    transfer_syntax_uid = dcm.file_meta.TransferSyntaxUID
  except (AttributeError, ValueError) as exp:
    raise data_accessor_errors.DicomError(
        'DICOM missing TransferSyntaxUID.'
    ) from exp
  if transfer_syntax_uid in VALID_UNENCAPSULATED_DICOM_TRANSFER_SYNTAXES:
    return
  if dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
      transfer_syntax_uid
  ):
    return
  raise data_accessor_errors.DicomError(
      'DICOM instance encoded using unsupported transfer syntax.'
      f' {transfer_syntax_uid}.'
  )


def _transform_image_to_target_icc_profile(
    decoded_image_bytes: np.ndarray,
    compressed_image_bytes: bytes,
    dcm: pydicom.FileDataset,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
) -> np.ndarray:
  """Transforms image to target ICC profile."""
  if target_icc_profile is None:
    return decoded_image_bytes
  if dcm.SamplesPerPixel != 3:
    return decoded_image_bytes
  if dcm.BitsAllocated != 8:
    return decoded_image_bytes
  if decoded_image_bytes.ndim != 3 or decoded_image_bytes.shape[2] != 3:
    return decoded_image_bytes
  icc_profile_bytes = icc_profile_utils.get_dicom_icc_profile_bytes(dcm)
  if compressed_image_bytes and not icc_profile_bytes:
    icc_profile_bytes = (
        icc_profile_utils.get_icc_profile_bytes_from_compressed_image(
            compressed_image_bytes
        )
    )
  if not icc_profile_bytes:
    return decoded_image_bytes
  icc_profile_transformation = (
      icc_profile_utils.create_icc_profile_transformation(
          icc_profile_bytes, target_icc_profile
      )
  )
  return icc_profile_utils.transform_image_bytes_to_target_icc_profile(
      decoded_image_bytes, icc_profile_transformation
  )


def _get_encapsulated_dicom_frame_bytes(
    ds: pydicom.FileDataset,
) -> Iterator[bytes]:
  """Returns DICOM bytes from encapsulated PixelData."""
  if _PIXEL_DATA not in ds or not ds.PixelData:
    return iter([])
  try:
    number_of_frames = int(ds.NumberOfFrames)
  except (TypeError, ValueError, AttributeError) as _:
    # DICOM IOD that do not define multi-frame do not contain NumberOfFrames
    # tag. For these IOD, we assume that the image has only one frame.
    number_of_frames = 1
  if number_of_frames < 1:
    return iter([])
  # pytype: disable=module-attr
  if _PYDICOM_MAJOR_VERSION <= 2:
    return pydicom.encaps.generate_pixel_data_frame(
        ds.PixelData, number_of_frames
    )
  return pydicom.encaps.generate_frames(
      ds.PixelData, number_of_frames=number_of_frames
  )
  # pytype: enable=module-attr


@dataclasses.dataclass(frozen=True)
class _FrameBytes:
  compressed_frame_bytes: bytes
  decoded_frame_bytes: np.ndarray


def _safe_yield(dicom_frames: Iterator[bytes]) -> Iterator[bytes]:
  """Catch exceptions raised from from generator and re-raise as DicomError."""
  try:
    for compressed_frame_bytes in dicom_frames:
      yield compressed_frame_bytes
  except ValueError as exp:
    raise data_accessor_errors.DicomError(
        'Error decoding encapsulated DICOM frame.'
    ) from exp


def _decode_encapsulated_dicom_frames(
    dcm: pydicom.FileDataset, dicom_frames_compressed_bytes: Iterator[bytes]
) -> Iterator[_FrameBytes]:
  """Decodes compressed DICOM frames."""
  transfer_syntax_uid = dcm.file_meta.TransferSyntaxUID
  missing_pixel_data = True
  for compressed_frame_bytes in _safe_yield(dicom_frames_compressed_bytes):
    try:
      decoded_frame_bytes = (
          dicom_frame_decoder.decode_dicom_compressed_frame_bytes(
              compressed_frame_bytes, transfer_syntax_uid
          )
      )
    except RuntimeError as exp:
      # image codecs raises a runtime error when it cannot decode the frame.
      raise data_accessor_errors.DicomError(
          'Error decoding encapsulated DICOM PixelData.'
      ) from exp
    if decoded_frame_bytes is None:
      raise data_accessor_errors.DicomError(
          'Error decoding encapsulated DICOM PixelData.'
      )
    if dcm.SamplesPerPixel == 1 and decoded_frame_bytes.shape[2] == 3:
      decoded_frame_bytes = decoded_frame_bytes[..., 0]
    missing_pixel_data = False
    yield _FrameBytes(compressed_frame_bytes, decoded_frame_bytes)
  if missing_pixel_data:
    raise data_accessor_errors.DicomError('DICOM instance missing PixelData.')


def _decode_unencapsulated_dicom_frames(
    dcm: pydicom.FileDataset,
) -> Iterator[_FrameBytes]:
  """Decodes uncompressed DICOM frames."""
  try:
    if not dcm.PixelData:
      raise data_accessor_errors.DicomError('DICOM instance missing PixelData.')
    if dcm.SamplesPerPixel == 1 and dcm.pixel_array.ndim == 3:
      for frame_index in range(dcm.pixel_array.shape[0]):
        yield _FrameBytes(b'', dcm.pixel_array[frame_index])
    elif dcm.SamplesPerPixel > 1 and dcm.pixel_array.ndim == 4:
      for frame_index in range(dcm.pixel_array.shape[0]):
        yield _FrameBytes(b'', dcm.pixel_array[frame_index])
    else:
      yield _FrameBytes(b'', dcm.pixel_array)
  except (AttributeError, ValueError) as exp:
    raise data_accessor_errors.DicomError(
        'DICOM instance missing PixelData.'
    ) from exp


def _rescale_cxr_dynamic_range(image_bytes: np.ndarray) -> np.ndarray:
  """Rescales dynamic range of image bytes to make across image range."""
  try:
    # For uint8 images, rescaling is not needed
    if image_bytes.dtype == np.uint8:
      return image_bytes
    if np.dtype(image_bytes.dtype).kind != 'u':
      image_bytes = image_utils.shift_to_unsigned(image_bytes)
    # Rescaling dynamic range enables 12 bit imaging to be scaled to uint16.
    # Also will scale signed imaging across full range.
    # Side effect is that will make imaging relative to self.
    return image_utils.rescale_dynamic_range(image_bytes)
  except ValueError as exp:
    raise data_accessor_errors.DicomError(
        'DICOM PixelData contains has incompatible encoding.'
    ) from exp


def _default_cxr_window_op(
    dcm: pydicom.Dataset,
) -> Optional[ImageTransform]:
  """Returns default window operation for CXR imaging."""
  if _WINDOW_WIDTH in dcm and _WINDOW_CENTER in dcm:
    if dcm.WindowCenter is not None and dcm.WindowWidth is not None:
      return TraditionalWindow(dcm.WindowCenter, dcm.WindowWidth)
  return None


def norm_cxr_imaging(
    arr: np.ndarray,
    ds: pydicom.FileDataset,
) -> np.ndarray:
  """Applies data handling from pydicom."""
  # pytype: disable=module-attr
  try:
    if _PYDICOM_MAJOR_VERSION <= 2:
      pixel_array = pydicom.pixels.processing.apply_modality_lut(arr, ds)
    else:
      pixel_array = pydicom.pixels.apply_modality_lut(arr, ds)
  except TypeError:
    # If rescale slope and intercept are not defined, pydicom raises a
    # TypeError. pixel data is not altered by the LUT, we can use the original
    # pixel array.
    pixel_array = arr
  # pytype: enable=module-attr
  # Rescale imaging dynamic range to be across the full bit range.
  pixel_array = _rescale_cxr_dynamic_range(pixel_array)
  if ds.PhotometricInterpretation == MONOCHROME1:
    return np.iinfo(pixel_array.dtype).max - pixel_array
  return pixel_array


def _default_ct_volume_window_op(unused_dcm: pydicom.Dataset) -> RGBWindow:
  """Returns default window operation for CT imaging."""
  return _MEDGEMMA_1_CT_DEFAULT_WINDOW


def _default_mri_volume_window_op(
    unused_dcm: pydicom.Dataset,
) -> NopImageTransform:
  """Returns default window operation for MRI imaging."""
  return NopImageTransform()


def rescale_ct_imaging(arr: np.ndarray, ds: pydicom.FileDataset) -> np.ndarray:
  """Rescales CT to HU."""
  # Not applying ModalityLUTSequence on CT images.
  has_rescale_slope = 'RescaleSlope' in ds and isinstance(
      ds.RescaleSlope, float | int
  )
  has_rescale_intercept = 'RescaleIntercept' in ds and isinstance(
      ds.RescaleIntercept, float | int
  )
  if has_rescale_slope and has_rescale_intercept:
    pixel_array = arr.astype(np.float64) * float(ds.RescaleSlope)
    pixel_array += float(ds.RescaleIntercept)
    return np.round(pixel_array, 0).astype(np.int32)
  elif has_rescale_slope or has_rescale_intercept:
    raise data_accessor_errors.DicomError(
        'DICOM instance is missing RescaleSlope or RescaleIntercept tags.'
    )
  else:
    return arr


def norm_ct_imaging(
    window: Optional[ImageTransform],
    arr: np.ndarray,
    ds: pydicom.FileDataset,
) -> np.ndarray:
  """Applies data handling from pydicom."""
  if window is None:
    # This is a currently largely a internal error. At the time of writing it
    # is expected that all processed radiology modalities will use a default
    # defined windowing operation. In the future it is expected that we will
    # support user defined windowing operations. When that occures this
    # would become a user facing error.
    raise data_accessor_errors.InvalidRequestFieldError(
        'Radiology volume imaging is missing a windowing configuration.'
    )
  # windowing will normalize imaging to uint16.
  # with dynamic range scaled across the windowed range.
  return window.apply(rescale_ct_imaging(arr, ds))


def validate_samples_per_pixel(dcm: pydicom.FileDataset) -> None:
  """Validates samples per pixel metadata."""
  try:
    if dcm.SamplesPerPixel not in _SUPPORTED_SAMPLES_PER_PIXEL:
      raise data_accessor_errors.DicomError(
          'DICOM instance contains unsupported number of samples per pixel;'
          f' expected: {_SUPPORTED_SAMPLES_PER_PIXEL} found:'
          f' {dcm.SamplesPerPixel}.'
      )
  except (ValueError, AttributeError) as _:
    raise data_accessor_errors.DicomError(
        'DICOM instance missing SamplesPerPixel metadata.'
    )


def validate_samples_per_pixel_and_photometric_interpretation_match(
    dcm: pydicom.FileDataset,
) -> None:
  """Validates samples per pixel and photometric interpretation."""
  if (
      dcm.SamplesPerPixel == 1
      and dcm.PhotometricInterpretation
      not in _SINGLE_CHANNEL_PHOTOMETRIC_INTERPRETATION
  ):
    raise data_accessor_errors.DicomError(
        'DICOM instance has 1 sample per pixel but contains multichannel'
        ' PhotometricInterpretation.'
    )


def validate_unencapsulated_photometric_interpretation(
    dcm: pydicom.FileDataset,
) -> None:
  """Validates pixel unencapsulated pixel encoding pixel encoding."""
  try:
    photometric_interpretation = dcm.PhotometricInterpretation
  except (ValueError, AttributeError) as exp:
    raise data_accessor_errors.DicomError(
        'PhotometricInterpretation is required for DICOM images.'
    ) from exp
  if (
      photometric_interpretation
      not in _SUPPORTED_UNENCAPSULATED_PHOTOMETRIC_INTERPRETATIONS
  ):
    raise data_accessor_errors.DicomError(
        'DICOM image encoded using unsupported PhotometricInterpretation:'
        f' {photometric_interpretation}.'
    )


# Define default modality windowing operations
_DEFAULT_MODALITY_IMAGE_TRANSFORMS = {
    _MODALITY.CR: ModalityDefaultImageTransform(_default_cxr_window_op),
    _MODALITY.DX: ModalityDefaultImageTransform(_default_cxr_window_op),
    _MODALITY.CT: ModalityDefaultImageTransform(_default_ct_volume_window_op),
    _MODALITY.MR: ModalityDefaultImageTransform(_default_mri_volume_window_op),
}


def _get_dicom_modality(dcm: pydicom.FileDataset) -> str:
  """Returns modality of DICOM instance."""
  try:
    modality = dcm.Modality
    if modality:
      return modality
  except (AttributeError, ValueError, TypeError) as _:
    pass
  modality = dicom_source_utils.infer_modality_from_sop_class_uid(
      dcm.SOPClassUID
  )
  if modality:
    return modality
  raise data_accessor_errors.DicomError(
      'DICOM missing a modality tag metadata with a defined value.'
  )


def _get_modality_image_transform(
    dcm: pydicom.FileDataset,
    modality: str,
    modality_default_image_transform: Mapping[
        str, ModalityDefaultImageTransform
    ],
) -> Optional[ImageTransform]:
  """Get window from base request or modality default window."""
  if modality not in _WINDOWED_MODALITIES:
    return None
  modality_default_image_transform_dict = (
      _DEFAULT_MODALITY_IMAGE_TRANSFORMS
      | dict(modality_default_image_transform)
  )
  return modality_default_image_transform_dict[
      modality
  ].get_image_image_transform(dcm)


def decode_dicom_image(
    dcm: pydicom.FileDataset,
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
) -> Iterator[np.ndarray]:
  """Decode DICOM image and return decoded image bytes."""
  validate_transfer_syntax(dcm)
  validate_samples_per_pixel(dcm)
  encapsulated_dicom = dicom_frame_decoder.can_decompress_dicom_transfer_syntax(
      dcm.file_meta.TransferSyntaxUID
  )
  if encapsulated_dicom:
    image_frame_generator = _decode_encapsulated_dicom_frames(
        dcm, _get_encapsulated_dicom_frame_bytes(dcm)
    )
  else:
    validate_unencapsulated_photometric_interpretation(dcm)
    image_frame_generator = _decode_unencapsulated_dicom_frames(dcm)
  validate_samples_per_pixel_and_photometric_interpretation_match(dcm)
  for frame_image_bytes in image_frame_generator:
    decoded_frame_bytes = frame_image_bytes.decoded_frame_bytes
    if dcm.SamplesPerPixel == 1 and decoded_frame_bytes.ndim == 2:
      decoded_frame_bytes = np.expand_dims(decoded_frame_bytes, 2)
    decoded_frame_bytes = _transform_image_to_target_icc_profile(
        decoded_frame_bytes,
        frame_image_bytes.compressed_frame_bytes,
        dcm,
        target_icc_profile,
    )
    yield decoded_frame_bytes


def _decode_dicom_images(
    base_request: Mapping[str, Any],
    file_paths: abstract_handler.InputFileIterator,
    raise_error_if_invalid_dicom: bool = False,
) -> Iterator[tuple[pydicom.FileDataset, np.ndarray]]:
  """Decodes Images contained in iterator of DICOM files."""
  instance_extensions = abstract_handler.get_base_request_extensions(
      base_request
  )
  for file_path in file_paths:
    try:
      with pydicom.dcmread(file_path, specific_tags=['SOPClassUID']) as dcm:
        if (
            dcm.SOPClassUID
            == dicom_source_utils.VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID
        ):
          return
      if isinstance(file_path, io.BytesIO):
        file_path.seek(0)
      with pydicom.dcmread(file_path) as dcm:
        target_icc_profile = icc_profile_utils.get_target_icc_profile(
            instance_extensions
        )
        if 'ConcatenationUID' in dcm and dcm.ConcatenationUID:
          raise data_accessor_errors.DicomError(
              'Concatenated DICOM are not supported.'
          )
        for img in decode_dicom_image(
            dcm,
            target_icc_profile,
        ):
          yield dcm, img
        # mark file as being processed so custom iterator will now return next
        # file in sequence.
        file_paths.processed_file()
    except pydicom.errors.InvalidDicomError:
      # The handler is purposefully eating the message here.
      # if a handler fails to process the image it returns an empty iterator.
      if raise_error_if_invalid_dicom:
        raise
      return


def _process_buffered_mri_volume(
    images: Sequence[np.ndarray],
) -> Iterator[np.ndarray]:
  """Post process buffered MRI volumes."""
  # MRI imaging is commonly expressed 16-bit values. Unlike CT the imaging is
  # typically not captured in a calibrated acquision, i.e. the voxel values have
  # reliative meaning only. MRI imaging is commonly transformed for
  # visualization by scaling the values across the dynamic range.
  if not images:
    return
  output_dtype = np.uint8
  max_dtype_value = np.iinfo(output_dtype).max
  min_val = np.min(images)
  max_delta = np.max(images) - min_val
  for image in images:
    image = image.astype(np.float64)
    if max_delta == 0:
      image[...] = max_dtype_value
    else:
      image -= min_val
      image /= max_delta
      image *= max_dtype_value
      np.round(image, 0, out=image)
    yield image.astype(output_dtype)


def _same_acquisition(
    buffered_dicom: pydicom.FileDataset, dcm: pydicom.FileDataset
) -> bool:
  """Returns true if the two dicoms are from the same acquisition."""
  for kw in _DICOM_TAG_KEYWORDS_EQUAL_VALUE_FOR_INSTANCES_IN_SAME_ACQUISITION:
    if buffered_dicom.get(kw) != dcm.get(kw):
      return False
  return True


class _DicomImageQueue:
  """Wraps a interator of DICOM images in a Queue."""

  def __init__(
      self, dicom_images: Iterator[tuple[pydicom.FileDataset, np.ndarray]]
  ):
    self._dicom_images = dicom_images
    self._has_images = True
    try:
      self._buffered_output = next(self._dicom_images)
    except StopIteration:
      self._has_images = False

  def has_images(self) -> bool:
    return self._has_images

  def peek_image(self) -> tuple[pydicom.FileDataset, np.ndarray]:
    if not self._has_images:
      raise ValueError('No more images to peek.')
    return self._buffered_output

  def pop_image(self) -> None:
    try:
      self._buffered_output = next(self._dicom_images)
    except StopIteration:
      self._has_images = False


def _process_mri_modality_images(
    dicom_images: _DicomImageQueue,
) -> abstract_data_accessor.DataAcquisition[np.ndarray]:
  """Yields transformed MRI imaging from same acquisition."""
  buffered_dicom, img = dicom_images.peek_image()
  dicom_images.pop_image()
  buffered_images = [img]
  while dicom_images.has_images():
    dcm, img = dicom_images.peek_image()
    modality = _get_dicom_modality(dcm)
    dicom_source_utils.validate_modality_supported(modality)
    if dcm.Modality != _MODALITY.MR or not _same_acquisition(
        buffered_dicom, dcm
    ):
      return abstract_data_accessor.DataAcquisition(
          abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
          _process_buffered_mri_volume(buffered_images),
      )
    buffered_images.append(img)
    dicom_images.pop_image()
  return abstract_data_accessor.DataAcquisition(
      abstract_data_accessor.AccessorDataSource.DICOM_MRI_VOLUME,
      _process_buffered_mri_volume(buffered_images),
  )


def _transformed_non_mri_image(
    dcm: pydicom.FileDataset,
    modality: str,
    img: np.ndarray,
    modality_default_image_transform: Mapping[
        str, ModalityDefaultImageTransform
    ],
) -> np.ndarray:
  """Returns transformed imaging from non-MRI modalities."""
  image_transform = _get_modality_image_transform(
      dcm, modality, modality_default_image_transform
  )
  if img.ndim == 3 and img.shape[2] == 1:
    if modality in _CXR_MODALITIES:
      return norm_cxr_imaging(img, dcm)
    elif modality == _MODALITY.CT:
      return norm_ct_imaging(image_transform, img, dcm)
  return img


def _process_non_mri_modality_images(
    modality_default_image_transform: Mapping[
        str, ModalityDefaultImageTransform
    ],
    dicom_images: _DicomImageQueue,
) -> Iterator[np.ndarray]:
  """Yields transformed non-MRI imaging from same acquisition."""
  buffered_dicom, img = dicom_images.peek_image()
  modality = _get_dicom_modality(buffered_dicom)
  dicom_source_utils.validate_modality_supported(modality)
  yield _transformed_non_mri_image(
      buffered_dicom, modality, img, modality_default_image_transform
  )
  dicom_images.pop_image()
  while dicom_images.has_images():
    dcm, img = dicom_images.peek_image()
    modality = _get_dicom_modality(dcm)
    if not _same_acquisition(buffered_dicom, dcm):
      return
    dicom_source_utils.validate_modality_supported(modality)
    yield _transformed_non_mri_image(
        dcm, modality, img, modality_default_image_transform
    )
    dicom_images.pop_image()


def _modality_norm(
    modality_default_image_transform: Mapping[
        str, ModalityDefaultImageTransform
    ],
    dicom_images: Iterator[tuple[pydicom.FileDataset, np.ndarray]],
) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
  """Post process DICOM images and yields blocks image from same acquisition."""
  dicom_images = _DicomImageQueue(dicom_images)
  while dicom_images.has_images():
    dcm, _ = dicom_images.peek_image()
    modality = _get_dicom_modality(dcm)
    dicom_source_utils.validate_modality_supported(modality)
    if modality == _MODALITY.MR:
      yield _process_mri_modality_images(dicom_images)
    else:
      match modality:
        case _MODALITY.CT:
          image_source = (
              abstract_data_accessor.AccessorDataSource.DICOM_CT_VOLUME
          )
        case _MODALITY.DX | _MODALITY.CR:
          image_source = (
              abstract_data_accessor.AccessorDataSource.DICOM_CXR_IMAGES
          )
        case _MODALITY.GM | _MODALITY.SM:
          # general DICOM accessor excludes WSI DICOM images.
          # so we can assume MICROSCOPY_IMAGES
          image_source = (
              abstract_data_accessor.AccessorDataSource.DICOM_MICROSCOPY_IMAGES
          )
        case _MODALITY.XC:
          image_source = (
              abstract_data_accessor.AccessorDataSource.DICOM_EXTERNAL_CAMERA_IMAGES
          )
        case _:
          raise data_accessor_errors.DicomError(
              f'Unsupported DICOM modality: {modality}.'
          )
      yield abstract_data_accessor.DataAcquisition(
          image_source,
          _process_non_mri_modality_images(
              modality_default_image_transform, dicom_images
          ),
      )


def _generate_acquisition_images(
    instance_extensions: Mapping[str, Any],
    patch_coordinates: Sequence[patch_coordinate_module.PatchCoordinate],
    decoded_images: Iterator[np.ndarray],
) -> Iterator[np.ndarray]:
  """Process images from same acquisition."""
  patch_required_to_be_fully_in_source_image = (
      patch_coordinate_module.patch_required_to_be_fully_in_source_image(
          instance_extensions
      )
  )
  resize_image_dimensions = image_dimension_utils.get_resize_image_dimensions(
      instance_extensions
  )
  for decoded_image_bytes in decoded_images:
    if resize_image_dimensions is not None:
      decoded_image_bytes = image_dimension_utils.resize_image_dimensions(
          decoded_image_bytes, resize_image_dimensions
      )
    if not patch_coordinates:
      yield decoded_image_bytes
      continue
    image_shape = image_dimension_utils.ImageDimensions(
        width=decoded_image_bytes.shape[1],
        height=decoded_image_bytes.shape[0],
    )
    for pc in patch_coordinates:
      if patch_required_to_be_fully_in_source_image:
        pc.validate_patch_in_dim(image_shape)
      yield patch_coordinate_module.get_patch_from_memory(
          pc, decoded_image_bytes
      )


class GenericDicomHandler(abstract_handler.AbstractHandler):
  """Reads a generic DICOM image from file system. Returns None on failure."""

  def __init__(
      self,
      modality_default_image_transform: Optional[
          Mapping[str, ModalityDefaultImageTransform]
      ] = None,
      raise_error_if_invalid_dicom: bool = False,
  ):
    super().__init__()
    self._raise_error_if_invalid_dicom = raise_error_if_invalid_dicom
    self._modality_default_image_transform = (
        modality_default_image_transform
        if modality_default_image_transform is not None
        else {}
    )

  def process_files(
      self,
      instance_patch_coordinates: Sequence[
          patch_coordinate_module.PatchCoordinate
      ],
      base_request: Mapping[str, Any],
      file_paths: abstract_handler.InputFileIterator,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    for data_source in _modality_norm(
        self._modality_default_image_transform,
        _decode_dicom_images(
            base_request,
            file_paths,
            raise_error_if_invalid_dicom=self._raise_error_if_invalid_dicom,
        ),
    ):
      yield abstract_data_accessor.DataAcquisition(
          data_source.acquision_data_source,
          _generate_acquisition_images(
              abstract_handler.get_base_request_extensions(base_request),
              instance_patch_coordinates,
              data_source.acquision_data_source_iterator,
          ),
      )
