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
"""Data accessor for data DICOM WSI stored within a DICOM store."""

from __future__ import annotations

from concurrent import futures
import contextlib
import copy
import dataclasses
import functools
import time
import typing
from typing import Any, Iterator, Mapping, Optional, Sequence, Union

from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb import local_dicom_slide_cache
from ez_wsi_dicomweb import local_dicom_slide_cache_types
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import numpy as np
from PIL import ImageCms

from data_accessors import abstract_data_accessor
from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.dicom_wsi import configuration
from data_accessors.dicom_wsi import data_accessor_definition
from data_accessors.dicom_wsi import ez_wsi_cloud_logging_adapter
from data_accessors.dicom_wsi import icc_profile_cache
from data_accessors.utils import icc_profile_utils
from data_accessors.utils import image_dimension_utils
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module
from serving.logging_lib import cloud_logging_client


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_MAX_DICOM_LEVEL_DOWNSAMPLE = 8.0

_SERIES_OR_INSTANCE_DICOM_PATH_TYPES = (
    dicom_path.Type.SERIES,
    dicom_path.Type.INSTANCE,
)


@dataclasses.dataclass(frozen=True)
class _LocalData:
  """Local data for DICOM WSI."""

  patches: Sequence[dicom_slide.DicomPatch]
  icc_profile_transformation: Optional[ImageCms.ImageCmsTransform]
  frame_cache: local_dicom_slide_cache.InMemoryDicomSlideCache


def _normalized_patch_channels(patch: np.ndarray) -> np.ndarray:
  """Normalize monochrome and RGBA imaging to RGB."""
  if patch.ndim == 3 and patch.shape[-1] == 3:
    return patch
  if patch.ndim == 2:
    patch = np.expand_dims(patch, axis=-1)
  if patch.ndim == 3 and patch.shape[-1] == 1:
    mem_shape = list(patch.shape)
    mem_shape[-1] = 3
    mem = np.zeros(mem_shape, dtype=patch.dtype)
    mem[..., np.arange(3)] = patch[...]
    return mem
  if patch.ndim == 3 and patch.shape[-1] == 4:
    return patch[..., :3]
  raise ez_wsi_errors.PatchEmbeddingDimensionError(
      f'Unexpected patch shape {patch.shape}.'
  )


def _validate_dicom_image_accessor(
    path: str,
    settings: configuration.ConfigurationSettings,
) -> None:
  """Validates that DICOM image accessor is in approved list if defined."""
  if settings.approved_dicom_stores is None:
    return
  test_path = path.lower()
  for val in settings.approved_dicom_stores:
    if test_path.startswith(val.lower()):
      return
  msg = f'DICOM store {str(path)} is not in the allowed list.'
  cloud_logging_client.info(msg, {'connection_attempted': path})
  raise data_accessor_errors.UnapprovedDicomStoreError(msg)


def _get_ez_wsi_state(extensions: Mapping[str, Any]) -> Mapping[str, Any]:
  """Returns optional state for EZ-WSI DICOM web."""
  try:
    return json_validation_utils.validate_str_key_dict(
        extensions.get(_InstanceJsonKeys.EZ_WSI_STATE, {})
    )
  except json_validation_utils.ValidationError as exp:
    msg = 'Invalid EZ-WSI state metadata.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.EzWsiStateError(msg) from exp


def _fetch_image_bytes(
    patch: dicom_slide.DicomPatch,
    icc_profile_transformation: Optional[ImageCms.ImageCmsTransform],
) -> np.ndarray:
  """Returns patch bytes for DICOM imaging."""
  try:
    image_data = patch.image_bytes(icc_profile_transformation)
  except (
      ez_wsi_errors.HttpForbiddenError,
      ez_wsi_errors.HttpUnauthorizedError,
  ) as exp:
    if isinstance(patch.source, dicom_slide.DicomSlide):
      msg = f'Invalid credentials reading DICOM image: {str(patch.source.path)}'
    else:
      msg = 'Invalid credentials accessing unrecognized data source'
    msg = f'{msg}; HTTP status code: {exp.status_code}; reason: {exp.reason}.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.InvalidCredentialsError(msg) from exp
  except ez_wsi_errors.HttpError as exp:
    if isinstance(patch.source, dicom_slide.DicomSlide):
      msg = (
          f'HTTP error occurred accessing DICOM image: {str(patch.source.path)}'
      )
    else:
      msg = 'HTTP error Coccurred accessing unrecognized data source'
    cloud_logging_client.info(msg, exp)
    msg = f'{msg}; HTTP status code: {exp.status_code}; reason: {exp.reason}.'
    raise data_accessor_errors.HttpError(msg) from exp
  return _normalized_patch_channels(image_data)


def _get_load_whole_slide_frame_ratio(extensions: Mapping[str, Any]) -> int:
  """Ratio of frames in level / requested; at which whole instance prefered."""
  try:
    return int(
        extensions.get(_InstanceJsonKeys.LOAD_WHOLE_SLIDE_FRAME_RATIO, 10)
    )
  except (ValueError, TypeError) as _:
    return 10


def _pre_load_slide_patches(
    ds: dicom_slide.DicomSlide,
    level: Union[dicom_slide.Level, dicom_slide.ResizedLevel],
    patches: Sequence[dicom_slide.DicomPatch],
    blocking: bool,
    load_whole_slide_frame_ratio: int,
) -> None:
  """Preload slide patches into frame cache."""
  if isinstance(level, dicom_slide.ResizedLevel):
    number_of_frames = level.source_level.number_of_frames
  else:
    number_of_frames = level.number_of_frames
  # ratio of frames in level to those requested is below threshold threshold
  # we preference to load the entire level.
  if number_of_frames / len(patches) <= load_whole_slide_frame_ratio:
    ds.preload_level_in_frame_cache(level, blocking=blocking)
  else:
    ds.preload_patches_in_frame_cache(patches, blocking=blocking)


def _load_slide_level(
    instance: data_accessor_definition.DicomWSIImage,
    settings: configuration.ConfigurationSettings,
    require_fully_in_source_image: bool,
    resize_level_dim: Optional[image_dimension_utils.ImageDimensions],
    target_icc_profile: Optional[ImageCms.core.CmsProfile],
    series_path_ds_level: tuple[
        dicom_path.Path,
        Union[dicom_slide.DicomSlide, dicom_slide.DicomMicroscopeImage],
        dicom_slide.Level,
    ],
) -> _LocalData:
  """Loads slide level imaging."""
  series_path, ds, level = series_path_ds_level
  # Initialize in-memory cache on slide to store frames requested in batch.
  # Optimization hints = MINIMIZE_LATENCY or MINIMIZE_DICOM_STORE_QPM
  # MINIMIZE_LATENCY: Batch load frames async. If a frame is
  # requested that is not in cache, issue an immediate request and return
  # data for the missing frame.
  # MINIMIZE_DICOM_STORE_QPM: Block, wait for cache to finish loading before
  # returning frames.

  ds.init_slide_frame_cache(
      optimization_hint=local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM
  )
  if level.pixel_spacing.is_defined:
    level_pixel_spacing = (
        f'{level.pixel_spacing.column_spacing_mm},'
        f' {level.pixel_spacing.row_spacing_mm}'
    )
  else:
    level_pixel_spacing = 'undefined'
  if instance.patch_coordinates:
    patch_coordinates = instance.patch_coordinates
  else:
    patch_coordinates = [
        patch_coordinate_module.PatchCoordinate(0, 0, level.width, level.height)
    ]
  cloud_logging_client.info(
      'Retrieved pyramid level for embedding generation.',
      {
          'level_index': level.level_index,
          'level_width': level.width,
          'level_height': level.height,
          'pixel_spacing': level_pixel_spacing,
          'frame_number_min': level.frame_number_min,
          'frame_number_max': level.frame_number_max,
          'transfer_syntax_uid': level.transfer_syntax_uid,
          'study_instance_uid': series_path.study_uid,
          'series_instance_uid': series_path.series_uid,
          'sop_instance_uids': level.get_level_sop_instance_uids(),
          _InstanceJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
              require_fully_in_source_image
          ),
      },
  )
  if resize_level_dim is not None:
    cloud_logging_client.info(
        f'Resizing image level dimensions to: {resize_level_dim}'
    )
    dicom_slide_resize_level_dim = dicom_slide.ImageDimensions(
        width_px=resize_level_dim.width,
        height_px=resize_level_dim.height,
    )
    downsample_scale_factor = max(
        level.scale_factors(dicom_slide_resize_level_dim)
    )
    if downsample_scale_factor > _MAX_DICOM_LEVEL_DOWNSAMPLE:
      msg = (
          f'Image downsampling, {round(downsample_scale_factor, 5)}X'
          ' exceeds 8X.'
      )
      cloud_logging_client.info(msg)
      raise data_accessor_errors.DicomImageDownsamplingTooLargeError(msg)
    level = level.resize(dicom_slide_resize_level_dim)
  # init patches from level
  try:
    patches = []
    for patch_coordinate in patch_coordinates:
      try:
        patches.append(
            ds.get_patch(
                level,
                x=patch_coordinate.x_origin,
                y=patch_coordinate.y_origin,
                width=patch_coordinate.width,
                height=patch_coordinate.height,
                require_fully_in_source_image=require_fully_in_source_image,
            )
        )
      except ez_wsi_errors.PatchOutsideOfImageDimensionsError as exp:
        msg = (
            f'Patch dimensions {dataclasses.asdict(patch_coordinate)} fall'
            ' outside of DICOM level pyramid imaging dimensions'
            f' ({level.width} x {level.height}).'
        )
        cloud_logging_client.info(msg, exp)
        raise data_accessor_errors.PatchOutsideOfImageDimensionsError(
            msg
        ) from exp
      except ez_wsi_errors.DicomPatchGenerationError as exp:
        msg = (
            'Can not generate patches from DICOM instances with more than'
            ' one frame and Dimension Organization Type != TILED_FULL.'
        )
        cloud_logging_client.info(msg, exp)
        raise data_accessor_errors.DicomTiledFullError(msg) from exp
    # Pre-fetch only the frames required for inference into the EZ-WSI frame
    # cache.
    fc = typing.cast(
        local_dicom_slide_cache.InMemoryDicomSlideCache,
        ds.slide_frame_cache,
    )
    fc.reset_cache_stats()
    load_whole_slide_frame_ratio = _get_load_whole_slide_frame_ratio(
        instance.extensions
    )
    if target_icc_profile is None:
      _pre_load_slide_patches(
          ds, level, patches, True, load_whole_slide_frame_ratio
      )
      icc_profile_transformation = None
    else:
      fc.optimization_hint = (
          local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_LATENCY
      )
      _pre_load_slide_patches(
          ds, level, patches, False, load_whole_slide_frame_ratio
      )
      try:
        dicom_slide_icc_profile = icc_profile_cache.get_dicom_icc_profile(
            settings.icc_profile_cache_configuration,
            slide=ds,
            slide_level=level,
        )
      except (
          ez_wsi_errors.HttpForbiddenError,
          ez_wsi_errors.HttpUnauthorizedError,
      ) as exp:
        msg = (
            'Failed to retrieve ICC profile from DICOM store. Invalid DICOM'
            f' store credentials; HTTP status code: {exp.status_code};'
            f' reason: {exp.reason}.'
        )
        cloud_logging_client.info(msg, exp)
        raise data_accessor_errors.InvalidCredentialsError(msg) from exp
      except ez_wsi_errors.HttpError as exp:
        msg = (
            'Failed to retrieve ICC profile from DICOM store. A HTTP error'
            f' occurred; HTTP status code: {exp.status_code}; reason:'
            f' {exp.reason}.'
        )
        cloud_logging_client.info(msg, exp)
        raise data_accessor_errors.HttpError(msg) from exp
      if not dicom_slide_icc_profile:
        cloud_logging_client.info(
            'DICOM slide does not have ICC profile; imaging will not be'
            ' transformed to target ICC profile.'
        )
        icc_profile_transformation = None
      else:
        icc_profile_name = (
            icc_profile_utils.get_transform_imaging_to_icc_profile_name(
                instance.extensions
            )
        )
        cloud_logging_client.info(
            'Creating ICC profile transformation to transform RGB values in'
            f' DICOM patches to {icc_profile_name} colorspace.'
        )
        icc_profile_transformation = (
            icc_profile_utils.create_icc_profile_transformation(
                dicom_slide_icc_profile, target_icc_profile
            )
        )
      fc.block_until_frames_are_loaded()
      fc.optimization_hint = (
          local_dicom_slide_cache_types.CacheConfigOptimizationHint.MINIMIZE_DICOM_STORE_QPM
      )
    return _LocalData(patches, icc_profile_transformation, fc)
  except (
      ez_wsi_errors.HttpForbiddenError,
      ez_wsi_errors.HttpUnauthorizedError,
  ) as exp:
    msg = (
        'Error retrieving DICOM patch imaging. Invalid DICOM store'
        f' credentials; HTTP status code: {exp.status_code}; reason:'
        f' {exp.reason}.'
    )
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.InvalidCredentialsError(msg) from exp
  except ez_wsi_errors.HttpError as exp:
    msg = (
        'Error retrieving DICOM patch imageing. A HTTP error occurred; HTTP'
        f' status code: {exp.status_code}; reason: {exp.reason}.'
    )
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.HttpError(msg) from exp
  except ez_wsi_errors.UnsupportedPixelFormatError as exp:
    msg = 'DICOM contains instances with imaging bits allocated != 8.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.DicomError(msg) from exp
  except ez_wsi_errors.LevelNotFoundError as exp:
    msg = 'Cannot locate expected level. Request references invalid metadata.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.LevelNotFoundError(msg) from exp
  except (
      ez_wsi_errors.CoordinateOutofImageDimensionsError,
      ez_wsi_errors.FrameNumberOutofBoundsError,
      ez_wsi_errors.InputFrameNumberOutOfRangeError,
      ez_wsi_errors.PatchIntersectionNotFoundError,
      ez_wsi_errors.SectionOutOfImageBoundsError,
  ) as exp:
    msg = (
        'Can not generate patches from DICOM instance. DICOM instance is'
        ' missing expected frames.'
    )
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.DicomError(msg) from exp
  except ez_wsi_errors.DicomInstanceReadError as exp:
    msg = 'Embedding request references a corrupt DICOM instance.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.DicomError(msg) from exp


def _load_slide_data(
    instance: data_accessor_definition.DicomWSIImage,
    settings: configuration.ConfigurationSettings,
) -> Sequence[_LocalData]:
  """Loads slide data for DICOM WSI into local memory."""
  cloud_logging_client.info('Generating embedding from DICOM.')
  require_fully_in_source_image = (
      patch_coordinate_module.patch_required_to_be_fully_in_source_image(
          instance.extensions
      )
  )
  resize_level_dim = image_dimension_utils.get_resize_image_dimensions(
      instance.extensions
  )
  ez_wsi_state = _get_ez_wsi_state(instance.extensions)
  target_icc_profile = icc_profile_utils.get_target_icc_profile(
      instance.extensions
  )
  dwi = dicom_web_interface.DicomWebInterface(instance.credential_factory)
  ds_and_series_path_and_level_data_to_load = []
  last_sop_instance_defined_level = None
  for dicomweb_path in instance.dicomweb_paths:
    if dicomweb_path.type not in _SERIES_OR_INSTANCE_DICOM_PATH_TYPES:
      msg = f'DICOM path is invalid: {dicomweb_path}.'
      cloud_logging_client.info(msg, {'slide_path_requested': dicomweb_path})
      raise data_accessor_errors.DicomPathError(
          f'Slide path is invalid: {dicomweb_path}.'
      )
    series_path = dicomweb_path.GetSeriesPath()
    if dicomweb_path.type == dicom_path.Type.INSTANCE:
      sop_instance_uid = dicomweb_path.instance_uid
    else:
      sop_instance_uid = None
    _validate_dicom_image_accessor(str(series_path), settings)
    start_time = time.time()
    try:
      series_images = dicom_slide.DicomMicroscopeSeries(
          dwi=dwi,
          path=series_path,
          enable_client_slide_frame_decompression=True,
          json_metadata=ez_wsi_state,
          instance_uids=None
          if sop_instance_uid is None
          else [sop_instance_uid],
          logging_factory=ez_wsi_cloud_logging_adapter.EZWSILoggingInterfaceFactory(
              cloud_logging_client.get_log_signature()
          ),
          dicom_instances_metadata=instance.dicom_instances_metadata,
      )
      if series_images.dicom_slide is not None:
        if series_images.dicom_microscope_image is not None:
          msg = (
              'Cannot return for DICOM images for '
              ' VL_Whole_Slide_Microscopy_Images and DICOM Microscopic_Images'
              ' in the same request.'
          )
          cloud_logging_client.info(msg)
          raise data_accessor_errors.DicomError(msg)
        ds = series_images.dicom_slide
      elif series_images.dicom_microscope_image is not None:
        ds = series_images.dicom_microscope_image
      else:
        msg = f'Could not find DICOM imaging; path: {series_path}.'
        cloud_logging_client.info(msg)
        raise data_accessor_errors.DicomPathError(msg)
    except ez_wsi_errors.DicomSlideInitError as exp:
      msg = f'DICOM metadata error; Path: {series_path}; {exp}'
      cloud_logging_client.info(msg, exp)
      raise data_accessor_errors.DicomError(msg) from exp
    except (
        ez_wsi_errors.SlideLevelContainsInstancesWithDifferentTransferSyntaxUIDError
    ) as exp:
      msg = (
          'All DICOM instances in a pyramid level are required to have same'
          ' TransferSyntaxUID.'
      )
      cloud_logging_client.info(msg, exp)
      raise data_accessor_errors.DicomError(msg) from exp
    except (
        ez_wsi_errors.DicomTagNotFoundError,
        ez_wsi_errors.InvalidDicomTagError,
    ) as exp:
      msg = f'DICOM instance missing required tags; Path: {series_path}; {exp}'
      cloud_logging_client.info(msg, exp)
      raise data_accessor_errors.DicomError(msg) from exp
    except (
        ez_wsi_errors.InvalidSlideJsonMetadataError,
        ez_wsi_errors.SlidePathDoesNotMatchJsonMetadataError,
        ez_wsi_errors.UnexpectedDicomObjectInstanceError,
    ) as exp:
      msg = 'Error decoding embedding request JSON metadata.'
      cloud_logging_client.error(
          msg,
          {'slide_path_requested': series_path},
          exp,
      )
      # If this occurs it would be a bug in the ez-wsi interface.
      raise data_accessor_errors.EzWsiStateError(msg) from exp
    cloud_logging_client.info(
        f'Retrieved metadata for slide: {series_path};'
        f' {time.time() - start_time} (sec).',
        {'slide_path_requested': series_path},
    )

    if sop_instance_uid is not None:
      # If SOP Instance is defined then load SOP Instance imaging
      level = ds.get_instance_level(sop_instance_uid)
      if level is None:
        msg = (
            f'{sop_instance_uid} is not part of DICOM WSI pyramid; path:'
            f' {series_path}.'
        )
        cloud_logging_client.info(msg)
        raise data_accessor_errors.LevelNotFoundError(msg)
      levels = [level]
      if (
          last_sop_instance_defined_level is not None
          and last_sop_instance_defined_level == level
      ):
        # if iterating over a list of sop instances then skip sequentially
        # defined instances which define the same pyrmaid level,
        # i.e. concatenated instances.
        continue
      last_sop_instance_defined_level = level
    else:
      last_sop_instance_defined_level = None
      # slide id is defined by series not instances.
      if isinstance(ds, dicom_slide.DicomSlide):
        # If series is defined and WSI VL Image then if patch coordiantes are
        # defined use the high magnification level for imaging.
        try:
          levels = [ds.native_level]
        except ez_wsi_errors.LevelNotFoundError:
          levels = []
        if not instance.patch_coordinates or not levels:
          # If patch coordinates are not defined then find find patch with
          # dimensions that are closest to the model input dimensions.
          # if no level found then defaults to available imaging.
          test_levels = [] if ds.thumbnail is None else [ds.thumbnail]
          test_levels.extend(ds.levels)
          for level in sorted(test_levels, key=lambda l: l.width * l.height):
            if (
                level.width >= settings.endpoint_input_width
                and level.height >= settings.endpoint_input_height
            ):
              levels = [level]
              break
        if not levels:
          msg = (
              f'DICOM {series_path} is missing WSI pyramid and WSI thumbnail'
              ' image.'
          )
          cloud_logging_client.info(msg)
          raise data_accessor_errors.LevelNotFoundError(msg)
      elif isinstance(ds, dicom_slide.DicomMicroscopeImage):
        # If series is defined and Microscopy Image is defined then iterator
        # over all images. Here level corresponds to a unique image.
        levels = ds.all_levels
      else:
        raise data_accessor_errors.InternalError(
            'Unsupported DICOM slide type.'
        )
    for level in levels:
      # create copy of ds to enable threaded loading of levels from the same
      # slide.
      ds_and_series_path_and_level_data_to_load.append(
          (series_path, copy.copy(ds), level)
      )
  levels_to_load = len(ds_and_series_path_and_level_data_to_load)
  if levels_to_load == 0:
    return []
  max_parallel_download_workers = max(1, settings.max_parallel_download_workers)
  if levels_to_load == 1 or max_parallel_download_workers == 1:
    return [
        _load_slide_level(
            instance,
            settings,
            require_fully_in_source_image,
            resize_level_dim,
            target_icc_profile,
            ds_and_series_path_and_level_data,
        )
        for ds_and_series_path_and_level_data in (
            ds_and_series_path_and_level_data_to_load
        )
    ]
  with futures.ThreadPoolExecutor(
      max_workers=max_parallel_download_workers
  ) as executor:
    return list(
        executor.map(
            functools.partial(
                _load_slide_level,
                instance,
                settings,
                require_fully_in_source_image,
                resize_level_dim,
                target_icc_profile,
            ),
            ds_and_series_path_and_level_data_to_load,
        )
    )


def _get_dicom_patches(local_data: _LocalData) -> Iterator[np.ndarray]:
  """Returns image patch bytes from DICOM series."""
  if local_data.icc_profile_transformation is not None:
    cloud_logging_client.info(
        'Transforming RGB values in patches to target ICC Profile.'
    )
  try:
    for patch in local_data.patches:
      yield _fetch_image_bytes(patch, local_data.icc_profile_transformation)
    cloud_logging_client.debug(
        'DICOM image retrieval stats.',
        dataclasses.asdict(local_data.frame_cache.cache_stats),
    )
  except ez_wsi_errors.UnsupportedPixelFormatError as exp:
    msg = 'DICOM contains instances with imaging bits allocated != 8.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.DicomError(msg) from exp
  except ez_wsi_errors.LevelNotFoundError as exp:
    msg = 'Cannot locate expected level. Request references invalid metadata.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.LevelNotFoundError(msg) from exp
  except (
      ez_wsi_errors.CoordinateOutofImageDimensionsError,
      ez_wsi_errors.FrameNumberOutofBoundsError,
      ez_wsi_errors.InputFrameNumberOutOfRangeError,
      ez_wsi_errors.PatchIntersectionNotFoundError,
      ez_wsi_errors.SectionOutOfImageBoundsError,
  ) as exp:
    msg = (
        'Can not generate patches from DICOM instance. DICOM instance is'
        ' missing expected frames.'
    )
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.DicomError(msg) from exp
  except ez_wsi_errors.DicomInstanceReadError as exp:
    msg = 'Embedding request references a corrupt DICOM instance.'
    cloud_logging_client.info(msg, exp)
    raise data_accessor_errors.DicomError(msg) from exp


class DicomDigitalPathologyData(
    abstract_data_accessor.AbstractDataAccessor[
        data_accessor_definition.DicomWSIImage, np.ndarray
    ]
):
  """Data accessor for data DICOM WSI stored within a DICOM store."""

  def __init__(
      self,
      instance: data_accessor_definition.DicomWSIImage,
      settings: configuration.ConfigurationSettings,
  ):
    super().__init__(instance)
    self._settings = settings
    self._local_data = []

  @contextlib.contextmanager
  def _reset_local_file_path(self, *args, **kwds):
    del args, kwds
    try:
      yield
    finally:
      self._local_data = []

  def load_data(self, stack: contextlib.ExitStack) -> None:
    """Method pre-loads data prior to data_iterator."""
    if self._local_data:
      return
    self._local_data = _load_slide_data(self.instance, self._settings)
    stack.enter_context(self._reset_local_file_path())

  def data_acquisition_iterator(
      self,
  ) -> Iterator[abstract_data_accessor.DataAcquisition[np.ndarray]]:
    for local_data in (
        self._local_data
        if self._local_data
        else _load_slide_data(self.instance, self._settings)
    ):
      if not local_data.patches:
        continue
      if isinstance(local_data.patches[0].source, dicom_slide.DicomSlide):
        source_type = (
            abstract_data_accessor.AccessorDataSource.DICOM_WSI_MICROSCOPY_PYRAMID_LEVEL
        )
      elif isinstance(
          local_data.patches[0].source, dicom_slide.DicomMicroscopeImage
      ):
        source_type = (
            abstract_data_accessor.AccessorDataSource.DICOM_MICROSCOPY_IMAGES
        )
      else:
        raise data_accessor_errors.InternalError(
            'Unsupported DICOM slide type.'
        )
      yield abstract_data_accessor.DataAcquisition(
          source_type,
          _get_dicom_patches(local_data),
      )

  def is_accessor_data_embedded_in_request(self) -> bool:
    """Returns true if data is inline with request."""
    return False
