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
# ==============================================================================
"""Determines the SOPClassUIDs of a DICOM data for modality specfic processing."""

import collections
import dataclasses
import enum
from typing import Any, Mapping, Sequence

from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from ez_wsi_dicomweb.ml_toolkit import tags

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import data_accessor_definition_utils
from data_accessors.utils import json_validation_utils

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_EZ_WSI_STATE = 'ez_wsi_state'

# DICOM VL Microscopy SOPClassUIDs
# https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_i.4.html
VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'
_VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.2'
_VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID = (
    '1.2.840.10008.5.1.4.1.1.77.1.3'
)

DICOM_MICROSCOPIC_IMAGE_IODS = frozenset([
    _VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
    _VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
])

DICOM_MICROSCOPY_IODS = frozenset([
    VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID,
    _VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
    _VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
])


class MODALITY:
  """Modality Coded Values."""

  CR = 'CR'  # Computed Radiography
  DX = 'DX'  # Digital X-Ray
  GM = 'GM'  # General Microscopy
  SM = 'SM'  # Slide Microscopy
  XC = 'XC'  # External Camera
  CT = 'CT'  # Computed Tomography
  MR = 'MR'  # Magnetic Resonance


CT_AND_MRI_MODALITIES = (MODALITY.CT, MODALITY.MR)
CXR_MODALITIES = (MODALITY.CR, MODALITY.DX)
MICROSCOPY_MODALITIES = (MODALITY.SM, MODALITY.GM)
_CT_SOP_CLASS_UIDS = frozenset([
    '1.2.840.10008.5.1.4.1.1.2',
    '1.2.840.10008.5.1.4.1.1.2.1',
    '1.2.840.10008.5.1.4.1.1.2.2',
])

_MR_SOP_CLASS_UIDS = frozenset([
    '1.2.840.10008.5.1.4.1.1.4',
    '1.2.840.10008.5.1.4.1.1.4.1',
    '1.2.840.10008.5.1.4.1.1.4.4',
])
_SM_SOP_CLASS_UIDS = frozenset([
    _VL_SLIDE_COORDINATES_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
    _VL_MICROSCOPIC_IMAGE_SOP_CLASS_UID,
    VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID,
])

_DX_SOP_CLASS_UIDS = frozenset([
    '1.2.840.10008.5.1.4.1.1.1.1',
    '1.2.840.10008.5.1.4.1.1.1.1.1',
    '1.2.840.10008.5.1.4.1.1.1.2',
    '1.2.840.10008.5.1.4.1.1.1.2.1',
    '1.2.840.10008.5.1.4.1.1.1.3',
    '1.2.840.10008.5.1.4.1.1.1.3.1',
])


#  Start: Utility functions to identify a single CT or MRI volume
#  if given a series


def _filter_image_type(
    dicom_series_metadata: Sequence[dicom_web_interface.DicomObject],
) -> Sequence[dicom_web_interface.DicomObject]:
  """Filters a list of DICOM instances by image type."""
  missing_image_type = []
  derived_metadata = []
  non_derived_metadata = []
  for i_md in dicom_series_metadata:
    image_type = i_md.get_list_value(tags.IMAGE_TYPE, [])
    if not image_type:
      missing_image_type.append(i_md)
    elif 'DERIVED' in image_type:
      derived_metadata.append(i_md)
    else:
      non_derived_metadata.append(i_md)
  if non_derived_metadata:
    return non_derived_metadata
  if derived_metadata:
    return derived_metadata
  return missing_image_type


def _filter_acquisition_number(
    dicom_series_metadata: Sequence[dicom_web_interface.DicomObject],
) -> Sequence[dicom_web_interface.DicomObject]:
  """Return metadata for single acquisition from a list of DICOM metadata.

  Behavior:
    1. returns acquisition with most instances.
    2. If acquisition number is are not defined returns list of all instances.

  Args:
    dicom_series_metadata: A list of DICOM instances metadata.

  Returns:
    A list of DICOM instances metadata.
  """
  acquisitions = collections.defaultdict(list)
  for instance_metadata in dicom_series_metadata:
    acquisition_number = instance_metadata.get_value(tags.ACQUISITION_NUMBER)
    if acquisition_number is not None:
      acquisitions[acquisition_number].append(instance_metadata)
  if acquisitions:
    return acquisitions[max(acquisitions, key=lambda k: len(acquisitions[k]))]
  return dicom_series_metadata


def _sort_by_acquisition_time(
    dicom_series_metadata: Sequence[dicom_web_interface.DicomObject],
) -> Sequence[dicom_web_interface.DicomObject]:
  """Sorts a list of DICOM instances by acquisition time."""
  for required_tags in [
      (tags.ACQUISITION_DATE_TIME,),
      (tags.ACQUISITION_DATE, tags.ACQUISITION_TIME),
  ]:
    tag_list = []
    for metadata in dicom_series_metadata:
      for tag in required_tags:
        if metadata.get_value(tag) is None:
          break
      else:
        tag_list.append(metadata)
    if tag_list:
      break
  else:
    return dicom_series_metadata
  return sorted(
      tag_list,
      key=lambda x: ''.join([str(x.get_value(t)) for t in required_tags]),
  )


def _split_instances_by_dicom_tag(
    dicom_series_metadata: Sequence[dicom_web_interface.DicomObject],
    tag: tags.DicomTag,
) -> tuple[
    Sequence[dicom_web_interface.DicomObject],
    Sequence[dicom_web_interface.DicomObject],
]:
  """Splits a list of DICOM instances into lists with and without a DICOM tag."""
  has_tag = []
  missing_tag = []
  for instance in dicom_series_metadata:
    if instance.get_list_value(tag) is not None:
      has_tag.append(instance)
    else:
      missing_tag.append(instance)
  return has_tag, missing_tag


def _remove_instances_with_duplicate_instance_numbers(
    dicom_series_metadata: Sequence[dicom_web_interface.DicomObject],
) -> Sequence[dicom_web_interface.DicomObject]:
  """Removes instances with duplicate instance numbers."""
  returned_list_dicoms = []
  instance_numbers = set()
  sorted_metadata = _sort_by_acquisition_time(dicom_series_metadata)
  for instance_metadata in sorted_metadata:
    instance_number = instance_metadata.get_value(tags.INSTANCE_NUMBER)
    if instance_number is not None and instance_number not in instance_numbers:
      instance_numbers.add(instance_number)
      returned_list_dicoms.append(instance_metadata)
  return returned_list_dicoms if returned_list_dicoms else sorted_metadata


def _sort_by_slice_position(obj: dicom_web_interface.DicomObject) -> int:
  img_pos_pat_zcoord_index = 0
  try:
    return obj.get_list_value(tags.IMAGE_POSITION_PATIENT)[
        img_pos_pat_zcoord_index
    ]
  except (IndexError, TypeError):
    return 0


def _identify_dicom_series_instances_for_single_ct_or_mri_volume(
    dicom_series_metadata: Sequence[dicom_web_interface.DicomObject],
) -> Sequence[dicom_web_interface.DicomObject]:
  """Identifies DICOM series instances for a single CT or MRI volume."""
  dicom_series_metadata = _filter_image_type(dicom_series_metadata)
  dicom_series_metadata = _filter_acquisition_number(dicom_series_metadata)
  dicom_series_metadata, metadata_no_slice_position = (
      _split_instances_by_dicom_tag(
          dicom_series_metadata, tags.IMAGE_POSITION_PATIENT
      )
  )
  dicom_series_metadata = _remove_instances_with_duplicate_instance_numbers(
      dicom_series_metadata
      if dicom_series_metadata
      else metadata_no_slice_position
  )
  return sorted(
      dicom_series_metadata,
      key=_sort_by_slice_position,
  )


#  End: Utility functions to identify a single CT or MRI volume
#  if given a series


def validate_modality_supported(modality: str) -> None:
  """Validates DICOM modality is supported."""
  if modality in CXR_MODALITIES:
    return
  if modality in MICROSCOPY_MODALITIES:
    return
  if modality in MODALITY.XC:
    return
  if modality == MODALITY.CT:
    return
  if modality == MODALITY.MR:
    return
  raise data_accessor_errors.DicomError(
      f'DICOM encodes a unsupported Modality; Modality: {modality}.'
  )


class DicomDataSourceEnum(enum.Enum):
  """Enum for DICOM data source type."""

  SLIDE_MICROSCOPY_IMAGE = 'slide_microscope_image'
  GENERIC_DICOM = 'generic_dicom'


@dataclasses.dataclass(frozen=True)
class _DicomSourceType:
  dicom_source_type: DicomDataSourceEnum
  dicom_instances_metadata: Sequence[dicom_web_interface.DicomObject]


def _get_vl_whole_slide_microscopy_image_instances(
    selected_instance: dicom_web_interface.DicomObject,
    instances: Sequence[dicom_web_interface.DicomObject],
) -> Sequence[dicom_web_interface.DicomObject]:
  """Returns DICOM instances for VL whole slide microscopy image pyramid layer."""
  concatination_uid = selected_instance.get_value(tags.CONCATENATION_UID)
  if concatination_uid is None:
    return [selected_instance]
  found_instances = []
  for i in instances:
    found_concatination_uid = i.get_value(tags.CONCATENATION_UID)
    if (
        found_concatination_uid is not None
        and found_concatination_uid == concatination_uid
    ):
      found_instances.append(i)
  return found_instances


def infer_modality_from_sop_class_uid(sop_class_uid: str) -> str:
  """Infers modality from SOP Class UID."""
  if sop_class_uid in _CT_SOP_CLASS_UIDS:
    return MODALITY.CT
  if sop_class_uid in _MR_SOP_CLASS_UIDS:
    return MODALITY.MR
  if sop_class_uid in _SM_SOP_CLASS_UIDS:
    return MODALITY.SM
  if sop_class_uid == '1.2.840.10008.5.1.4.1.1.1':
    return MODALITY.CR
  if sop_class_uid in _DX_SOP_CLASS_UIDS:
    return MODALITY.DX
  return ''


def get_dicom_source_type_and_instance_metadata(
    auth: credential_factory.AbstractCredentialFactory,
    instance: Mapping[str, Any],
) -> _DicomSourceType:
  """Returns DICOM modality and instnace metadata for requested path.

  Args:
    auth: Authentication credentials for DICOMweb access.
    instance: A JSON encoded DICOMweb instance.

  Returns:
    A _DicomSourceType object containing the DICOM data source type and the
    instance metadata for the requested path.
  """
  dcm_paths = data_accessor_definition_utils.parse_dicom_source(instance)
  extensions = instance.get(_InstanceJsonKeys.EXTENSIONS, {})
  ez_wsi_state = json_validation_utils.validate_str_key_dict(
      extensions.get(_EZ_WSI_STATE, {})
  )
  if ez_wsi_state:
    return _DicomSourceType(DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE, [])
  dwi = dicom_web_interface.DicomWebInterface(auth)
  series_path = dcm_paths[0].GetSeriesPath()
  try:
    instances = dwi.get_instances(series_path, includefield='all')
  except (
      ez_wsi_errors.HttpForbiddenError,
      ez_wsi_errors.HttpUnauthorizedError,
  ) as exp:
    raise data_accessor_errors.InvalidCredentialsError(
        'Credentials not accepted for listing DICOM instances for path: '
        f'{series_path}.'
    ) from exp
  except ez_wsi_errors.HttpError as exp:
    raise data_accessor_errors.HttpError(
        f'HTTP error with status {exp.status_code} when listing DICOM instances'
        f' for path: {series_path}.'
    ) from exp
  if not instances:
    raise data_accessor_errors.InvalidRequestFieldError(
        f'No instances found for DICOM path: {series_path}.'
    )
  # if dcm_path defines instances then limit series metadata to those instances.
  if dcm_paths[0].type == dicom_path.Type.INSTANCE:
    path_defined_sop_instances = {i.instance_uid: i for i in dcm_paths}
    found_instances = []
    for i in instances:
      if i.sop_instance_uid not in path_defined_sop_instances:
        continue
      if i.sop_class_uid != VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID:
        found_instances.append(i)
      else:
        # Add VL instances implicity defined by concatenation uid.
        found_instances.extend(
            _get_vl_whole_slide_microscopy_image_instances(i, instances)
        )
    instances = found_instances
  else:
    path_defined_sop_instances = {}

  # determine modality of instances
  modalities = set()
  instances_defining_known_modality = []
  for i in instances:
    modality = i.get_value(tags.MODALITY)
    if modality is None:
      modality = infer_modality_from_sop_class_uid(i.sop_class_uid)
    if modality:
      modalities.add(modality)
      instances_defining_known_modality.append(i)
  instances = instances_defining_known_modality
  if len(modalities) > 1:
    raise data_accessor_errors.InvalidRequestFieldError(
        'DICOMweb URI defines series that contains more than one modality.'
    )
  if not modalities:
    raise data_accessor_errors.InvalidRequestFieldError(
        'DICOMweb URI does not define a instance with a recognized modality.'
    )
  modality = modalities.pop().upper()
  validate_modality_supported(modality)

  if (
      modality != MODALITY.SM
      and dcm_paths[0].type == dicom_path.Type.SERIES
      and modality in CT_AND_MRI_MODALITIES
  ):
    # if CT or MRI volume then identify single volume is defined by series.
    # Identify set of instances that define a single CT or MRI volume.
    instances = _identify_dicom_series_instances_for_single_ct_or_mri_volume(
        instances
    )

  if path_defined_sop_instances:
    # Test if instances are defined by sop_instance_uid then that the instance
    # metadata being returned is a super set of the defined intances.
    identified_sop_instances = {i.sop_instance_uid for i in instances}
    for path_sop_instance in path_defined_sop_instances:
      if path_sop_instance not in identified_sop_instances:
        raise data_accessor_errors.InvalidRequestFieldError(
            f'DICOMweb URI defines invalid sop instance: {path_sop_instance}.'
        )

  if modality != MODALITY.SM:
    return _DicomSourceType(DicomDataSourceEnum.GENERIC_DICOM, instances)
  sop_class_uids = {i.sop_class_uid for i in instances}
  if (
      len(sop_class_uids) > 1
      and VL_WHOLE_SLIDE_MICROSCOPY_IMAGE_SOP_CLASS_UID in sop_class_uids
  ):
    raise data_accessor_errors.InvalidRequestFieldError(
        'DICOMweb URI define both VL_WHOLE_SLIDE_MICROSCOPY_IMAGE IOD and other'
        ' IODs.'
    )
  return _DicomSourceType(
      DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
      instances,
  )
