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
"""Tests for gcs_generic data accessor ."""
import os
from typing import Iterator, Sequence

import numpy as np
import pydicom

from data_accessors import abstract_data_accessor

_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])


def testdata_path(*filename: str) -> str:
  base_path = [
      os.path.join(os.path.dirname(os.path.dirname(__file__)), 'testdata')
  ]
  base_path.extend(filename)
  return os.path.normpath(os.path.join(*base_path))


def create_test_dicom_instance(
    sop_class_uid: str,
    study_uid: str,
    series_uid: str,
    instance_uid: str,
    pixeldata: np.ndarray,
) -> pydicom.FileDataset:
  """Creates pydicom instance for testing."""
  file_meta = pydicom.dataset.FileMetaDataset()
  file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
  file_meta.MediaStorageSOPClassUID = sop_class_uid
  file_meta.MediaStorageSOPInstanceUID = instance_uid
  file_meta.ImplementationClassUID = '1.2.3'
  test_instance = pydicom.FileDataset(
      '', {}, file_meta=file_meta, preamble=b'\0' * 128
  )
  test_instance.StudyInstanceUID = study_uid
  test_instance.SeriesInstanceUID = series_uid
  test_instance.SOPInstanceUID = instance_uid
  test_instance.SOPClassUID = sop_class_uid
  test_instance.NumberOfFrames = 1
  test_instance.BitsAllocated = int(np.dtype(pixeldata.dtype).itemsize * 8)
  test_instance.InstanceNumber = 1
  test_instance.Columns = pixeldata.shape[1]
  test_instance.Rows = pixeldata.shape[0]
  test_instance.PixelRepresentation = (
      0 if np.dtype(pixeldata.dtype).kind == 'u' else 1
  )
  test_instance.BitsStored = test_instance.BitsAllocated
  test_instance.SamplesPerPixel = (
      1 if pixeldata.ndim != 3 else pixeldata.shape[2]
  )
  if test_instance.SamplesPerPixel > 1:
    test_instance.PlanarConfiguration = 0
  test_instance.HighBit = 7
  test_instance.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME']
  pixel_data = pixeldata.tobytes()
  pixel_data = pixel_data if len(pixel_data) % 2 == 0 else b'{pixel_data}0'
  test_instance.PixelData = pixel_data
  if _PYDICOM_MAJOR_VERSION <= 2:
    test_instance.is_implicit_VR = False
    test_instance.is_little_endian = True
  return test_instance


def flatten_data_acquisition(
    data_acquisitions: Iterator[
        abstract_data_accessor.DataAcquisition[np.ndarray]
    ],
) -> Sequence[np.ndarray]:
  results = []
  for data_acquisition in data_acquisitions:
    results.extend(list(data_acquisition.acquision_data_source_iterator))
  return results

