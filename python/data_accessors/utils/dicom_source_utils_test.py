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
from ez_wsi_dicomweb import credential_factory
import pydicom

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import dicom_source_utils
from data_accessors.utils import test_utils
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_MOCK_DICOM_STORE_PATH = 'https://www.mock_dicom_store.com'


class DicomSourceUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='dicom_source_type_generic',
          dcm_file=test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
          expected_source_type=dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      ),
      dict(
          testcase_name='dicom_source_type_slide_microscopy_image',
          dcm_file=test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          ),
          expected_source_type=dicom_source_utils.DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
      ),
  )
  def test_identify_dicom_source_type(self, dcm_file, expected_source_type):
    with pydicom.dcmread(dcm_file) as dcm:
      dcm_path = f'{_MOCK_DICOM_STORE_PATH}/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}/instances/{dcm.SOPInstanceUID}'
      with dicom_store_mock.MockDicomStores(
          _MOCK_DICOM_STORE_PATH
      ) as dicom_store:
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        source_type = (
            dicom_source_utils.get_dicom_source_type_and_instance_metadata(
                credential_factory.NoAuthCredentialsFactory(),
                {
                    _InstanceJsonKeys.DICOM_WEB_URI: dcm_path,
                }
            )
        )
        self.assertEqual(source_type.dicom_source_type, expected_source_type)
        self.assertLen(source_type.dicom_instances_metadata, 1)
        self.assertEqual(
            source_type.dicom_instances_metadata[0].sop_instance_uid,
            dcm.SOPInstanceUID,
        )

  @parameterized.parameters([
      dict(uid='1.2.840.10008.5.1.4.1.1.2', modality='CT'),
      dict(uid='1.2.840.10008.5.1.4.1.1.2.1', modality='CT'),
      dict(uid='1.2.840.10008.5.1.4.1.1.2.2', modality='CT'),
      dict(uid='1.2.840.10008.5.1.4.1.1.4', modality='MR'),
      dict(uid='1.2.840.10008.5.1.4.1.1.4.1', modality='MR'),
      dict(uid='1.2.840.10008.5.1.4.1.1.4.4', modality='MR'),
      dict(uid='1.2.840.10008.5.1.4.1.1.77.1.3', modality='SM'),
      dict(uid='1.2.840.10008.5.1.4.1.1.77.1.2', modality='SM'),
      dict(uid='1.2.840.10008.5.1.4.1.1.77.1.6', modality='SM'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1.1', modality='DX'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1.1.1', modality='DX'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1.2', modality='DX'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1.2.1', modality='DX'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1.3', modality='DX'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1.3.1', modality='DX'),
      dict(uid='1.2.840.10008.5.1.4.1.1.1', modality='CR'),
      dict(uid='1.2.840.9999.5.1.4.1.1.1', modality=''),
  ])
  def test_infer_modality_from_sop_class_uid(self, uid, modality):
    self.assertEqual(
        dicom_source_utils.infer_modality_from_sop_class_uid(uid), modality
    )

  @parameterized.parameters(['CR', 'DX', 'GM', 'SM', 'XC', 'CT', 'MR'])
  def test_validate_modality_supported(self, modality):
    self.assertIsNone(dicom_source_utils.validate_modality_supported(modality))

  def test_validate_modality_unsupported_rasises(self):
    with self.assertRaises(data_accessor_errors.DicomError):
      dicom_source_utils.validate_modality_supported('US')

  def test_identify_source_with_multiple_modalities_in_series_raisese(self):
    dcm_path = f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for dcm_file in (
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          ),
      ):
        with pydicom.dcmread(dcm_file) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
        dicom_source_utils.get_dicom_source_type_and_instance_metadata(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.DICOM_WEB_URI: dcm_path,
            }
        )

  def test_identify_source_with_no_identify_multiple_modalities_raises(self):
    dcm_path = f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for dcm_file in (
          test_utils.testdata_path('cxr', 'encapsulated_cxr.dcm'),
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          ),
      ):
        with pydicom.dcmread(dcm_file) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          del dcm['Modality']
          dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.6.2'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
        dicom_source_utils.get_dicom_source_type_and_instance_metadata(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.DICOM_WEB_URI: dcm_path,
            }
        )

  def test_sm_series_with_wsi_and_non_wsi_instances_raises(self):
    dcm_path = f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dcm.SOPInstanceUID = '1.2.3'
        dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.3'
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
        dicom_source_utils.get_dicom_source_type_and_instance_metadata(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.DICOM_WEB_URI: dcm_path,
            }
        )

  def test_definintion_of_unsupported_instances_raises(self):
    dcm_path = []
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dcm_path.append(
            f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2/instances/{dcm.SOPInstanceUID}'
        )
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        del dcm['Modality']
        dcm.SeriesInstanceUID = '1.2'
        dcm.SOPInstanceUID = '1.2.3'
        dcm.SOPClassUID = '1.2.840.10008.5.1.4.1.1.6.2'
        dcm_path.append(
            f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2/instances/{dcm.SOPInstanceUID}'
        )
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      with self.assertRaises(data_accessor_errors.InvalidRequestFieldError):
        dicom_source_utils.get_dicom_source_type_and_instance_metadata(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.DICOM_SOURCE: dcm_path,
            }
        )

  def test_filter_out_unspecified_sop_instances(self):
    dcm_path = []
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        expected_sop_instance_uid = dcm.SOPInstanceUID
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dcm_path.append(
            f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2/instances/{expected_sop_instance_uid}'
        )
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dcm.SOPInstanceUID = '1.2.3'
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = (
          dicom_source_utils.get_dicom_source_type_and_instance_metadata(
              credential_factory.NoAuthCredentialsFactory(),
              {
                  _InstanceJsonKeys.DICOM_SOURCE: dcm_path,
              }
          )
      )
    self.assertEqual(
        source_type.dicom_source_type,
        dicom_source_utils.DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
    )
    self.assertLen(source_type.dicom_instances_metadata, 1)
    self.assertEqual(
        source_type.dicom_instances_metadata[0].sop_instance_uid,
        expected_sop_instance_uid,
    )

  def test_add_concatenation_instances_to_wsi_specified_sop_instances(self):
    dcm_path = []
    expected_sop_instance_uid = set()
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dcm.ConcatenationUID = '1.2.3.4'
        expected_sop_instance_uid.add(dcm.SOPInstanceUID)
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dcm_path.append(
            f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2/instances/{dcm.SOPInstanceUID}'
        )
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dcm.SOPInstanceUID = '1.2.3'
        dcm.ConcatenationUID = '1.2.3.4'
        expected_sop_instance_uid.add(dcm.SOPInstanceUID)
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = (
          dicom_source_utils.get_dicom_source_type_and_instance_metadata(
              credential_factory.NoAuthCredentialsFactory(),
              {
                  _InstanceJsonKeys.DICOM_SOURCE: dcm_path,
              }
          )
      )
    self.assertEqual(
        source_type.dicom_source_type,
        dicom_source_utils.DicomDataSourceEnum.SLIDE_MICROSCOPY_IMAGE,
    )
    self.assertLen(source_type.dicom_instances_metadata, 2)
    self.assertEqual(
        {m.sop_instance_uid for m in source_type.dicom_instances_metadata},
        expected_sop_instance_uid,
    )

  def test_validate_suported_modality_raises_if_invalid_modality(self):
    dcm_path = []
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      with pydicom.dcmread(
          test_utils.testdata_path(
              'wsi', 'multiframe_camelyon_challenge_image.dcm'
          )
      ) as dcm:
        dcm.StudyInstanceUID = '1.1'
        dcm.SeriesInstanceUID = '1.2'
        dcm.Modality = 'US'
        dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
        dcm_path.append(
            f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2/instances/{dcm.SOPInstanceUID}'
        )
      with self.assertRaises(data_accessor_errors.DicomError):
        dicom_source_utils.get_dicom_source_type_and_instance_metadata(
            credential_factory.NoAuthCredentialsFactory(),
            {
                _InstanceJsonKeys.DICOM_SOURCE: dcm_path,
            }
        )

  def test_discover_ct_series_instances(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 10)

  def test_prefer_ct_instances_with_defined_non_derived_image_type(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          if index < 6:
            dcm.ImageType = ['PRIMARY'] if index < 4 else ['DERIVED']
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 4)

  def test_prefer_derived_ct_instances_over_undefined_image_type(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          if index < 6:
            dcm.ImageType = ['DERIVED']
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 6)

  def test_prefer_ct_acquisition_instances_greatest_number_of_instances(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          if index < 6:
            dcm.ImageType = ['DERIVED']
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 6)

  def test_sort_cs_series_instances_by_instances_defining_slice_position(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dcm.SOPInstanceUID = f'{9-index}'
          if index < 5:
            dcm.ImagePositionPatient = [index, 2, 2]
          else:
            del dcm['ImagePositionPatient']
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertEqual(
          [m.sop_instance_uid for m in source_type.dicom_instances_metadata],
          [str(i) for i in range(9, 4, -1)],
      )

  def test_cs_series_instances_prefer_greatest_acquisition_number(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          if index < 6:
            dcm.AcquisitionNumber = 1
          else:
            dcm.AcquisitionNumber = 2
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 6)

  def test_cs_series_instance_with_lowest_acqusion_date_time_tag(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dcm.InstanceNumber = 1
          dcm.SOPInstanceUID = str(index)
          dcm.AcquisitionDateTime = f'20240130{index:0>2}000000'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          }
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 1)
      self.assertEqual(
          source_type.dicom_instances_metadata[0].sop_instance_uid, '0'
      )

  def test_cs_series_instance_with_lowest_acqusion_date_and_time_tags(self):
    with dicom_store_mock.MockDicomStores(
        _MOCK_DICOM_STORE_PATH
    ) as dicom_store:
      for index in range(10):
        with pydicom.dcmread(
            test_utils.testdata_path('ct', 'test_series', f'image{index}.dcm')
        ) as dcm:
          dcm.StudyInstanceUID = '1.1'
          dcm.SeriesInstanceUID = '1.2'
          dcm.InstanceNumber = 1
          dcm.SOPInstanceUID = str(index)
          dcm.AcquisitionDate = f'202401{index:0>2}'
          dcm.AcquisitionTime = f'{index:0>2}'
          dicom_store[_MOCK_DICOM_STORE_PATH].add_instance(dcm)
      source_type = dicom_source_utils.get_dicom_source_type_and_instance_metadata(
          credential_factory.NoAuthCredentialsFactory(),
          {
              _InstanceJsonKeys.DICOM_SOURCE: (
                  f'{_MOCK_DICOM_STORE_PATH}/studies/1.1/series/1.2'
              ),
          },
      )
      self.assertEqual(
          source_type.dicom_source_type,
          dicom_source_utils.DicomDataSourceEnum.GENERIC_DICOM,
      )
      self.assertLen(source_type.dicom_instances_metadata, 1)
      self.assertEqual(
          source_type.dicom_instances_metadata[0].sop_instance_uid, '0'
      )


if __name__ == '__main__':
  absltest.main()
