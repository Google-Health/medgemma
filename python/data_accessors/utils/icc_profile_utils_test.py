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

"""Tests for icc profile utils."""

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import dicom_slide

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import icc_profile_utils


_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys


class ICCProfileUtilsTest(parameterized.TestCase):

  @parameterized.parameters([1, ({},), ([],), 1.2])
  def test_get_icc_profile_invalid_json_raises_not_a_string(self, val):
    with self.assertRaisesRegex(
        data_accessor_errors.InvalidRequestFieldError,
        f'{_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value is not a'
        ' string',
    ):
      icc_profile_utils.get_target_icc_profile(
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: val}
      )

  def test_get_icc_profile_set_to_invalid_value_raises_not_a_string(self):
    with self.assertRaisesRegex(
        data_accessor_errors.InvalidIccProfileTransformError,
        f'{_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value is not'
        ' valid; expecting: ADOBERGB, ROMMRGB, SRGB, or NONE.',
    ):
      icc_profile_utils.get_target_icc_profile(
          {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'bad_value'}
      )

  def test_get_icc_profile_missing_key(self):
    self.assertIsNone(icc_profile_utils.get_target_icc_profile({}))

  def test_get_icc_profile_none(self):
    self.assertIsNone(
        icc_profile_utils.get_target_icc_profile(
            {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'noNe'}
        )
    )

  @parameterized.parameters(['adobeRGB', 'ADOBERGB'])
  def test_get_adobe_iccprofile(self, profile_name):
    profile = icc_profile_utils.get_target_icc_profile(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(),
        dicom_slide.get_adobergb_icc_profile_bytes(),
    )

  @parameterized.parameters(['sRGB', 'SRGB'])
  def test_get_srgb_iccprofile(self, profile_name):
    profile = icc_profile_utils.get_target_icc_profile(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(),
        dicom_slide.get_srgb_icc_profile_bytes(),
    )

  @parameterized.parameters(['rommRGB', 'ROMMRGB'])
  def test_get_rommrgb_iccprofile(self, profile_name):
    profile = icc_profile_utils.get_target_icc_profile(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(), dicom_slide.get_rommrgb_icc_profile_bytes()
    )

  @parameterized.parameters(['displayP3', 'DISPLAYP3'])
  def test_get_displayp3_iccprofile(self, profile_name):
    profile = icc_profile_utils.get_target_icc_profile(
        {_InstanceJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(), dicom_slide.get_displayp3_icc_profile_bytes()
    )


if __name__ == '__main__':
  absltest.main()
