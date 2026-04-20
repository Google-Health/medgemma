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

"""Tests for icc profile cache."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import tempfile
import threading
import typing
from typing import Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import ez_wsi_errors
from ez_wsi_dicomweb.ml_toolkit import dicom_path
from google.cloud import storage
import pydicom
import redis

from data_accessors.dicom_wsi import configuration
from data_accessors.dicom_wsi import icc_profile_cache
from data_accessors.dicom_wsi.test_utils import redis_mock
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock

_PYDICOM_MAJOR_VERSION = int((pydicom.__version__).split('.')[0])
_IccProfileConfig = (
    configuration.IccProfileCacheConfiguration
)


def _mock_dicom_instance() -> pydicom.FileDataset:
  """Returns Mock WSI DICOM."""
  sop_class_uid = '1.2.840.10008.5.1.4.1.1.77.1.6'
  sop_instance_uid = '1.2.3.4.5'
  frame_data = b'1234'
  file_meta = pydicom.dataset.FileMetaDataset()
  file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
  file_meta.MediaStorageSOPClassUID = sop_class_uid
  file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
  file_meta.ImplementationClassUID = '1.2.4'
  mk_instance = pydicom.FileDataset(
      '', {}, file_meta=file_meta, preamble=b'\0' * 128
  )
  mk_instance.StudyInstanceUID = '1.2.3'
  mk_instance.SOPClassUID = sop_class_uid
  mk_instance.SeriesInstanceUID = '1.2.3.4'
  mk_instance.SOPInstanceUID = sop_instance_uid
  # Tags required for EZ-WSI
  mk_instance.ImageType = ['ORIGINAL', 'PRIMARY', 'VOLUME']
  mk_instance.InstanceNumber = 1
  mk_instance.TotalPixelMatrixColumns = len(frame_data)
  mk_instance.TotalPixelMatrixRows = 1
  mk_instance.Columns = len(frame_data)
  mk_instance.Rows = 1
  mk_instance.SamplesPerPixel = 1
  mk_instance.BitsAllocated = 8
  mk_instance.NumberOfFrames = 1
  mk_instance.ImagedVolumeWidth = 1
  mk_instance.ImagedVolumeHeight = 1.0 / len(frame_data)
  mk_instance.HighBit = 7
  mk_instance.PixelData = [frame_data]
  if _PYDICOM_MAJOR_VERSION <= 2:
    mk_instance.is_implicit_VR = False
    mk_instance.is_little_endian = True
  return mk_instance


def _run_mock_cache_update_thread(th: threading.Thread) -> None:
  th.start()
  th.join()


class _MocGcpEnv(contextlib.ExitStack):

  def __init__(
      self,
      icc_profile_bytes: bytes = b'',
      gcs_bucket: str = 'gcs_bucket',
      redis_ip: str = 'mock_redishost',
      redis_port: int = 123,
  ):
    super().__init__()
    self._mock_redis_ip = redis_ip
    self._mock_redis_port = redis_port
    self.mock_redis_impl = redis_mock.MockRedisClient(
        self._mock_redis_ip, self._mock_redis_port
    )
    self.gcs_cache_bucket = gcs_bucket
    self._test_store = dicom_path.FromString(
        'projects/project_name/locations/us-west1/datasets/dataset_name/dicomStores/dicom_store_name/dicomWeb'
    )
    self.dicom_store_path = self._test_store
    # create test DICOM instance
    self.mk_dicom_instance = _mock_dicom_instance()
    if icc_profile_bytes:
      self.mk_dicom_instance.ICCProfile = icc_profile_bytes
    self._mk_dicom_store = None
    self._dicom_slide = None

  def __enter__(self) -> _MocGcpEnv:
    super().__enter__()
    self.enter_context(
        mock.patch.object(
            redis, 'Redis', autospec=True, return_value=self.mock_redis_impl
        )
    )
    self.enter_context(
        mock.patch.object(
            icc_profile_cache,
            '_run_cache_update_thread',
            side_effect=_run_mock_cache_update_thread,
        )
    )
    self.mock_bucket = self.enter_context(tempfile.TemporaryDirectory())
    self.enter_context(
        gcs_mock.GcsMock({self.gcs_cache_bucket: str(self.mock_bucket)})
    )
    self._mk_dicom_store = self.enter_context(
        dicom_store_mock.MockDicomStores(str(self.dicom_store_path))
    )
    self._mk_dicom_store[str(self.dicom_store_path)].add_instance(
        self.mk_dicom_instance
    )
    return self

  @property
  def dicom_slide(self) -> dicom_slide.DicomSlide:
    if self._dicom_slide is not None:
      return self._dicom_slide
    self._dicom_slide = dicom_slide.DicomSlide(
        dicom_web_interface.DicomWebInterface(
            credential_factory.CredentialFactory()
        ),
        dicom_path.FromPath(
            self._test_store,
            study_uid=self.mk_dicom_instance.StudyInstanceUID,
            series_uid=self.mk_dicom_instance.SeriesInstanceUID,
        ),
    )
    return self._dicom_slide

  @property
  def mk_dicom_store(self) -> dicom_store_mock.MockDicomStoreClient:
    if self._mk_dicom_store is None:
      raise ValueError('mk_dicom_store not initialized')
    return self._mk_dicom_store[str(self.dicom_store_path)]

  def get_dicom_icc_profile(
      self,
      gcs_bucket: Optional[str] = None,
      redis_ip: Optional[str] = None,
      redis_port: Optional[int] = None,
      testing: bool = True,
  ) -> bytes:
    ds = self.dicom_slide
    gcs_bucket = self.gcs_cache_bucket if gcs_bucket is None else gcs_bucket
    redis_ip = self._mock_redis_ip if redis_ip is None else redis_ip
    redis_port = self._mock_redis_port if redis_port is None else redis_port
    return icc_profile_cache.get_dicom_icc_profile(
        _IccProfileConfig(
            gcs_bucket=gcs_bucket,
            redis_ip=redis_ip,
            redis_port=redis_port,
            testing=testing,
        ),
        ds,
        ds.native_level,
    )


class PeteIccProfileCacheTest(parameterized.TestCase):

  def test_missing_dicom_instance(self, *unused_mocks):
    with _MocGcpEnv() as mock_gcp_env:
      _ = mock_gcp_env.dicom_slide
      mock_gcp_env.mk_dicom_store.remove_instance(
          mock_gcp_env.mk_dicom_instance
      )
      with self.assertRaises(ez_wsi_errors.HttpNotFoundError):
        mock_gcp_env.get_dicom_icc_profile()

  @parameterized.named_parameters(
      dict(
          testcase_name='no_icc_profile',
          icc_profile_bytes=b'',
      ),
      dict(
          testcase_name='icc_profile',
          icc_profile_bytes=b'icc_profile',
      ),
  )
  def test_return_uncached_icc_profile(
      self, *unused_mocks, icc_profile_bytes: bytes
  ):
    with _MocGcpEnv(icc_profile_bytes) as mock_gcp_env:
      # test get_dicom_icc_profile no cached data.
      self.assertEqual(mock_gcp_env.get_dicom_icc_profile(), icc_profile_bytes)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_icc_profile',
          icc_profile_bytes=b'',
      ),
      dict(
          testcase_name='icc_profile',
          icc_profile_bytes=b'icc_profile',
      ),
  )
  def test_two_icc_profile_requests_return_cached_profile(
      self, icc_profile_bytes
  ):
    with _MocGcpEnv(icc_profile_bytes) as mock_gcp_env:
      mock_gcp_env.get_dicom_icc_profile()
      # Test ICC Profile is returned from cache and not re-feteched from the
      # store.
      with mock.patch.object(
          dicom_slide.DicomSlide, 'get_icc_profile_bytes', autospec=True
      ) as mock_get_icc_profile_bytes:
        # test get_dicom_icc_profile
        self.assertEqual(
            mock_gcp_env.get_dicom_icc_profile(), icc_profile_bytes
        )
        mock_get_icc_profile_bytes.assert_not_called()

  def test_two_icc_profile_requests_missing_gcs_cached_bytes(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mock_gcp_env:
      mock_gcp_env.get_dicom_icc_profile()
      # delete blobs from GCS cache.
      for blob in storage.Client().list_blobs(mock_gcp_env.gcs_cache_bucket):
        blob.delete()
      # test get_dicom_icc_profile
      self.assertEqual(mock_gcp_env.get_dicom_icc_profile(), icc_profile_bytes)

  def test_two_icc_profile_requests_missing_redis_cached_instance(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mock_gcp_env:
      mock_gcp_env.get_dicom_icc_profile()
      mock_gcp_env.mock_redis_impl.clear()
      # test get_dicom_icc_profile
      with mock.patch.object(
          icc_profile_cache, '_upload_file_to_blob', autospec=True
      ) as mock_blob_upload:
        self.assertEqual(
            mock_gcp_env.get_dicom_icc_profile(), icc_profile_bytes
        )
        mock_blob_upload.assert_not_called()

  def test_two_icc_profile_requests_returns_in_memory_cached_instance(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mock_gcp_env:
      mock_gcp_env.get_dicom_icc_profile(gcs_bucket='gcs_bucket', redis_ip='')
      with mock.patch.object(
          icc_profile_cache, '_download_blob_as_bytes', autospec=True
      ) as mock_blob_download:
        self.assertEqual(
            mock_gcp_env.get_dicom_icc_profile(testing=False), icc_profile_bytes
        )
        mock_blob_download.assert_not_called()

  def test_cache_key_gcs_path(self):
    self.assertEqual(
        icc_profile_cache._cache_key_gcs_path('bucket', 'hash'),
        'gs://bucket/slide_keys/hash',
    )

  def test_init_fork_module_state(self):
    icc_profile_cache._local_icc_profile_bytes_cache = None
    icc_profile_cache._local_icc_profile_bytes_index_cache = None
    icc_profile_cache._local_icc_profile_bytes_cache_lock = None
    icc_profile_cache._init_fork_module_state()
    self.assertIsNotNone(icc_profile_cache._local_icc_profile_bytes_cache)
    self.assertIsNotNone(icc_profile_cache._local_icc_profile_bytes_index_cache)
    self.assertIsNotNone(icc_profile_cache._local_icc_profile_bytes_cache_lock)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_prefix',
          bucket_name='gs://bucket',
          expected='bucket',
      ),
      dict(
          testcase_name='prefix_with_slash',
          bucket_name='gs://bucket/',
          expected='bucket',
      ),
      dict(
          testcase_name='slash',
          bucket_name='bucket/',
          expected='bucket',
      ),
  )
  def test_normalize_bucket_name(self, bucket_name, expected):
    self.assertEqual(
        icc_profile_cache._normalize_bucket_name(bucket_name), expected
    )

  def test_in_memory_cache_set_icc_profile_bytes(self):
    icc_profile_bytes = b'icc_profile'
    redis_cache_key = 'cache_key'
    icc_profile_hash = hashlib.sha3_512(icc_profile_bytes).hexdigest()
    icc_profile_hash = icc_profile_hash.encode('utf-8')
    with _MocGcpEnv(icc_profile_bytes) as _:
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(gcs_bucket='gcs_bucket', testing=True),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      self.assertEqual(
          icc_profile_cache._local_icc_profile_bytes_index_cache[
              redis_cache_key
          ],
          icc_profile_hash,
      )
      self.assertEqual(
          icc_profile_cache._local_icc_profile_bytes_cache[icc_profile_hash],
          icc_profile_bytes,
      )

  def test_in_memory_cache_set_no_icc_profile_bytes(self):
    icc_profile_bytes = b''
    redis_cache_key = 'cache_key'
    with _MocGcpEnv(icc_profile_bytes) as _:
      length_before = len(icc_profile_cache._local_icc_profile_bytes_cache)
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(gcs_bucket='gcs_bucket', testing=True),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      self.assertEqual(
          icc_profile_cache._local_icc_profile_bytes_index_cache[
              redis_cache_key
          ],
          icc_profile_cache._NO_ICC_PROFILE_BYTES,
      )
      self.assertLen(
          icc_profile_cache._local_icc_profile_bytes_cache, length_before
      )

  def test_set_no_icc_profile_bytes_redis(self):
    icc_profile_bytes = b''
    redis_cache_key = 'cache_key'
    with _MocGcpEnv(icc_profile_bytes) as mk:
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(
              gcs_bucket='gcs_bucket', redis_ip='localhost', testing=True
          ),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      self.assertEqual(
          mk.mock_redis_impl.get(redis_cache_key),
          icc_profile_cache._NO_ICC_PROFILE_BYTES,
      )
      self.assertLen(mk.mock_redis_impl, 1)
      # nothing written to the bucket
      self.assertEmpty(os.listdir(mk.mock_bucket))

  def test_set_no_icc_profile_bytes_gcs(self):
    icc_profile_bytes = b''
    redis_cache_key = 'cache_key'
    with _MocGcpEnv(icc_profile_bytes) as mk:
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(gcs_bucket='gcs_bucket', testing=True),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      # nothing written to the redis
      self.assertEmpty(mk.mock_redis_impl)
      # one file written to gcs
      self.assertLen(os.listdir(mk.mock_bucket), 1)
      # validate expected key file points to nothing.
      with open(
          os.path.join(mk.mock_bucket, 'slide_keys', redis_cache_key), 'rb'
      ) as infile:
        file_bytes = infile.read()
        self.assertEqual(file_bytes, icc_profile_cache._NO_ICC_PROFILE_BYTES)

  def test_set_icc_profile_key_redis_store_icc_profile_in_redis(self):
    icc_profile_bytes = b'icc_profile'
    redis_cache_key = 'cache_key'
    icc_profile_hash = hashlib.sha3_512(icc_profile_bytes).hexdigest()
    icc_profile_hash = icc_profile_hash.encode('utf-8')
    with _MocGcpEnv(icc_profile_bytes) as mk:
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(
              gcs_bucket='gcs_bucket',
              redis_ip='localhost',
              store_icc_profile_bytes_in_redis=True,
              testing=True,
          ),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      self.assertEqual(
          mk.mock_redis_impl.get(redis_cache_key),
          icc_profile_hash,
      )
      self.assertEqual(
          mk.mock_redis_impl.get(
              icc_profile_cache._get_gcs_cache_path(
                  'gcs_bucket', icc_profile_hash.decode('utf-8')
              )
          ),
          icc_profile_bytes,
      )
      self.assertLen(mk.mock_redis_impl, 2)
      # nothing written to the bucket
      self.assertEmpty(os.listdir(mk.mock_bucket))

  def test_set_icc_profile_key_redis_store_icc_profile_in_gcs(self):
    icc_profile_bytes = b'icc_profile'
    redis_cache_key = 'cache_key'
    icc_profile_hash = hashlib.sha3_512(icc_profile_bytes).hexdigest()
    icc_profile_hash = icc_profile_hash.encode('utf-8')
    with _MocGcpEnv(icc_profile_bytes) as mk:
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(
              gcs_bucket='gcs_bucket',
              redis_ip='localhost',
              store_icc_profile_bytes_in_redis=False,
              testing=True,
          ),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      self.assertEqual(
          mk.mock_redis_impl.get(redis_cache_key),
          icc_profile_hash,
      )
      self.assertLen(mk.mock_redis_impl, 1)
      # one file written to the bucket
      self.assertLen(os.listdir(mk.mock_bucket), 1)
      with open(
          os.path.join(
              mk.mock_bucket, f'{icc_profile_hash.decode("utf-8")}.icc_profile'
          ),
          'rb',
      ) as infile:
        file_bytes = infile.read()
        self.assertEqual(file_bytes, icc_profile_bytes)

  def test_set_icc_profile_key_gcs_store_icc_profile_in_gcs(self):
    icc_profile_bytes = b'icc_profile'
    redis_cache_key = 'cache_key'
    icc_profile_hash = hashlib.sha3_512(icc_profile_bytes).hexdigest()
    icc_profile_hash = icc_profile_hash.encode('utf-8')
    with _MocGcpEnv(icc_profile_bytes) as mk:
      icc_profile_cache._update_profile_cache(
          _IccProfileConfig(gcs_bucket='gcs_bucket', testing=True),
          redis_cache_key,
          icc_profile_bytes,
          {},
      )
      # nothing written to redis
      self.assertEmpty(mk.mock_redis_impl)
      # one two files written to the bucket
      self.assertLen(os.listdir(mk.mock_bucket), 2)
      with open(
          os.path.join(mk.mock_bucket, 'slide_keys', redis_cache_key), 'rb'
      ) as infile:
        file_bytes = infile.read()
        self.assertEqual(file_bytes, icc_profile_hash)

      self.assertLen(os.listdir(mk.mock_bucket), 2)
      with open(
          os.path.join(
              mk.mock_bucket, f'{icc_profile_hash.decode("utf-8")}.icc_profile'
          ),
          'rb',
      ) as infile:
        file_bytes = infile.read()
        self.assertEqual(file_bytes, icc_profile_bytes)

  def test_download_icc_profile_redis_found(self):
    icc_profile_bytes = b'icc_profile'
    gcs_cache_path = icc_profile_cache._get_gcs_cache_path(
        'gcs_bucket', 'mock_hash'
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      mk.mock_redis_impl.set(
          gcs_cache_path, b'icc_profile'
      )  # fill redis cache.
      icc_profile = icc_profile_cache._download_icc_profile(
          _IccProfileConfig(
              store_icc_profile_bytes_in_redis=True, testing=True
          ),
          gcs_cache_path,
          'localhost',
          1,
          {},
      )
      self.assertEqual(icc_profile, b'icc_profile')

  def test_download_icc_profile_no_redis_or_gcs(self):
    self.assertIsNone(
        icc_profile_cache._download_icc_profile(
            _IccProfileConfig(testing=True), '', '', 1, {}
        )
    )

  def test_download_icc_profile_redis_not_found(self):
    icc_profile_bytes = b'icc_profile'
    gcs_cache_path = icc_profile_cache._get_gcs_cache_path(
        'gcs_bucket', 'mock_hash'
    )
    with _MocGcpEnv(icc_profile_bytes):
      icc_profile = icc_profile_cache._download_icc_profile(
          _IccProfileConfig(
              store_icc_profile_bytes_in_redis=True, testing=True
          ),
          gcs_cache_path,
          'localhost',
          1,
          {},
      )
      self.assertIsNone(icc_profile)

  def test_download_icc_profile_gcs_not_found(self):
    icc_profile_bytes = b'icc_profile'
    gcs_cache_path = icc_profile_cache._get_gcs_cache_path(
        'gcs_bucket', 'mock_hash'
    )
    with _MocGcpEnv(icc_profile_bytes):
      icc_profile = icc_profile_cache._download_icc_profile(
          _IccProfileConfig(testing=True), gcs_cache_path, 'localhost', 1, {}
      )
      self.assertIsNone(icc_profile)

  def test_download_icc_profile_gcs_found(self):
    icc_profile_bytes = b'icc_profile'
    gcs_cache_path = icc_profile_cache._get_gcs_cache_path(
        'gcs_bucket', 'mock_hash'
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      with open(
          os.path.join(mk.mock_bucket, 'mock_hash.icc_profile'), 'wb'
      ) as outfile:
        outfile.write(icc_profile_bytes)
      icc_profile = icc_profile_cache._download_icc_profile(
          _IccProfileConfig(testing=True), gcs_cache_path, 'localhost', 1, {}
      )
      self.assertEqual(icc_profile, icc_profile_bytes)

  def test_get_icc_profile_wsi_slide(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mk:
      self.assertEqual(
          icc_profile_cache._get_dicom_slide_icc_profile(mk.dicom_slide),
          icc_profile_bytes,
      )

  def test_get_icc_profile_microscope_image(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mk:
      dcm_image = _mock_dicom_instance()
      dcm_image.file_meta.MediaStorageSOPClassUID = (
          '1.2.840.10008.5.1.4.1.1.77.1.3'
      )
      dcm_image.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.3'
      dcm_image.SeriesInstanceUID = '4.5.6'
      dcm_image.ICCProfile = icc_profile_bytes
      mk.mk_dicom_store.add_instance(dcm_image)
      image = dicom_slide.DicomMicroscopeImage(
          dicom_web_interface.DicomWebInterface(
              credential_factory.CredentialFactory()
          ),
          dicom_path.FromPath(
              mk.dicom_store_path,
              study_uid=dcm_image.StudyInstanceUID,
              series_uid=dcm_image.SeriesInstanceUID,
          ),
      )
      level = next(image.levels)
      self.assertEqual(
          icc_profile_cache._get_dicom_slide_icc_profile(image, level),
          icc_profile_bytes + b'\x00',
      )

  def test_get_icc_profile_invalid_class_raises(self):
    with self.assertRaises(ValueError):
      icc_profile_cache._get_dicom_slide_icc_profile(
          typing.cast(dicom_slide.DicomSlide, 'abc')
      )

  def test_get_icc_profile_from_dicom_if_no_redis_or_gcs_defined(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mk:
      icc_profile = icc_profile_cache.get_dicom_icc_profile(
          _IccProfileConfig(testing=True), mk.dicom_slide
      )
      self.assertEqual(icc_profile, icc_profile_bytes)

  def test_get_icc_profile_no_metadata_in_redis_triggers_download(self):
    icc_profile_bytes = b'icc_profile'
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      with mock.patch.object(
          icc_profile_cache, '_update_profile_cache', autospec=True
      ) as update_profile_cache:
        with mock.patch.object(
            icc_profile_cache,
            '_get_dicom_slide_icc_profile',
            autospec=True,
            return_value=icc_profile_bytes,
        ) as mock_download_icc_profile:
          icc_profile_cache_config = _IccProfileConfig(
              gcs_bucket='gcs_bucket', testing=True
          )
          result = icc_profile_cache.get_dicom_icc_profile(
              icc_profile_cache_config,
              mk.dicom_slide,
          )
          mock_download_icc_profile.assert_called()
          update_profile_cache.assert_called_with(
              icc_profile_cache_config,
              dicom_slide_path_cache_key,
              icc_profile_bytes,
              mock.ANY,
          )
          self.assertEqual(result, icc_profile_bytes)

  def test_get_icc_profile_key_local_cache_retrieve_local_cache(self):
    icc_profile_bytes = b'icc_profile'
    icc_profile_cache_config = _IccProfileConfig(
        gcs_bucket='gcs_bucket', testing=True
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      icc_profile_cache._update_profile_cache(
          icc_profile_cache_config,
          dicom_slide_path_cache_key,
          icc_profile_bytes,
          {},
      )
      # remove files cached in mock gcs bucket.
      for root, _, files in os.walk(mk.mock_bucket):
        for fname in files:
          os.remove(os.path.join(root, fname))
      with mock.patch.object(
          icc_profile_cache,
          '_get_dicom_slide_icc_profile',
          autospec=True,
          return_value=icc_profile_bytes,
      ) as mock_download_icc_profile:
        result = icc_profile_cache.get_dicom_icc_profile(
            dataclasses.replace(icc_profile_cache_config, testing=False),
            mk.dicom_slide,
        )
        mock_download_icc_profile.assert_not_called()  # not downloaded
      self.assertEmpty(mk.mock_redis_impl)  # no data in redis
      self.assertEqual(result, icc_profile_bytes)  # returned icc profile bytes

  def test_get_icc_profile_key_redis_retrieve_redis(self):
    icc_profile_bytes = b'icc_profile'
    icc_profile_cache_config = _IccProfileConfig(
        gcs_bucket='gcs_bucket',
        redis_ip='localhost',
        store_icc_profile_bytes_in_redis=True,
        testing=True,
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      icc_profile_cache._update_profile_cache(
          icc_profile_cache_config,
          dicom_slide_path_cache_key,
          icc_profile_bytes,
          {},
      )
      # remove files cached in mock gcs bucket.
      for root, _, files in os.walk(mk.mock_bucket):
        for fname in files:
          os.remove(os.path.join(root, fname))
      with mock.patch.object(
          icc_profile_cache,
          '_get_dicom_slide_icc_profile',
          autospec=True,
          return_value=icc_profile_bytes,
      ) as mock_download_icc_profile:
        result = icc_profile_cache.get_dicom_icc_profile(
            icc_profile_cache_config,
            mk.dicom_slide,
        )
        mock_download_icc_profile.assert_not_called()  # not downloaded
      self.assertLen(mk.mock_redis_impl, 2)  # data stored in redis
      self.assertEqual(result, icc_profile_bytes)  # returned icc profile bytes

  def test_get_icc_profile_key_redis_retrieve_gcs(self):
    icc_profile_bytes = b'icc_profile'
    icc_profile_cache_config = _IccProfileConfig(
        gcs_bucket='gcs_bucket',
        redis_ip='localhost',
        store_icc_profile_bytes_in_redis=False,
        testing=True,
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      icc_profile_cache._update_profile_cache(
          icc_profile_cache_config,
          dicom_slide_path_cache_key,
          icc_profile_bytes,
          {},
      )
      # remove files cached in mock gcs bucket.
      for root, _, files in os.walk(mk.mock_bucket):
        if root == mk.mock_bucket:  # skip cached icc profile bytes
          continue
        for fname in files:
          os.remove(os.path.join(root, fname))
      with mock.patch.object(
          icc_profile_cache,
          '_get_dicom_slide_icc_profile',
          autospec=True,
          return_value=icc_profile_bytes,
      ) as mock_download_icc_profile:
        result = icc_profile_cache.get_dicom_icc_profile(
            icc_profile_cache_config, mk.dicom_slide
        )
        mock_download_icc_profile.assert_not_called()  # not downloaded
      self.assertLen(mk.mock_redis_impl, 1)  # only key stored in redis
      self.assertEqual(result, icc_profile_bytes)  # returned icc profile bytes

  def test_get_icc_profile_key_gcs_retrieve_gcs(self):
    icc_profile_bytes = b'icc_profile'
    icc_profile_cache_config = _IccProfileConfig(
        gcs_bucket='gcs_bucket', testing=True
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      icc_profile_cache._update_profile_cache(
          icc_profile_cache_config,
          dicom_slide_path_cache_key,
          icc_profile_bytes,
          {},
      )
      with mock.patch.object(
          icc_profile_cache,
          '_get_dicom_slide_icc_profile',
          autospec=True,
          return_value=icc_profile_bytes,
      ) as mock_download_icc_profile:
        result = icc_profile_cache.get_dicom_icc_profile(
            icc_profile_cache_config, mk.dicom_slide
        )
        mock_download_icc_profile.assert_not_called()  # not downloaded
      self.assertEmpty(mk.mock_redis_impl)  # no data stored in redis
      self.assertEqual(result, icc_profile_bytes)  # returned icc profile bytes

  def test_get_icc_profile_key_local_cache_retrieve_gcs(self):
    icc_profile_bytes = b'icc_profile'
    icc_profile_cache_config = _IccProfileConfig(
        gcs_bucket='gcs_bucket', testing=True
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      icc_profile_cache._update_profile_cache(
          icc_profile_cache_config,
          dicom_slide_path_cache_key,
          icc_profile_bytes,
          {},
      )
      for root, _, files in os.walk(mk.mock_bucket):
        if root == mk.mock_bucket:  # skip cached icc profile bytes
          continue
        for fname in files:
          os.remove(os.path.join(root, fname))
      with mock.patch.object(
          icc_profile_cache,
          '_get_dicom_slide_icc_profile',
          autospec=True,
          return_value=icc_profile_bytes,
      ) as mock_download_icc_profile:
        icc_profile_cache._local_icc_profile_bytes_cache.clear()
        result = icc_profile_cache.get_dicom_icc_profile(
            dataclasses.replace(icc_profile_cache_config, testing=False),
            mk.dicom_slide,
        )
        mock_download_icc_profile.assert_not_called()  # not downloaded
      self.assertEmpty(mk.mock_redis_impl)  # no data stored in redis
      self.assertEqual(result, icc_profile_bytes)  # returned icc profile bytes

  def test_get_icc_profile_key_local_cache_retrieve_redis(self):
    icc_profile_bytes = b'icc_profile'
    icc_profile_cache_config = _IccProfileConfig(
        gcs_bucket='gcs_bucket',
        redis_ip='localhost',
        store_icc_profile_bytes_in_redis=True,
        testing=True,
    )
    with _MocGcpEnv(icc_profile_bytes) as mk:
      path = str(mk.dicom_slide.path).encode('utf-8')
      dicom_slide_path_cache_key = hashlib.sha3_512(path).hexdigest()
      icc_profile_cache._update_profile_cache(
          icc_profile_cache_config,
          dicom_slide_path_cache_key,
          icc_profile_bytes,
          {},
      )
      # remove all files from gs mock
      for root, _, files in os.walk(mk.mock_bucket):
        for fname in files:
          os.remove(os.path.join(root, fname))
      with mock.patch.object(
          icc_profile_cache,
          '_get_dicom_slide_icc_profile',
          autospec=True,
          return_value=icc_profile_bytes,
      ) as mock_download_icc_profile:
        icc_profile_cache._local_icc_profile_bytes_cache.clear()
        result = icc_profile_cache.get_dicom_icc_profile(
            dataclasses.replace(icc_profile_cache_config, testing=False),
            mk.dicom_slide,
        )
        mock_download_icc_profile.assert_not_called()  # not downloaded
    self.assertLen(mk.mock_redis_impl, 2)  # cached key and value
    self.assertEqual(result, icc_profile_bytes)  # returned icc profile bytes


if __name__ == '__main__':
  absltest.main()
