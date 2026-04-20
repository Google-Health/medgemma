# Copyright 2023 Google LLC
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
"""Tests cloud logging client."""
import collections
import concurrent.futures
import inspect
import logging
import os
import re
import threading
import time
import traceback
from unittest import mock

from absl import flags
from absl import logging as absl_logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import google.auth
from google.cloud import logging as cloud_logging

from serving.logging_lib import cloud_logging_client
from serving.logging_lib import cloud_logging_client_instance

_MOCK_BUILD_VERSION = {'BUILD_VERSION': 'MOCK_UNIT_TEST'}


def _logger_test_thread(val: int) -> None:
  cl = cloud_logging_client.logger()
  cl.log_signature = {'Thread': str(val)}
  with mock.patch.dict(cl._thread_local_storage.signature, _MOCK_BUILD_VERSION):
    cl.info(f'Thread_test_{val}')


class CloudLoggingTest(parameterized.TestCase):
  """Test clould logging Client."""

  def setUp(self):
    super().setUp()
    cloud_logging_client.CloudLoggingClient._singleton_lock = threading.RLock()
    cloud_logging_client.CloudLoggingClient._startup_message_logged = False
    cloud_logging_client.CloudLoggingClient._singleton_instance = None
    cloud_logging_client_instance.CloudLoggingClientInstance._global_lock = (
        threading.Lock()
    )
    cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params = (
        ''
    )
    cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler = (
        None
    )
    cloud_logging_client_instance.CloudLoggingClientInstance._per_thread_log_signatures = (
        True
    )
    self.enter_context(flagsaver.flagsaver(ops_log_project='TEST_PROJECT'))
    self._mock_client_instance = mock.create_autospec(
        cloud_logging.Client, instance=True
    )
    self._mock_client_instance._handlers = set()
    self._mock_client_instance.project = ''

  def tearDown(self):
    # force cloud logger to re-initalize for each unit test.
    cloud_logging_client_instance.CloudLoggingClientInstance.fork_shutdown()
    super().tearDown()

  def test_get_source_location_to_log(self):
    location = cloud_logging_client_instance._get_source_location_to_log(0)

    self.assertIn('source_location', location)
    self.assertIn('file', location['source_location'])
    self.assertEndsWith(
        location['source_location']['file'], 'cloud_logging_client_test.py'
    )
    self.assertEqual(
        location['source_location']['function'],
        'test_get_source_location_to_log',
    )
    # Test line number of function call was within a range.
    # test will fail if line number of function call changes dramatically.
    line_number = location['source_location']['line']
    self.assertGreater(line_number, 10)
    self.assertLess(line_number, 250)

  @parameterized.parameters([
      ('error', cloud_logging_client_instance._LogSeverity.ERROR),
      ('critical', cloud_logging_client_instance._LogSeverity.CRITICAL),
      ('debug', cloud_logging_client_instance._LogSeverity.DEBUG),
      ('info', cloud_logging_client_instance._LogSeverity.INFO),
      ('warning', cloud_logging_client_instance._LogSeverity.WARNING),
  ])
  def test_python_logger(
      self,
      python_logger: str,
      severity: cloud_logging_client_instance._LogSeverity,
  ):
    py_logger = logging.getLogger()
    with mock.patch.object(
        py_logger, python_logger, autospec=True
    ) as mock_logger:
      cloud_logging_client_instance._py_log(
          py_logger, 'test', {'foo': 'bar'}, severity
      )
      mock_logger.assert_called_once_with('test', extra={'foo': 'bar'})

  def test_python_logger_raises_with_unrecognized_severity_level(self):
    with self.assertRaisesRegex(
        cloud_logging_client_instance.CloudLoggerInstanceExceptionError,
        'Unsupported logging severity level; Severity=',
    ):
      cloud_logging_client_instance._py_log(
          logging.getLogger(), 'test', {'foo': 'bar'}, 99999
      )  # pytype: disable=wrong-arg-types

  def test_get_source_location_to_log_with_stack_frames_back_beyond_stack(self):
    with self.assertRaises(ValueError):
      cloud_logging_client_instance._get_source_location_to_log(999)

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  @mock.patch.object(inspect.Traceback, 'lineno', 64)
  @mock.patch.object(cloud_logging_client_instance, '_py_log', autospec=True)
  def test_log_structure(self, mock_pylogger):
    """Tests input to cloud logging log_struct.

    The cloud logging functions, cl._google_logger.log_struct, tested here are
    execulded from most of the unit testing due to the inability to communicate
    with cloud logging from the unit test. In this and similar unit tests.
    the logging client is initialized as in other unit tests with the logging
    to cloud disabled (debug_logging_use_absl_logging=True). The tested code
    path is then enabled by mocking the cloud_logging_client_instance property
    use_absl_logging=False. The test here validates that the logging interface
    is being called with the expected parameters. The test invalidates the
    logging interface singleton to force the singleton to be re-created in
    subsequent unit tests.

    Args:
      mock_pylogger: Mock wrapper for python logging.
    """
    with flagsaver.flagsaver(debug_logging_use_absl_logging=True):
      cl = cloud_logging_client.logger()
    with mock.patch.object(
        cloud_logging_client_instance.CloudLoggingClientInstance,
        'use_absl_logging',
        autospec=True,
        return_value=False,
    ):
      cl.log_signature = {}
      with mock.patch.dict(
          cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
      ):
        cl._log(
            'test',
            cloud_logging_client_instance._LogSeverity.INFO,
            ({'foo': 'bar'},),
        )
    mock_pylogger.assert_called_once_with(
        logging.getLogger(),
        'test',
        extra={
            'json_fields': {
                'foo': 'bar',
                'HOSTNAME': 'test_pod',
                'POD_UID': '123',
                'THREAD_ID': mock.ANY,
                'BUILD_VERSION': 'MOCK_UNIT_TEST',
            },
            'source_location': {
                'file': mock.ANY,
                'function': 'test_log_structure',
                'line': 64,
            },
        },
        severity=cloud_logging_client_instance._LogSeverity.INFO,
    )

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  @mock.patch.object(inspect.Traceback, 'lineno', 64)
  @mock.patch.object(cloud_logging_client_instance, '_py_log', autospec=True)
  def test_log_no_structure(self, mock_pylogger):
    """Tests input to cloud logging log_struct.

    The cloud logging functions, cl._google_logger.log_struct, tested here are
    execulded from most of the unit testing due to the inability to communicate
    with cloud logging from the unit test. In this and similar unit tests.
    the logging client is initialized as in other unit tests with the logging
    to cloud disabled (debug_logging_use_absl_logging=True). The tested code
    path is then enabled by mocking the cloud_logging_client_instance property
    use_absl_logging=False. The test here validates that the logging
    interface is being called with the expected parameters. The test invalidates
    the logging interface singleton to force the singleton to be re-created in
    subsequent unit tests.

    Args:
      mock_pylogger: Mock wrapper for python logging.
    """
    with flagsaver.flagsaver(debug_logging_use_absl_logging=True):
      cl = cloud_logging_client.logger()
    with mock.patch.object(
        cloud_logging_client_instance.CloudLoggingClientInstance,
        'use_absl_logging',
        autospec=True,
        return_value=False,
    ):
      cl.log_signature = {}
      with mock.patch.dict(
          cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
      ):
        cl._log(
            'test', cloud_logging_client_instance._LogSeverity.INFO, (None,)
        )

    # test mocked call excluding source location
    mock_pylogger.assert_called_once_with(
        logging.getLogger(),
        'test',
        extra={
            'json_fields': collections.OrderedDict([
                ('HOSTNAME', 'test_pod'),
                ('POD_UID', '123'),
                ('THREAD_ID', mock.ANY),
                ('BUILD_VERSION', 'MOCK_UNIT_TEST'),
            ]),
            'source_location': {
                'file': mock.ANY,
                'function': 'test_log_no_structure',
                'line': 64,
            },
        },
        severity=cloud_logging_client_instance._LogSeverity.INFO,
    )

  @flagsaver.flagsaver(
      pod_hostname='test_pod', pod_uid='123', enable_structured_logging=False
  )
  @mock.patch.object(absl_logging, 'info', autospec=True)
  def test_log_absl(self, mock_func):
    cl = cloud_logging_client.logger()
    cl.log_signature = {}
    with mock.patch.dict(
        cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
    ):
      cl._log(
          'test',
          cloud_logging_client_instance._LogSeverity.INFO,
          ({'abc': 123},),
      )
    mock_func.assert_called_once_with(
        'test; abc: 123; HOSTNAME: test_pod; POD_UID: 123; THREAD_ID:'
        f' {threading.get_native_id()}; BUILD_VERSION: MOCK_UNIT_TEST'
    )

  @flagsaver.flagsaver(ops_log_project='test')
  def test_flags_initialized(self):
    self.assertTrue(cloud_logging_client._are_flags_initialized())

  def test_flags_not_initialized(self):
    # store original flag replace with mock which will raise exception
    # on access
    ops_log_project_flag = cloud_logging_client.CLOUD_OPS_LOG_PROJECT_FLG
    try:
      flg_mock = mock.Mock()
      type(flg_mock).value = mock.PropertyMock(
          side_effect=flags.UnparsedFlagAccessError
      )
      cloud_logging_client.CLOUD_OPS_LOG_PROJECT_FLG = flg_mock
      self.assertFalse(cloud_logging_client._are_flags_initialized())
    finally:
      cloud_logging_client.CLOUD_OPS_LOG_PROJECT_FLG = ops_log_project_flag

  @parameterized.named_parameters([
      dict(
          testcase_name='small_msg_unclipped',
          test_struct={'test1': 'foo', 'test2': 'bar'},
          expected={'test1': 'foo', 'test2': 'bar'},
      ),
      dict(
          testcase_name='clip_msg_proportionally',
          test_struct={'test1': 'foo', 'test2': 'bar', 'test3': 'sho'},
          expected={'test1': 'f', 'test2': 'b', 'test3': 'sho'},
      ),
      dict(
          testcase_name='clip_large_msg_slective',
          test_struct={'test1': 'fo', 'test2': 'b', 'test3': 'very_large_msg'},
          expected={'test1': 'fo', 'test2': 'b', 'test3': 've'},
      ),
      dict(
          testcase_name='clip_large_msg_slective_and_proportionally',
          test_struct={
              'test1': 'foo',
              'test2': 'bar',
              'test3': 'very_large_msg',
          },
          expected={'test1': 'f', 'test2': 'b', 'test3': 'ver'},
      ),
  ])
  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_clip_strict_log(self, test_struct, expected):
    """Tests clipping msg size < max does not change msg."""
    max_log_size = 20
    cloud_logging_client.logger()._clip_struct_log(test_struct, max_log_size)
    self.assertEqual(test_struct, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='large_nonresizable_components',
          test_struct={
              'test1': 'fo',
              'test2': 'b',
              'test3': 'very_large_msg',
              'HOSTNAME': 'very_long',
              'message': 'message_text_is_much',
              'POD_UID': str(list(range(20))),
          },
      ),
      dict(
          testcase_name='cannot_clip_large_key',
          test_struct={
              'test1': 'foo',
              'test2': 'bar',
              'test3': 'sho',
              'key_is_way_to_big_test3': 'sho',
          },
      ),
  ])
  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_clip_strict_log_raises_value_error(self, test_struct):
    max_log_size = 20
    with self.assertRaises(ValueError):
      cloud_logging_client.logger()._clip_struct_log(test_struct, max_log_size)

  def test_merge_struct_empty_input(self):
    self.assertIsNone(cloud_logging_client_instance._merge_struct(tuple()))

  def test_merge_struct_nop(self):
    self.assertEqual(
        {'foo': 'bar'},
        cloud_logging_client_instance._merge_struct(({'foo': 'bar'},)),
    )

  def test_merge_struct_none_and_dict(self):
    self.assertEqual(
        {'foo': 'bar'},
        cloud_logging_client_instance._merge_struct((None, {'foo': 'bar'})),
    )

  def test_merge_struct_return_sorted_dict(self):
    """Test return ordered dict."""
    ret_dict = cloud_logging_client_instance._merge_struct((
        {'foo': 'bar', 'shoe': 'far'},
    ))
    self.assertEqual({'foo': 'bar', 'shoe': 'far'}, ret_dict)
    self.assertEqual(['foo', 'shoe'], list(ret_dict))

  def test_merge_struct_merge_and_return_sorted_dict(self):
    """Test multiple dict structure merge."""
    ret_dict = cloud_logging_client_instance._merge_struct(
        ({'foo': 'bar', 'shoe': 'far'}, {'abc': 'efg', 'google': 'brain'})
    )
    self.assertEqual(
        {'foo': 'bar', 'shoe': 'far', 'google': 'brain', 'abc': 'efg'}, ret_dict
    )
    self.assertEqual(['foo', 'shoe', 'abc', 'google'], list(ret_dict))

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_default_signature(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    default_tags = cl._merge_signature(None)
    self.assertEqual(
        {
            'HOSTNAME': 'test_pod',
            'POD_UID': '123',
            'THREAD_ID': str(threading.get_native_id()),
        },
        default_tags,
    )

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_merge_default_signature(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    default_tags = cl._merge_signature({'test': 'test_val'})
    self.assertEqual(
        {
            'test': 'test_val',
            'HOSTNAME': 'test_pod',
            'POD_UID': '123',
            'THREAD_ID': str(threading.get_native_id()),
        },
        default_tags,
    )

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_merge_default_signature_no_structure(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    default_tags = cl._merge_signature(None)
    # Remove build version from signature dictionary to make test robust.
    # Build version is automatically generated for each build.
    self.assertEqual(
        {
            'HOSTNAME': 'test_pod',
            'POD_UID': '123',
            'THREAD_ID': str(threading.get_native_id()),
        },
        default_tags,
    )

  def test_merge_struct_empty(self):
    self.assertIsNone(cloud_logging_client_instance._merge_struct(tuple()))

  def test_merge_simple_mapping(self):
    test_mapping = cloud_logging_client_instance._merge_struct((
        {'abc': '123'},
    ))
    expected_mapping = {'abc': '123'}
    self.assertEqual(test_mapping, expected_mapping)

  def test_merge_simple_multiple_mapping(self):
    test_mapping = cloud_logging_client_instance._merge_struct(
        ({'abc': '123'}, {'efg': '456'})
    )
    expected_mapping = {'abc': '123', 'efg': '456'}
    self.assertEqual(test_mapping, expected_mapping)

  @parameterized.parameters(['', 'something bad'])
  @mock.patch.object(traceback, 'format_exc', autospec=True)
  def test_merge_simple_multiple_mapping_and_exception(
      self, exception_text, mock_function
  ):
    stack_trace = 'UnitTestStackTrace'
    mock_function.return_value = stack_trace
    try:
      raise ValueError(exception_text)
    except ValueError as exp:
      test_mapping = cloud_logging_client_instance._merge_struct(
          ({'abc': '123'}, {'efg': '456'}, exp)
      )
    expected_mapping = {
        'abc': '123',
        'efg': '456',
        'exception': (
            f'{exception_text}\n{stack_trace}'
            if exception_text
            else stack_trace
        ),
    }
    self.assertEqual(test_mapping, expected_mapping)

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_set_log_signature(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    cl.log_signature = {'test': 'test_val'}
    log_sig = cl.log_signature
    self.assertEqual(
        log_sig,
        {
            'test': 'test_val',
            'HOSTNAME': 'test_pod',
            'POD_UID': '123',
            'THREAD_ID': str(threading.get_native_id()),
        },
    )

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_clear_log_signature(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    cl.log_signature = {'test': 'test_val'}
    cl.clear_log_signature()
    log_sig = cl.log_signature
    self.assertEqual(
        log_sig,
        {
            'HOSTNAME': 'test_pod',
            'POD_UID': '123',
            'THREAD_ID': str(threading.get_native_id()),
        },
    )

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_host(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    self.assertEqual(cl.hostname, 'test_pod')

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  def test_poduid(self):
    """Test merge logging signature with empt message structure."""
    cl = cloud_logging_client.logger()
    self.assertEqual(cl.pod_uid, '123')

  @mock.patch.object(absl_logging, 'info', autospec=True)
  def test_absl_info_log(self, mocked_method):
    cloud_logging_client_instance._absl_log(
        'test', severity=cloud_logging_client_instance._LogSeverity.INFO
    )
    mocked_method.assert_called_once_with('test')

  @mock.patch.object(absl_logging, 'debug', autospec=True)
  def test_absl_debug_log(self, mocked_method):
    cloud_logging_client_instance._absl_log(
        'test', severity=cloud_logging_client_instance._LogSeverity.DEBUG
    )
    mocked_method.assert_called_once_with('test')

  @mock.patch.object(absl_logging, 'warning', autospec=True)
  def test_absl_warning_log(self, mocked_method):
    cloud_logging_client_instance._absl_log(
        'test', severity=cloud_logging_client_instance._LogSeverity.WARNING
    )
    mocked_method.assert_called_once_with('test')

  @mock.patch.object(absl_logging, 'error', autospec=True)
  def test_absl_error_log(self, mocked_method):
    cloud_logging_client_instance._absl_log(
        'test', severity=cloud_logging_client_instance._LogSeverity.ERROR
    )
    mocked_method.assert_called_once_with('test')

  @mock.patch.object(absl_logging, 'error', autospec=True)
  def test_absl_critical_log(self, mocked_method):
    cloud_logging_client_instance._absl_log(
        'test', severity=cloud_logging_client_instance._LogSeverity.CRITICAL
    )
    mocked_method.assert_called_once_with('test')

  @mock.patch.object(
      cloud_logging_client.CloudLoggingClient, '_log', autospec=True
  )
  def test_debug_log(self, mocked_method):
    logger_instance = cloud_logging_client.logger()

    logger_instance.debug('test')

    mocked_method.assert_called_once_with(
        logger_instance,
        'test',
        cloud_logging_client_instance._LogSeverity.DEBUG,
        tuple(),
        1,
    )

  @mock.patch.object(
      cloud_logging_client.CloudLoggingClient, '_log', autospec=True
  )
  @mock.patch.object(time, 'time', side_effect=[2.22, 5.55], autospec=True)
  @flagsaver.flagsaver(ops_log_project='test_project')
  def test_timed_debug_log(self, _, mocked_method):
    logger_instance = cloud_logging_client.logger()

    logger_instance.timed_debug('test')

    mocked_method.assert_called_once_with(
        logger_instance,
        '[3.330] test',
        cloud_logging_client_instance._LogSeverity.DEBUG,
        tuple(),
        1,
    )

  @mock.patch.object(
      cloud_logging_client.CloudLoggingClient, '_log', autospec=True
  )
  def test_info_log(self, mocked_method):
    logger_instance = cloud_logging_client.logger()

    logger_instance.info('test')

    mocked_method.assert_called_once_with(
        logger_instance,
        'test',
        cloud_logging_client_instance._LogSeverity.INFO,
        tuple(),
        1,
    )

  @mock.patch.object(
      cloud_logging_client.CloudLoggingClient, '_log', autospec=True
  )
  def test_warning_log(self, mocked_method):
    logger_instance = cloud_logging_client.logger()

    logger_instance.warning('test')

    mocked_method.assert_called_once_with(
        logger_instance,
        'test',
        cloud_logging_client_instance._LogSeverity.WARNING,
        tuple(),
        1,
    )

  @mock.patch.object(
      cloud_logging_client.CloudLoggingClient, '_log', autospec=True
  )
  def test_error_log(self, mocked_method):
    logger_instance = cloud_logging_client.logger()

    logger_instance.error('test')

    mocked_method.assert_called_once_with(
        logger_instance,
        'test',
        cloud_logging_client_instance._LogSeverity.ERROR,
        tuple(),
        1,
    )

  @mock.patch.object(
      cloud_logging_client.CloudLoggingClient, '_log', autospec=True
  )
  def test_critical_log(self, mocked_method):
    logger_instance = cloud_logging_client.logger()

    logger_instance.critical('test')

    mocked_method.assert_called_once_with(
        logger_instance,
        'test',
        cloud_logging_client_instance._LogSeverity.CRITICAL,
        tuple(),
        1,
    )

  @flagsaver.flagsaver(pod_hostname='test_pod', pod_uid='123')
  @mock.patch.object(inspect.Traceback, 'lineno', 64)
  @mock.patch.object(cloud_logging_client_instance, '_py_log', autospec=True)
  @mock.patch.object(traceback, 'format_exc', autospec=True)
  def test_logging_info_e2e(self, mock_function, mock_pylogger):
    """End-to-end unit test from logging interface to cloud logging log_struct.

    The cloud logging functions, cl._google_logger.log_struct, tested here are
    execulded from most of the unit testing due to the inability to communicate
    with cloud logging from the unit test. In this and similar unit tests.
    the logging client is initialized as in other unit tests with the logging
    to cloud disabled (debug_logging_use_absl_logging=True). The tested code
    path is then enabled by mocking the cloud_logging_client_instance property
    use_absl_logging=False. The test here validates that the logging
    interface is being called with the expected parameters. The test invalidates
    the logging interface singleton to force the singleton to be re-created in
    subsequent unit tests.

    Args:
      mock_function: Traceback.format_exc mock.
      mock_pylogger: Mock wrapper for python logging.
    """
    exception_text = 'something bad'
    stack_trace = 'UnitTestStackTrace'
    mock_function.return_value = stack_trace
    with flagsaver.flagsaver(debug_logging_use_absl_logging=True):
      cl = cloud_logging_client.logger()
    with mock.patch.object(
        cloud_logging_client_instance.CloudLoggingClientInstance,
        'use_absl_logging',
        autospec=True,
        return_value=False,
    ):
      try:
        raise ValueError(exception_text)
      except ValueError as exp:
        cl.log_signature = {}
        with mock.patch.dict(
            cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
        ):
          cl.info('test', {'abc': 123}, {'457': 789}, exp)
    mock_pylogger.assert_called_once_with(
        logging.getLogger(),
        'test',
        extra={
            'json_fields': collections.OrderedDict([
                ('abc', '123'),
                ('457', '789'),
                ('exception', 'something bad\nUnitTestStackTrace'),
                ('HOSTNAME', 'test_pod'),
                ('POD_UID', '123'),
                ('THREAD_ID', mock.ANY),
                ('BUILD_VERSION', 'MOCK_UNIT_TEST'),
            ]),
            'source_location': {
                'file': mock.ANY,
                'function': 'test_logging_info_e2e',
                'line': 64,
            },
        },
        severity=cloud_logging_client_instance._LogSeverity.INFO,
    )

  @flagsaver.flagsaver(pod_hostname='test_pod')
  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_threaded_signature_logging(self, mock_log):
    cl = cloud_logging_client.logger()
    cl.log_signature = {'Thread': 'Main'}
    with mock.patch.dict(
        cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
    ):
      cl.info('main_test_start')
      for x in range(3):
        th = threading.Thread(target=_logger_test_thread, args=(x,))
        th.start()
        th.join()
      # test expected number of messages were logged
      self.assertLen(mock_log.call_args_list, 4)

      main_log = []
      thread_id = set()
      log_regex = re.compile(r'(.+); HOSTNAME: .*; THREAD_ID: (.+)')
      for call in mock_log.call_args_list:
        match = log_regex.fullmatch(call[0][0])
        main_log.append(match.groups()[0])  # pytype: disable=attribute-error
        thread_id.add(match.groups()[1])  # pytype: disable=attribute-error

      # test that the thread's log signature was used for all messages
      self.assertEqual(
          main_log,
          [
              'main_test_start; Thread: Main',
              'Thread_test_0; Thread: 0',
              'Thread_test_1; Thread: 1',
              'Thread_test_2; Thread: 2',
          ],
      )
      # validate that each messages was logged in different thread
      self.assertLen(thread_id, 4)

  @flagsaver.flagsaver(pod_hostname='test_pod')
  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_threaded_signature_logging_with_thread_id(self, mock_log):
    cl = cloud_logging_client.logger()
    cl.log_signature = {'Thread': 'Main'}
    with mock.patch.dict(
        cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
    ):
      cl.info('main_test_start')
      for x in range(3):
        th = threading.Thread(target=_logger_test_thread, args=(x,))
        th.start()
        th.join()
      self.assertLen(mock_log.call_args_list, 4)

      main_log = []
      log_regex = re.compile(r'(.+); HOSTNAME: .*; THREAD_ID: (.+)')
      for call in mock_log.call_args_list:
        match = log_regex.fullmatch(call[0][0])
        main_log.append(match.groups()[0])  # pytype: disable=attribute-error

      # test that the thread's log signature was used for all messages
      self.assertEqual(
          main_log,
          [
              'main_test_start; Thread: Main',
              'Thread_test_0; Thread: 0',
              'Thread_test_1; Thread: 1',
              'Thread_test_2; Thread: 2',
          ],
      )

  @parameterized.parameters([({'Thread': 'Main'}, ' Thread: Main;'), ({}, '')])
  @flagsaver.flagsaver(pod_hostname='test_pod')
  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_threaded_signature_logging_with_invalid_thread_id(
      self, log_signature, expected_txt, mock_log
  ):
    thread_id = threading.get_native_id()
    cl = cloud_logging_client.logger()
    cl.log_signature = log_signature
    with mock.patch.dict(
        cl._thread_local_storage.signature, _MOCK_BUILD_VERSION
    ):
      cl.info('main_test_start')
      # test that the thread's log signature was used for all messages
      mock_log.assert_called_once_with(
          (
              f'main_test_start;{expected_txt} HOSTNAME: test_pod;'
              f' THREAD_ID: {thread_id}; BUILD_VERSION: MOCK_UNIT_TEST'
          ),
          severity=cloud_logging_client_instance._LogSeverity.INFO,
      )

  def test_logging_client_removes_lock_from_get_state(self):
    logger_instance = cloud_logging_client.logger()
    self.assertNotIn('_log_lock', logger_instance.__getstate__())

  def test_logging_client__setstate__initalizes_new_lock(self):
    logger_instance = cloud_logging_client.logger()
    intial_lock = logger_instance._log_lock

    logger_instance.__setstate__(logger_instance.__getstate__())

    self.assertIsNotNone(logger_instance._log_lock)
    self.assertIsNot(intial_lock, logger_instance._log_lock)

  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_logging_not_called_if_disabled(self, mocked_method):
    logger_instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        enabled=False
    )
    logger_instance.error('test')
    mocked_method.assert_not_called()

  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_logging_is_called_if_enabled(self, mocked_method):
    logger_instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        enabled=True
    )
    logger_instance.error('test')
    mocked_method.assert_called_once()

  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_logging_is_called_if_at_error_level(self, mocked_method):
    logger_instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        enabled=True
    )
    logger_instance.log_error_level = logging.ERROR
    logger_instance.error('test')
    mocked_method.assert_called_once()

  @mock.patch.object(cloud_logging_client_instance, '_absl_log')
  def test_logging_is_not_called_if_below_error_level(self, mocked_method):
    logger_instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        enabled=True
    )
    logger_instance.log_error_level = logging.ERROR
    logger_instance.debug('test')
    logger_instance.warning('test')
    mocked_method.assert_not_called()

  def test_cloud_logging_client_init_fork_module_state(self):
    cloud_logging_client.CloudLoggingClient._singleton_instance = 'mock'
    cloud_logging_client.CloudLoggingClient._startup_message_logged = 'mock'
    cloud_logging_client.CloudLoggingClient._singleton_lock = None

    cloud_logging_client.CloudLoggingClient._init_fork_module_state()

    self.assertTrue(
        cloud_logging_client.CloudLoggingClient._startup_message_logged
    )
    self.assertIsNotNone(
        cloud_logging_client.CloudLoggingClient._singleton_lock
    )
    self.assertNotEqual(
        cloud_logging_client.CloudLoggingClient._singleton_lock, 'mock'
    )
    self.assertIsNone(
        cloud_logging_client.CloudLoggingClient._singleton_instance
    )

  def test_cloud_logging_client_instance_init_fork_module_state(self):
    cloud_logging_client_instance.CloudLoggingClientInstance._global_lock = (
        'mock'
    )
    cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params = (
        'mock'
    )
    cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler = (
        'mock'
    )

    cloud_logging_client_instance.CloudLoggingClientInstance._init_fork_module_state()

    self.assertIsNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
    )
    self.assertIsNotNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._global_lock
    )
    self.assertNotEqual(
        cloud_logging_client_instance.CloudLoggingClientInstance._global_lock,
        'mock',
    )
    self.assertEmpty(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params
    )

  def test_fork_shutdown(self):
    cloud_logging_client.CloudLoggingClient._singleton_instance = 'mock'

    cloud_logging_client.CloudLoggingClient._fork_shutdown()

    self.assertIsNone(
        cloud_logging_client.CloudLoggingClient._singleton_instance
    )

  def test_cloud_logging_client_instance_fork_shutdown_no_handler(self):
    cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler = (
        None
    )
    cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params = (
        'mock'
    )
    cloud_logging_client_instance.CloudLoggingClientInstance.fork_shutdown()
    self.assertEmpty(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params
    )

  def test_cloud_logging_client_instance_fork_shutdown_handler(self):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      client = cloud_logging.Client()
    handler = cloud_logging.handlers.CloudLoggingHandler(
        client=client,
        name='foo',
    )
    self.assertNotIn(handler, logging.getLogger().handlers)
    logging.getLogger().addHandler(handler)
    try:
      self.assertIn(handler, logging.getLogger().handlers)
      cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler = (
          handler
      )
      cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params = (
          'mock'
      )

      cloud_logging_client_instance.CloudLoggingClientInstance.fork_shutdown()

      self.assertNotIn(handler, logging.getLogger().handlers)
      self.assertIsNone(
          cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
      )
      self.assertEmpty(
          cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params
      )
    finally:
      logging.getLogger().removeHandler(handler)

  def test_starting_multiple_instances_throws(self):
    cloud_logging_client.CloudLoggingClient()
    with self.assertRaises(
        cloud_logging_client_instance.CloudLoggerInstanceExceptionError
    ):
      cloud_logging_client.CloudLoggingClient()

  @flagsaver.flagsaver(debug_logging_use_absl_logging=False)
  def test_first_logger_calls_startup(self):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      with mock.patch.object(
          cloud_logging_client.CloudLoggingClient,
          'debug',
          autospec=True,
      ) as mock_debug:
        cloud_logging_client.logger().info('test')
    self.assertEqual(mock_debug.call_count, 5)
    self.assertTrue(
        cloud_logging_client.CloudLoggingClient._startup_message_logged
    )

  @flagsaver.flagsaver(debug_logging_use_absl_logging=False)
  def test_second_logger_call_not_call_startup(self):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      cloud_logging_client.logger().info('test')
      with mock.patch.object(
          cloud_logging_client.CloudLoggingClient,
          'debug',
          autospec=True,
      ) as mock_debug:
        cloud_logging_client.logger().info('test')
    mock_debug.assert_not_called()
    self.assertTrue(
        cloud_logging_client.CloudLoggingClient._startup_message_logged
    )

  def test_log_error_level(self):
    logger = cloud_logging_client_instance.CloudLoggingClientInstance()
    self.assertEqual(
        logger.log_error_level,
        cloud_logging_client_instance._LogSeverity.DEBUG.value,
    )
    test_value = cloud_logging_client_instance._LogSeverity.DEBUG.value + 5
    logger.log_error_level = test_value
    self.assertEqual(logger.log_error_level, test_value)

  @flagsaver.flagsaver(debug_logging_use_absl_logging=False)
  def test_cloud_logging_client_instance_python_logger_init_if_none(self):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      instance = cloud_logging_client_instance.CloudLoggingClientInstance(
          use_absl_logging=False
      )
      cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler = (
          None
      )
    self.assertIsNotNone(instance.python_logger)

  @parameterized.named_parameters(
      dict(
          testcase_name='enabled_false_absl_false',
          enabled=False,
          use_absl_logging=False,
      ),
      dict(
          testcase_name='enabled_false_absl_true',
          enabled=False,
          use_absl_logging=True,
      ),
      dict(
          testcase_name='enabled_true_absl_true',
          enabled=True,
          use_absl_logging=True,
      ),
  )
  def test_client_instance_not_logging_to_cloud_init(
      self, enabled, use_absl_logging
  ):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        enabled=enabled, use_absl_logging=use_absl_logging
    )
    self.assertIsNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
    )
    self.assertEmpty(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params
    )
    self.assertIs(instance.python_logger, logging.getLogger())

  @parameterized.named_parameters([
      dict(
          testcase_name='log_dpas_only',
          logger_name='DPASLogger',
          log_all=False,
      ),
      dict(
          testcase_name='capture_all_python_logs',
          logger_name=None,
          log_all=True,
      ),
  ])
  def test_client_instance_cloud_logging_init(
      self,
      log_all,
      logger_name,
  ):
    project_name = 'test_project'
    log_name = 'test_log'
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      instance = cloud_logging_client_instance.CloudLoggingClientInstance(
          gcp_project_to_write_logs_to=project_name,
          log_name=log_name,
          log_all_python_logs_to_cloud=log_all,
          use_absl_logging=False,
      )
    self.assertIsNotNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
    )
    self.assertIn(
        project_name,
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params,
    )
    self.assertIn(
        log_name,
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params,
    )
    self.assertIs(instance.python_logger, logging.getLogger(logger_name))
    self.assertFalse(instance.use_absl_logging())

  @mock.patch.object(
      cloud_logging,
      'Client',
      autospec=True,
      side_effect=google.auth.exceptions.DefaultCredentialsError(),
  )
  def test_client_instance_cloud_logging_init_auth_error(
      self, unused_mock_client
  ):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        gcp_project_to_write_logs_to='test_project',
        log_name='test_log',
        log_all_python_logs_to_cloud=False,
        use_absl_logging=False,
    )
    self.assertIsNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
    )
    self.assertEmpty(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params
    )
    self.assertIs(instance.python_logger, logging.getLogger())
    self.assertTrue(instance.use_absl_logging())

  @mock.patch.object(
      cloud_logging,
      'Client',
      autospec=True,
      side_effect=ValueError(),
  )
  def test_client_instance_cloud_logging_init_unexpected_exception(
      self, unused_mock_client
  ):
    with self.assertRaises(ValueError):
      cloud_logging_client_instance.CloudLoggingClientInstance(
          gcp_project_to_write_logs_to='test_project',
          log_name='test_log',
          log_all_python_logs_to_cloud=False,
          use_absl_logging=False,
      )
    self.assertIsNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
    )
    self.assertEmpty(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='log_dpas_only',
          logger_name='DPASLogger',
          log_all=False,
      ),
      dict(
          testcase_name='capture_all_python_logs',
          logger_name=None,
          log_all=True,
      ),
  ])
  def test_duplicate_compatiable_handler_init_succeeds(
      self,
      log_all,
      logger_name,
  ):
    project_name = 'test_project'
    log_name = 'test_log'
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      cloud_logging_client_instance.CloudLoggingClientInstance(
          gcp_project_to_write_logs_to=project_name,
          log_name=log_name,
          log_all_python_logs_to_cloud=log_all,
          use_absl_logging=False,
      )
      instance = cloud_logging_client_instance.CloudLoggingClientInstance(
          gcp_project_to_write_logs_to=project_name,
          log_name=log_name,
          log_all_python_logs_to_cloud=log_all,
          use_absl_logging=False,
      )
    self.assertIsNotNone(
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
    )
    self.assertIn(
        project_name,
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params,
    )
    self.assertIn(
        log_name,
        cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler_init_params,
    )
    self.assertIs(instance.python_logger, logging.getLogger(logger_name))
    self.assertFalse(instance.use_absl_logging())

  @parameterized.named_parameters([
      dict(
          testcase_name='project_different',
          project_name_1='P1',
          log_name_1='log',
          log_all_1=True,
          project_name_2='P2',
          log_name_2='log',
          log_all_2=True,
      ),
      dict(
          testcase_name='log_name_different',
          project_name_1='Proj',
          log_name_1='log_1',
          log_all_1=True,
          project_name_2='Proj',
          log_name_2='log_2',
          log_all_2=True,
      ),
      dict(
          testcase_name='log_all_different',
          project_name_1='Proj',
          log_name_1='log',
          log_all_1=True,
          project_name_2='Proj',
          log_name_2='log',
          log_all_2=False,
      ),
  ])
  def test_duplicate_incompatiable_loggers_fail(
      self,
      project_name_1,
      log_name_1,
      log_all_1,
      project_name_2,
      log_name_2,
      log_all_2,
  ):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      cloud_logging_client_instance.CloudLoggingClientInstance(
          gcp_project_to_write_logs_to=project_name_1,
          log_name=log_name_1,
          log_all_python_logs_to_cloud=log_all_1,
          use_absl_logging=False,
      )
      with self.assertRaises(
          cloud_logging_client_instance.CloudLoggerInstanceExceptionError
      ):
        cloud_logging_client_instance.CloudLoggingClientInstance(
            gcp_project_to_write_logs_to=project_name_2,
            log_name=log_name_2,
            log_all_python_logs_to_cloud=log_all_2,
            use_absl_logging=False,
        )

  def test_enable_e_structured_logging_default(self):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance()
    self.assertTrue(instance.enable_structured_logging)

  def test_enable__structured_logging_init(self):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        enable_structured_logging=False
    )
    self.assertFalse(instance.enable_structured_logging)

  def test_enable_structured_logging_setter(self):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance()
    set_val = not instance.enable_structured_logging
    instance.enable_structured_logging = set_val
    self.assertEqual(instance.enable_structured_logging, set_val)

  def test_default_gcp_project_name(self):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance()
    self.assertEmpty(instance.gcp_project_name)

  def test_init_gcp_project_name(self):
    project_name = 'foo'
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        gcp_project_to_write_logs_to=project_name
    )
    self.assertEqual(instance.gcp_project_name, project_name)

  def test_log_name_default(self):
    instance = cloud_logging_client_instance.CloudLoggingClientInstance()
    self.assertEqual(instance.log_name, 'python')

  def test_log_name_init(self):
    log_name = 'foo'
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        log_name=log_name
    )
    self.assertEqual(instance.log_name, log_name)

  def test_undefined_log_name_raises(self):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      with self.assertRaises(ValueError):
        cloud_logging_client_instance.CloudLoggingClientInstance(
            log_name='', use_absl_logging=False
        )

  @mock.patch.dict(os.environ, {'GOOGLE_CLOUD_PROJECT': 'TEST_PROJECT'})
  def test_undefined_pod_uid_raises(self):
    with self.assertRaises(ValueError):
      instance = cloud_logging_client_instance.CloudLoggingClientInstance(
          pod_uid='', use_absl_logging=False
      )
      _ = instance.pod_uid

  def test_undefined_pod_hostname_raises(self):
    with self.assertRaises(ValueError):
      instance = cloud_logging_client_instance.CloudLoggingClientInstance()
      _ = instance.hostname

  def test_defined_pod_hostname(self):
    pod_hostname = 'foo'
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        pod_hostname=pod_hostname
    )
    self.assertEqual(instance.hostname, pod_hostname)

  def test_defined_pod_pod_uid(self):
    pod_uid = '123'
    instance = cloud_logging_client_instance.CloudLoggingClientInstance(
        pod_uid=pod_uid
    )
    self.assertEqual(instance.pod_uid, pod_uid)

  def test_log_signature_from_new_thread(self):
    signature_main_thread_dict = cloud_logging_client.logger().log_signature
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
      signature_thread = list(
          pool.map(lambda x: cloud_logging_client.logger().log_signature, [1])
      )[0]
    self.assertNotEmpty(signature_main_thread_dict)
    self.assertNotEmpty(signature_thread)
    self.assertNotEqual(
        signature_main_thread_dict['THREAD_ID'], signature_thread['THREAD_ID']
    )
    self.assertNotEqual(signature_main_thread_dict['THREAD_ID'], os.getpid())

  def test_get_per_thread_log_signatures_default(self):
    cl = cloud_logging_client_instance.CloudLoggingClientInstance()
    self.assertTrue(cl.per_thread_log_signatures)

  def test_get_per_thread_log_signatures_constructor_init(self):
    cl = cloud_logging_client_instance.CloudLoggingClientInstance(
        per_thread_log_signatures=False
    )
    self.assertFalse(cl.per_thread_log_signatures)

  @parameterized.parameters([True, False])
  def test_set_per_thread_log_signatures(self, val):
    cl = cloud_logging_client_instance.CloudLoggingClientInstance()
    cl.per_thread_log_signatures = val
    self.assertEqual(cl.per_thread_log_signatures, val)

  def test_per_thread_log_signatures_true_logs_thread_id(self):
    cl = cloud_logging_client_instance.CloudLoggingClientInstance(
        per_thread_log_signatures=True
    )
    log_sig = cl.log_signature
    self.assertIn('THREAD_ID', log_sig)
    self.assertEqual(log_sig['THREAD_ID'], str(threading.get_native_id()))

  def test_per_thread_log_signatures_false_not_log_thread_id(self):
    cl = cloud_logging_client_instance.CloudLoggingClientInstance(
        per_thread_log_signatures=False
    )
    log_sig = cl.log_signature
    self.assertNotIn('THREAD_ID', log_sig)

  @flagsaver.flagsaver(per_thread_log_signatures=False)
  def test_per_thread_log_signatures_flag(self):
    self.assertFalse(cloud_logging_client.logger().per_thread_log_signatures)

  def test_per_thread_log_signatures_flag_default(self):
    self.assertTrue(cloud_logging_client.logger().per_thread_log_signatures)

  @parameterized.named_parameters([
      dict(
          testcase_name='cloud_logging', use_absl_logging=False, expected=False
      ),
      dict(
          testcase_name='abseil_logging', use_absl_logging=True, expected=True
      ),
  ])
  def test_python_logging_handler_reinit_if_called_across_process_forks(
      self, use_absl_logging, expected
  ):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      cl = cloud_logging_client_instance.CloudLoggingClientInstance(
          use_absl_logging=use_absl_logging
      )
      logger_1 = cl.python_logger
      handler_1 = (
          cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
      )
      cloud_logging_client_instance.CloudLoggingClientInstance._init_fork_module_state()

      logger_2 = cl.python_logger
      handler_2 = (
          cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
      )
    self.assertIsNotNone(logger_1)
    self.assertIsNotNone(logger_2)
    self.assertEqual(handler_1 is handler_2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='cloud_logging', use_absl_logging=False, expected=True
      ),
      dict(
          testcase_name='abseil_logging', use_absl_logging=True, expected=True
      ),
  ])
  def test_python_logging_handler_reinit_if_called_repeatedly_in_same_process(
      self, use_absl_logging, expected
  ):
    with mock.patch.object(
        cloud_logging,
        'Client',
        autospec=True,
        return_value=self._mock_client_instance,
    ):
      cl = cloud_logging_client_instance.CloudLoggingClientInstance(
          use_absl_logging=use_absl_logging
      )
      logger_1 = cl.python_logger
      handler_1 = (
          cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
      )
      logger_2 = cl.python_logger
      handler_2 = (
          cloud_logging_client_instance.CloudLoggingClientInstance._cloud_logging_handler
      )
    self.assertIsNotNone(logger_1)
    self.assertIsNotNone(logger_2)
    self.assertEqual(handler_1 is handler_2, expected)

  @flagsaver.flagsaver(ops_log_project='test_project')
  def test_get_cloud_logging_client_project_from_env_var(self):
    self.assertEqual(
        cloud_logging_client.CloudLoggingClient().gcp_project_name,
        'test_project',
    )

  @flagsaver.flagsaver(ops_log_project=None)
  @mock.patch.object(
      google.auth,
      'default',
      autospec=True,
      return_value=(mock.Mock(), 'test_project_2'),
  )
  def test_get_cloud_logging_client_project_from_default_cred(self, _):
    self.assertEqual(
        cloud_logging_client.CloudLoggingClient().gcp_project_name,
        'test_project_2',
    )

  @parameterized.named_parameters([
      dict(
          testcase_name='no_project',
          project='',
          key='key',
          struct={'key': 'value'},
          expected={},
      ),
      dict(
          testcase_name='no_key',
          project='project',
          key='',
          struct={'key': 'value'},
          expected={},
      ),
      dict(
          testcase_name='key_not_in_struct',
          project='project',
          key='key',
          struct={'foo': 'value'},
          expected={},
      ),
      dict(
          testcase_name='key_in_struct',
          project='project',
          key='key',
          struct={'key': 'value'},
          expected={'trace': 'projects/project/traces/value'},
      ),
  ])
  def test_add_trace_to_log(self, project, key, struct, expected):
    self.assertEqual(
        cloud_logging_client_instance._add_trace_to_log(project, key, struct),
        expected,
    )

  def test_trace_empty_if_not_set(self):
    cl = cloud_logging_client.CloudLoggingClient()
    self.assertEmpty(cl.trace_key)

  def test_trace_set_value(self):
    cl = cloud_logging_client.CloudLoggingClient()
    cl.trace_key = 'test_trace'
    self.assertEqual(cl.trace_key, 'test_trace')

  def test_do_not_log_startup_msg(self):
    cloud_logging_client.CloudLoggingClient._startup_message_logged = False
    with mock.patch.object(
        cloud_logging_client.CloudLoggingClient, 'startup_msg', autospec=True
    ) as mock_startup_msg:
      cloud_logging_client.do_not_log_startup_msg()
      cloud_logging_client.info('test_log')
      mock_startup_msg.assert_not_called()
      self.assertTrue(
          cloud_logging_client.CloudLoggingClient._startup_message_logged
      )


if __name__ == '__main__':
  absltest.main()
