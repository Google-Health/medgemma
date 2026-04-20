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
"""Wrapper for cloud ops structured logging.

Get instance of logger:
  logger() -> CloudLoggingClient

  logger().log_signature  = dict to append to logs.
"""
import collections
import copy
import enum
import inspect
import logging
import math
import os
import sys
import threading
import time
import traceback
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging as absl_logging
import google.auth
from google.cloud import logging as cloud_logging

# Debug/testing option to logs to absl.logger. Automatically set when
# running unit tests
DEBUG_LOGGING_USE_ABSL_LOGGING = bool(
    'UNITTEST_ON_FORGE' in os.environ or 'unittest' in sys.modules
)


class _LogSeverity(enum.Enum):
  CRITICAL = logging.CRITICAL
  ERROR = logging.ERROR
  WARNING = logging.WARNING
  INFO = logging.INFO
  DEBUG = logging.DEBUG


MAX_LOG_SIZE = 246000


class CloudLoggerInstanceExceptionError(Exception):
  pass


def _merge_struct(
    dict_tuple: Tuple[Union[Mapping[str, Any], Exception, None], ...],
) -> Optional[MutableMapping[str, str]]:
  """Merges a list of dict and ordered dicts.

       * for dict adds item in key sorted order
       * preserves order for ordered dicts.
  Args:
    dict_tuple: dicts and ordered dicts to merge

  Returns:
     merged dict.
  """
  if not dict_tuple:
    return None
  return_dict = collections.OrderedDict()
  for dt in dict_tuple:
    if dt is None:
      continue
    if isinstance(dt, Exception):
      # Log exception text and exception stack trace.
      exception_str = str(dt)
      if exception_str:
        exception_str = f'{exception_str}\n'
      return_dict['exception'] = f'{exception_str}{traceback.format_exc()}'
    else:
      keylist = list(dt)
      if not isinstance(dt, collections.OrderedDict):
        keylist = sorted(keylist)
      for key in keylist:
        return_dict[key] = str(dt[key])
  return return_dict


def _absl_log(msg: str, severity: _LogSeverity = _LogSeverity.INFO) -> None:
  """Logs using absl logging.

  Args:
    msg: Message to log.
    severity: Severity of message.
  """
  if severity == _LogSeverity.DEBUG:
    absl_logging.debug(msg)
  elif severity == _LogSeverity.WARNING:
    absl_logging.warning(msg)
  elif severity == _LogSeverity.INFO:
    absl_logging.info(msg)
  else:
    absl_logging.error(msg)


def _py_log(
    dpas_logger: logging.Logger,
    msg: str,
    extra: Mapping[str, Any],
    severity: _LogSeverity,
) -> None:
  """Logs msg and structured logging using python logger."""
  if severity == _LogSeverity.DEBUG:
    dpas_logger.debug(msg, extra=extra)
  elif severity == _LogSeverity.WARNING:
    dpas_logger.warning(msg, extra=extra)
  elif severity == _LogSeverity.INFO:
    dpas_logger.info(msg, extra=extra)
  elif severity == _LogSeverity.CRITICAL:
    dpas_logger.critical(msg, extra=extra)
  elif severity == _LogSeverity.ERROR:
    dpas_logger.error(msg, extra=extra)
  else:
    raise CloudLoggerInstanceExceptionError(
        f'Unsupported logging severity level; Severity="{severity}"'
    )


def _get_source_location_to_log(stack_frames_back: int) -> Mapping[str, Any]:
  """Adds Python source location information to cloud structured logs.

  The source location is added by adding (and overwriting if present) a
  "source_location" key to the provided additional_parameters.
  The value corresponding to that key is a dict mapping:
     "file" to the name of the file (str) containing the logging statement,
     "function" python function/method calling logging method
     "line" to the line number where the log was recorded (int).

  Args:
    stack_frames_back: Additional stack frames back to log source_location.

  Returns:
    Source location formatted for structured logging.

  Raises:
    ValueError: If stack frame cannot be found for specified position.
  """
  source_location = {}
  current_frame = inspect.currentframe()
  for _ in range(stack_frames_back + 1):
    if current_frame is None:
      raise ValueError('Cannot get stack frame for specified position.')
    current_frame = current_frame.f_back
  try:
    frame_info = inspect.getframeinfo(current_frame)
    source_location['source_location'] = dict(
        file=frame_info.filename,
        function=frame_info.function,
        line=frame_info.lineno,
    )
  finally:
    # https://docs.python.org/3/library/inspect.html
    del current_frame  # explicitly deleting
  return source_location


def _add_trace_to_log(
    project_id: str, trace_key: str, struct: Mapping[str, Any]
) -> Mapping[str, Any]:
  if not project_id or not trace_key:
    return {}
  trace_id = struct.get(trace_key, '')
  if trace_id:
    return {'trace': f'projects/{project_id}/traces/{trace_id}'}
  return {}


class CloudLoggingClientInstance:
  """Wrapper for cloud ops structured logging.

  Automatically adds signature to structured logs to make traceable.
  """

  # global state to prevent duplicate initialization of cloud logging interfaces
  # within a process.
  _global_lock = threading.Lock()
  # Cloud logging handler init at process level.
  _cloud_logging_handler: Optional[
      cloud_logging.handlers.CloudLoggingHandler
  ] = None
  _cloud_logging_handler_init_params = ''

  @classmethod
  def _init_fork_module_state(cls) -> None:
    cls._global_lock = threading.Lock()
    cls._cloud_logging_handler = None
    cls._cloud_logging_handler_init_params = ''

  @classmethod
  def fork_shutdown(cls) -> None:
    with cls._global_lock:
      cls._cloud_logging_handler_init_params = ''
      handler = cls._cloud_logging_handler
      if handler is None:
        return
      handler.transport.worker.stop()
      logging.getLogger().removeHandler(handler)
      handler.close()
      cls._cloud_logging_handler = None

  @classmethod
  def _set_absl_skip_frames(cls) -> None:
    """Sets absl logging attribution to skip over internal logging frames."""
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='_log',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='_absl_log',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='debug',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='timed_debug',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='info',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='warning',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='error',
    )
    absl_logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='critical',
    )

  def __init__(
      self,
      log_name: str = 'python',
      gcp_project_to_write_logs_to: str = '',
      gcp_credentials: Optional[google.auth.credentials.Credentials] = None,
      pod_hostname: str = '',
      pod_uid: str = '',
      enable_structured_logging: bool = True,
      use_absl_logging: bool = DEBUG_LOGGING_USE_ABSL_LOGGING,
      log_all_python_logs_to_cloud: bool = False,
      enabled: bool = True,
      log_error_level: int = _LogSeverity.DEBUG.value,
      per_thread_log_signatures: bool = True,
      trace_key: str = '',
      build_version: str = '',
  ):
    """Constructor.

    Args:
      log_name: Log name to write logs to.
      gcp_project_to_write_logs_to: GCP project name to write log to. Undefined
        = default.
      gcp_credentials: The OAuth2 Credentials to use for this client
        (None=default).
      pod_hostname: Host name of GKE pod. Should be empty if not running in GKE.
      pod_uid: UID of GKE pod. Should be empty if not running in GKE.
      enable_structured_logging: Enable structured logging.
      use_absl_logging: Send logs to absl logging instead of cloud_logging.
      log_all_python_logs_to_cloud: Logs everything to cloud.
      enabled: If disabled, logging is not initialized and logging operations
        are nops.
      log_error_level: Error level at which logger will log.
      per_thread_log_signatures: Log signatures reported per thread.
      trace_key: Log key value which contains a trace id value.
      build_version: Build version to embedd in logs container.
    """
    # lock for log makes access to singleton
    # safe across threads. Logging used in main thread and ack_timeout_mon
    CloudLoggingClientInstance._set_absl_skip_frames()
    self._build_version = build_version
    self._enabled = enabled
    self._trace_key = trace_key
    self._log_error_level = log_error_level
    self._log_lock = threading.RLock()
    self._log_name = log_name.strip()
    self._pod_hostname = pod_hostname.strip()
    self._pod_uid = pod_uid.strip()
    self._per_thread_log_signatures = per_thread_log_signatures
    self._thread_local_storage = threading.local()
    self._shared_log_signature = self._signature_defaults(0)
    self._enable_structured_logging = enable_structured_logging
    self._debug_log_time = time.time()
    self._gcp_project_name = gcp_project_to_write_logs_to.strip()
    self._use_absl_logging = use_absl_logging
    self._log_all_python_logs_to_cloud = log_all_python_logs_to_cloud
    self._gcp_credentials = gcp_credentials
    absl_logging.set_verbosity(absl_logging.INFO)
    self._python_logger = self._init_cloud_handler()

  @property
  def trace_key(self) -> str:
    return self._trace_key

  @trace_key.setter
  def trace_key(self, val: str) -> None:
    self._trace_key = val

  @property
  def per_thread_log_signatures(self) -> bool:
    return self._per_thread_log_signatures

  @per_thread_log_signatures.setter
  def per_thread_log_signatures(self, val: bool) -> None:
    with self._log_lock:
      self._per_thread_log_signatures = val

  @property
  def python_logger(self) -> logging.Logger:
    if (
        self._enabled
        and not self._use_absl_logging
        and CloudLoggingClientInstance._cloud_logging_handler is None
    ):
      self._python_logger = self._init_cloud_handler()
    return self._python_logger

  def _get_python_logger_name(self) -> Optional[str]:
    return None if self._log_all_python_logs_to_cloud else 'DPASLogger'

  def _get_cloud_logging_handler_init_params(self) -> str:
    return (
        f'GCP_PROJECT_NAME: {self._gcp_project_name}; LOG_NAME:'
        f' {self._log_name}; LOG_ALL: {self._log_all_python_logs_to_cloud}'
    )

  def _init_cloud_handler(self) -> logging.Logger:
    """Initializes cloud logging handler and returns python logger."""
    # Instantiates a cloud logging client to generate text logs for cloud
    # operations
    with CloudLoggingClientInstance._global_lock:
      if not self._enabled or self._use_absl_logging:
        return logging.getLogger()  # Default PY logger
      handler_instance_init_params = (
          self._get_cloud_logging_handler_init_params()
      )
      if CloudLoggingClientInstance._cloud_logging_handler is not None:
        running_handler_init_params = (
            CloudLoggingClientInstance._cloud_logging_handler_init_params
        )
        if running_handler_init_params != handler_instance_init_params:
          # Call fork_shutdown to shutdown the process's named logging handler.
          raise CloudLoggerInstanceExceptionError(
              'Cloud logging handler is running with parameters that do not'
              ' match instance defined parameters. Running handler parameters:'
              f' {running_handler_init_params}; Instance parameters:'
              f' {handler_instance_init_params}'
          )
        return logging.getLogger(self._get_python_logger_name())
      log_name = self.log_name
      struct_log = {}
      struct_log['log_name'] = log_name
      struct_log['log_all_python_logs'] = self._log_all_python_logs_to_cloud
      try:
        # Attach default python & absl logger to also write to named log.
        logging_client = cloud_logging.Client(
            project=self._gcp_project_name if self._gcp_project_name else None,
            credentials=self._gcp_credentials,
        )
        logging_client.project = (
            self._gcp_project_name if self._gcp_project_name else None
        )
        handler = cloud_logging.handlers.CloudLoggingHandler(
            client=logging_client,
            name=log_name,
        )
        CloudLoggingClientInstance._cloud_logging_handler = handler
        CloudLoggingClientInstance._cloud_logging_handler_init_params = (
            handler_instance_init_params
        )
        cloud_logging.handlers.setup_logging(
            handler,
            log_level=logging.DEBUG
            if self._log_all_python_logs_to_cloud
            else logging.INFO,
        )
        dpas_python_logger = logging.getLogger(self._get_python_logger_name())
        dpas_python_logger.setLevel(
            logging.DEBUG
        )  # pytype: disable=attribute-error
        return dpas_python_logger
      except google.auth.exceptions.DefaultCredentialsError as exp:
        self._use_absl_logging = True
        self.error('Error initializing logging.', struct_log, exp)
        return logging.getLogger()
      except Exception as exp:
        self._use_absl_logging = True
        self.error('Error unexpected exception.', struct_log, exp)
        raise

  def __getstate__(self) -> MutableMapping[str, Any]:
    """Returns log state for pickle removes lock."""
    dct = copy.copy(self.__dict__)
    del dct['_log_lock']
    del dct['_python_logger']
    del dct['_thread_local_storage']
    return dct

  def __setstate__(self, dct: MutableMapping[str, Any]):
    """Un-pickles class and re-creates log lock."""
    self.__dict__ = dct
    self._log_lock = threading.RLock()
    self._thread_local_storage = threading.local()
    # Re-init logging in process.
    self._python_logger = self._init_cloud_handler()

  def use_absl_logging(self) -> bool:
    return self._use_absl_logging

  @property
  def enable_structured_logging(self) -> bool:
    return self._enable_structured_logging

  @enable_structured_logging.setter
  def enable_structured_logging(self, val: bool) -> None:
    with self._log_lock:
      self._enable_structured_logging = val

  @property
  def enable(self) -> bool:
    return self._enabled

  @enable.setter
  def enable(self, val: bool) -> None:
    with self._log_lock:
      self._enabled = val

  @property
  def gcp_project_name(self) -> str:
    return self._gcp_project_name

  @property
  def log_name(self) -> str:
    if not self._log_name:
      raise ValueError('Undefined Log Name')
    return self._log_name

  @property
  def hostname(self) -> str:
    if not self._pod_hostname:
      raise ValueError('POD_HOSTNAME name is not defined.')
    return self._pod_hostname

  @property
  def build_version(self) -> str:
    """Returns build version # for container."""
    return self._build_version

  @build_version.setter
  def build_version(self, version: str) -> None:
    """Returns build version # for container."""
    self._build_version = version
    with self._log_lock:
      if self._per_thread_log_signatures:
        if not hasattr(self._thread_local_storage, 'signature'):
          self._thread_local_storage.signature = self._signature_defaults(
              threading.get_native_id()
          )
        log_sig = self._thread_local_storage.signature
      else:
        log_sig = self._shared_log_signature
      if not version and 'BUILD_VERSION' in log_sig:
        del log_sig['BUILD_VERSION']
      else:
        log_sig['BUILD_VERSION'] = str(version)

  @property
  def pod_uid(self) -> str:
    if not self._pod_uid:
      raise ValueError('Undefined POD UID')
    return self._pod_uid

  def _get_thread_signature(self) -> MutableMapping[str, Any]:
    if not self._per_thread_log_signatures:
      return self._shared_log_signature
    if not hasattr(self._thread_local_storage, 'signature'):
      self._thread_local_storage.signature = self._signature_defaults(
          threading.get_native_id()
      )
    return self._thread_local_storage.signature

  @property
  def log_signature(self) -> MutableMapping[str, Any]:
    """Returns log signature.

    Log signature returned may not match what is currently being logged.
    if thread is set to log using another threads log signature.
    """
    with self._log_lock:
      return copy.copy(self._get_thread_signature())

  def _signature_defaults(self, thread_id: int) -> MutableMapping[str, str]:
    """Returns default log signature."""
    log_signature = collections.OrderedDict()
    if self._pod_hostname:
      log_signature['HOSTNAME'] = str(self._pod_hostname)
    if self._pod_uid:
      log_signature['POD_UID'] = str(self._pod_uid)
    if self.build_version:
      log_signature['BUILD_VERSION'] = str(self.build_version)
    if self._per_thread_log_signatures:
      log_signature['THREAD_ID'] = str(thread_id)
    return log_signature

  @log_signature.setter
  def log_signature(self, sig: Mapping[str, Any]) -> None:
    """Sets log signature.

    Log signature of thread may not be altered if thread is set to log using
    another threads log signature.

    Args:
      sig: Signature for threads to logs to use.
    """
    with self._log_lock:
      if self._per_thread_log_signatures:
        thread_id = threading.get_native_id()
        if not hasattr(self._thread_local_storage, 'log_signature'):
          self._thread_local_storage.signature = collections.OrderedDict()
        log_sig = self._thread_local_storage.signature
      else:
        thread_id = 0
        log_sig = self._shared_log_signature
      log_sig.clear()
      if sig is not None:
        for key in sorted(sig):
          log_sig[str(key)] = str(sig[key])
      log_sig.update(self._signature_defaults(thread_id))

  def clear_log_signature(self) -> None:
    """Clears thread log signature."""
    with self._log_lock:
      if self._per_thread_log_signatures:
        self._thread_local_storage.signature = self._signature_defaults(
            threading.get_native_id()
        )
      else:
        self._shared_log_signature = self._signature_defaults(0)

  def _clip_struct_log(
      self, log: MutableMapping[str, Any], max_log_size: int
  ) -> None:
    """Clip log if structed log exceeds structured log size limits.

    Clipping logic:
      log size = total sum of key + value sizes of log structure

      log['message'] and signature components are not clipped to keep
      log message text un-altered and message traceability preserved.
      log structure keys are not altered.

      Structured logs exceeding size typically have a massive component which.
      First try to clip the log by just clipping the largest clippable value.
      If log still exceeds size. Proportionally clip log values.

    Args:
      log: Structured log.
      max_log_size: Max_size of the log.

    Returns:
      None

    Raises:
      ValueError if log cannot be clipped to maxsize.
    """
    # determine length of total message key + value
    total_size = 0
    total_value_size = 0
    for key, value in log.items():
      total_size += len(key)
      total_value_size += len(value)
    total_size += total_value_size
    exceeds_log = total_size - max_log_size
    if exceeds_log <= 0:
      return

    # remove keys for log values not being adjusted
    # message is not adjust
    # message signature is not adjusted
    log_keys = set(log)
    excluded_key_msg_length = 0
    excluded_key_msg_value_length = 0
    excluded_keys = list(self._get_thread_signature())
    excluded_keys.append('message')
    for excluded_key in excluded_keys:
      if excluded_key not in log_keys:
        continue
      value_len = len(str(log[excluded_key]))
      excluded_key_msg_length += len(excluded_key) + value_len
      excluded_key_msg_value_length += value_len
      log_keys.remove(excluded_key)
    if excluded_key_msg_length >= max_log_size:
      raise ValueError('Message exceeds logging msg length limit.')
    total_value_size -= excluded_key_msg_value_length

    # message exceeded length limits due to components in structure log
    self._log(
        'Next log message exceed cloud ops length limit and as clipped.',
        severity=_LogSeverity.WARNING,
        struct=tuple(),
        stack_frames_back=0,
    )

    # First clip largest entry. In most cases a message will have one
    # very large tag which causes the length issue first clip the single
    # largest entry so its at most not bigger than the second largest entry.
    if len(log_keys) > 1:
      key_size_list = []
      for key in log_keys:
        key_size_list.append((key, len(log[key])))
      key_size_list = sorted(key_size_list, key=lambda x: x[1])
      largest_key = key_size_list[-1][0]
      # difference in size between largest and second largest entry
      largest_key_size_delta = key_size_list[-1][1] - key_size_list[-2][1]
      clip_len = min(largest_key_size_delta, exceeds_log)
      if clip_len > 0:
        log[largest_key] = log[largest_key][:-clip_len]
        # adjust length that needs to be trimmed
        exceeds_log -= clip_len
        if exceeds_log == 0:
          return
        # adjust total size of trimmable value component
        total_value_size -= clip_len

    # Proportionally clip all tags
    new_exceeds_log = exceeds_log
    # iterate over a sorted list to make clipping deterministic
    for key in sorted(list(log_keys)):
      entry_size = len(log[key])
      clip_len = math.ceil(entry_size * exceeds_log / total_value_size)
      clip_len = min(min(clip_len, new_exceeds_log), entry_size)
      if clip_len > 0:
        log[key] = log[key][:-clip_len]
        new_exceeds_log -= clip_len
        if new_exceeds_log == 0:
          return
    raise ValueError('Message exceeds logging msg length limit.')

  def _merge_signature(
      self, struct: Optional[MutableMapping[str, Any]]
  ) -> MutableMapping[str, Any]:
    """Adds signature to logging struct.

    Args:
      struct: logging struct.

    Returns:
      Dict to log
    """
    if struct is None:
      struct = collections.OrderedDict()
    struct.update(self._get_thread_signature())
    return struct

  def _log(
      self,
      msg: str,
      severity: _LogSeverity,
      struct: Tuple[Union[Mapping[str, Any], Exception, None], ...],
      stack_frames_back: int = 0,
  ):
    """Posts structured log message, adds current_msg id to log structure.

    Args:
      msg: Message to log.
      severity: Severity level of message.
      struct: Structure to log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    if not self._enabled:
      return
    with self._log_lock:
      if severity.value < self._log_error_level:
        return
      struct = self._merge_signature(_merge_struct(struct))
      if not self.use_absl_logging() and self._enable_structured_logging:
        # Log using structured logs
        source_location = _get_source_location_to_log(stack_frames_back + 1)
        trace = _add_trace_to_log(
            self._gcp_project_name, self._trace_key, struct
        )
        self._clip_struct_log(struct, MAX_LOG_SIZE)
        _py_log(
            self.python_logger,
            msg,
            extra={'json_fields': struct, **source_location, **trace},
            severity=severity,
        )
        return

      # Log using unstructured logs.
      structure_str = [msg]
      for key in struct:
        structure_str.append(f'{key}: {struct[key]}')
      _absl_log('; '.join(structure_str), severity=severity)

  def debug(
      self,
      msg: str,
      *struct: Union[Mapping[str, Any], Exception, None],
      stack_frames_back: int = 0,
  ) -> None:
    """Logs with debug severity.

    Args:
      msg: message to log (string).
      *struct: zero or more dict or exception to log in structured log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    self._log(msg, _LogSeverity.DEBUG, struct, 1 + stack_frames_back)

  def timed_debug(
      self,
      msg: str,
      *struct: Union[Mapping[str, Any], Exception, None],
      stack_frames_back: int = 0,
  ) -> None:
    """Logs with debug severity and elapsed time since last timed debug log.

    Args:
      msg: message to log (string).
      *struct: zero or more dict or exception to log in structured log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    time_now = time.time()
    elapsed_time = '%.3f' % (time_now - self._debug_log_time)
    self._debug_log_time = time_now
    msg = f'[{elapsed_time}] {msg}'
    self._log(msg, _LogSeverity.DEBUG, struct, 1 + stack_frames_back)

  def info(
      self,
      msg: str,
      *struct: Union[Mapping[str, Any], Exception, None],
      stack_frames_back: int = 0,
  ) -> None:
    """Logs with info severity.

    Args:
      msg: message to log (string).
      *struct: zero or more dict or exception to log in structured log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    self._log(msg, _LogSeverity.INFO, struct, 1 + stack_frames_back)

  def warning(
      self,
      msg: str,
      *struct: Union[Mapping[str, Any], Exception, None],
      stack_frames_back: int = 0,
  ) -> None:
    """Logs with warning severity.

    Args:
      msg: Message to log (string).
      *struct: Zero or more dict or exception to log in structured log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    self._log(msg, _LogSeverity.WARNING, struct, 1 + stack_frames_back)

  def error(
      self,
      msg: str,
      *struct: Union[Mapping[str, Any], Exception, None],
      stack_frames_back: int = 0,
  ) -> None:
    """Logs with error severity.

    Args:
      msg: Message to log (string).
      *struct: Zero or more dict or exception to log in structured log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    self._log(msg, _LogSeverity.ERROR, struct, 1 + stack_frames_back)

  def critical(
      self,
      msg: str,
      *struct: Union[Mapping[str, Any], Exception, None],
      stack_frames_back: int = 0,
  ) -> None:
    """Logs with critical severity.

    Args:
      msg: Message to log (string).
      *struct: Zero or more dict or exception to log in structured log.
      stack_frames_back: Additional stack frames back to log source_location.
    """
    self._log(msg, _LogSeverity.CRITICAL, struct, 1 + stack_frames_back)

  @property
  def log_error_level(self) -> int:
    return self._log_error_level

  @log_error_level.setter
  def log_error_level(self, level: int) -> None:
    with self._log_lock:
      self._log_error_level = level


# Logging interfaces are used from processes which are forked (gunicorn,
# DICOM Proxy, Orchestrator, Refresher). In Python, forked processes do not
# copy threads running within parent processes or re-initialize global/module
# state. This can result in forked modules being executed with invalid global
# state, e.g., acquired locks that will not release or references to invalid
# state. The cloud logging library utilizes a background thread transporting
# logs to cloud. The background threading is not compatible with forking and
# will seg-fault (python queue wait). This can be avoided, by stopping the
# background transport prior to forking and then restarting the transport
# following the fork.
os.register_at_fork(
    before=CloudLoggingClientInstance.fork_shutdown,  # pylint: disable=protected-access
    after_in_child=CloudLoggingClientInstance._init_fork_module_state,  # pylint: disable=protected-access
)
