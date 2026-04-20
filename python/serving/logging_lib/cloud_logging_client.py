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
"""Wrapper for cloud ops structured logging."""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Mapping, Optional, Union

from absl import flags
from absl import logging
import google.auth
import psutil

from serving.logging_lib.flags import secret_flag_utils
from serving.logging_lib import cloud_logging_client_instance

# name of cloud ops log
CLOUD_OPS_LOG_NAME_FLG = flags.DEFINE_string(
    'ops_log_name',
    secret_flag_utils.get_secret_or_env('CLOUD_OPS_LOG_NAME', 'python'),
    'Cloud ops log name to write logs to.',
)
CLOUD_OPS_LOG_PROJECT_FLG = flags.DEFINE_string(
    'ops_log_project',
    secret_flag_utils.get_secret_or_env('CLOUD_OPS_LOG_PROJECT', None),
    'GCP project name to write log to. Undefined = default',
)
POD_HOSTNAME_FLG = flags.DEFINE_string(
    'pod_hostname',
    secret_flag_utils.get_secret_or_env('HOSTNAME', None),
    'Host name of GKE pod. Set by container ENV. '
    'Set to mock value in unit test.',
)
POD_UID_FLG = flags.DEFINE_string(
    'pod_uid',
    secret_flag_utils.get_secret_or_env('MY_POD_UID', None),
    'UID of GKE pod. Do not set unless in test.',
)
ENABLE_STRUCTURED_LOGGING_FLG = flags.DEFINE_boolean(
    'enable_structured_logging',
    secret_flag_utils.get_bool_secret_or_env(
        'ENABLE_STRUCTURED_LOGGING',
        not secret_flag_utils.get_bool_secret_or_env(
            'DISABLE_STRUCTURED_LOGGING'
        ),
    ),
    'Enable structured logging.',
)
ENABLE_LOGGING_FLG = flags.DEFINE_boolean(
    'enable_logging',
    secret_flag_utils.get_bool_secret_or_env('ENABLE_LOGGING_FLG', True),
    'Enable logging.',
)

_DEBUG_LOGGING_USE_ABSL_LOGGING_FLG = flags.DEFINE_boolean(
    'debug_logging_use_absl_logging',
    # Confusing double negative is used to enable external env be a postive
    # statement and override the existing default
    not secret_flag_utils.get_bool_secret_or_env(
        'ENABLE_CLOUD_LOGGING',
        not cloud_logging_client_instance.DEBUG_LOGGING_USE_ABSL_LOGGING,
    ),
    'Debug/testing option to logs to absl.logger. Automatically set when '
    'running unit tests.',
)

LOG_ALL_PYTHON_LOGS_TO_CLOUD_FLG = flags.DEFINE_boolean(
    'log_all_python_logs_to_cloud',
    secret_flag_utils.get_bool_secret_or_env('LOG_ALL_PYTHON_LOGS_TO_CLOUD'),
    'Logs every modules log to Cloud Ops.',
)

PER_THREAD_LOG_SIGNATURES_FLG = flags.DEFINE_boolean(
    'per_thread_log_signatures',
    secret_flag_utils.get_bool_secret_or_env('PER_THREAD_LOG_SIGNATURES', True),
    'If True Log signatures are not shared are across threads if false '
    'Process threads share a common log signature',
)


def _are_flags_initialized() -> bool:
  """Returns True if flags are initialized."""
  try:
    _ = CLOUD_OPS_LOG_PROJECT_FLG.value
    return True
  except (flags.UnparsedFlagAccessError, AttributeError):
    return False


def _get_flags() -> Mapping[str, str]:
  load_flags = {}
  unparsed_flags = []
  for flag_name in flags.FLAGS:
    try:
      load_flags[flag_name] = flags.FLAGS.__getattr__(flag_name)
    except flags.UnparsedFlagAccessError:
      unparsed_flags.append(flag_name)
  if unparsed_flags:
    load_flags['unparsed_flags'] = ', '.join(unparsed_flags)
  return load_flags


def _default_gcp_project() -> str:
  try:
    _, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    return project if project is not None else ''
  except google.auth.exceptions.DefaultCredentialsError:
    return ''


class CloudLoggingClient(
    cloud_logging_client_instance.CloudLoggingClientInstance
):
  """Wrapper for cloud ops structured logging.

  Automatically adds signature to structured logs to make traceable.
  """

  # lock for log makes access to singleton
  # safe across threads. Logging used in main thread and ack_timeout_mon
  _singleton_instance: Optional[CloudLoggingClient] = None
  _startup_message_logged = False
  _singleton_lock = threading.RLock()

  @classmethod
  def _init_fork_module_state(cls) -> None:
    cls._singleton_instance = None
    cls._startup_message_logged = True
    cls._singleton_lock = threading.RLock()

  @classmethod
  def _fork_shutdown(cls) -> None:
    with cls._singleton_lock:
      cls._singleton_instance = None

  @classmethod
  def _set_absl_skip_frames(cls) -> None:
    """Sets absl logging attribution to skip over internal logging frames."""
    logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='debug',
    )
    logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='timed_debug',
    )
    logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='info',
    )
    logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='warning',
    )
    logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='error',
    )
    logging.ABSLLogger.register_frame_to_skip(
        __file__,
        function_name='critical',
    )

  def __init__(self):
    with CloudLoggingClient._singleton_lock:
      if not _are_flags_initialized():
        # if flags are not initialize then init logging flags
        flags.FLAGS(sys.argv, known_only=True)
      if CloudLoggingClient._singleton_instance is not None:
        raise cloud_logging_client_instance.CloudLoggerInstanceExceptionError(
            'Singleton already initialized.'
        )
      CloudLoggingClient._set_absl_skip_frames()
      gcp_project = CLOUD_OPS_LOG_PROJECT_FLG.value
      if gcp_project is None or not gcp_project.strip():
        gcp_project = _default_gcp_project()
      pod_host_name = (
          '' if POD_HOSTNAME_FLG.value is None else POD_HOSTNAME_FLG.value
      )
      pod_uid = '' if POD_UID_FLG.value is None else POD_UID_FLG.value
      super().__init__(
          log_name=CLOUD_OPS_LOG_NAME_FLG.value,
          gcp_project_to_write_logs_to=gcp_project,
          gcp_credentials=None,
          pod_hostname=pod_host_name,
          pod_uid=pod_uid,
          enable_structured_logging=ENABLE_STRUCTURED_LOGGING_FLG.value,
          use_absl_logging=_DEBUG_LOGGING_USE_ABSL_LOGGING_FLG.value,
          log_all_python_logs_to_cloud=LOG_ALL_PYTHON_LOGS_TO_CLOUD_FLG.value,
          per_thread_log_signatures=PER_THREAD_LOG_SIGNATURES_FLG.value,
          enabled=ENABLE_LOGGING_FLG.value,
      )
      CloudLoggingClient._singleton_instance = self

  def startup_msg(self) -> None:
    """Logs default messages after logger fully initialized."""
    if self.use_absl_logging() or CloudLoggingClient._startup_message_logged:
      return
    CloudLoggingClient._startup_message_logged = True
    pid = os.getpid()
    process_name = psutil.Process(pid).name()
    self.debug(
        'Container process started.',
        {'process_name': process_name, 'process_id': pid},
    )
    self.debug(
        'Container environmental variables.', os.environ
    )  # pytype: disable=wrong-arg-types  # kwargs-checking
    vm = psutil.virtual_memory()
    self.debug(
        'Compute instance',
        {
            'processors(count)': os.cpu_count(),
            'total_system_mem_(bytes)': vm.total,
            'available_system_mem_(bytes)': vm.available,
        },
    )
    self.debug('Initalized flags', _get_flags())
    project_name = self.gcp_project_name if self.gcp_project_name else 'DEFAULT'
    self.debug(f'Logging to GCP project: {project_name}')

  @classmethod
  def logger(cls, show_startup_msg: bool = True) -> CloudLoggingClient:
    if cls._singleton_instance is None:
      with cls._singleton_lock:  # makes instance creation thread safe.
        if cls._singleton_instance is None:
          cls._singleton_instance = CloudLoggingClient()
          if not show_startup_msg:
            cls._startup_message_logged = True
          else:
            cls._singleton_instance.startup_msg()  # pytype: disable=attribute-error
    return cls._singleton_instance  # pytype: disable=bad-return-type


def logger() -> CloudLoggingClient:
  return CloudLoggingClient.logger()


def do_not_log_startup_msg() -> None:
  CloudLoggingClient.logger(show_startup_msg=False)


def debug(
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
  logger().debug(msg, *struct, stack_frames_back=stack_frames_back + 1)


def timed_debug(
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
  logger().timed_debug(msg, *struct, stack_frames_back=stack_frames_back + 1)


def info(
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
  logger().info(msg, *struct, stack_frames_back=stack_frames_back + 1)


def warning(
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
  logger().warning(msg, *struct, stack_frames_back=stack_frames_back + 1)


def error(
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
  logger().error(msg, *struct, stack_frames_back=stack_frames_back + 1)


def critical(
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
  logger().critical(msg, *struct, stack_frames_back=stack_frames_back + 1)


def clear_log_signature() -> None:
  logger().clear_log_signature()


def get_log_signature() -> Mapping[str, Any]:
  return logger().log_signature


def set_log_signature(sig: Mapping[str, Any]) -> None:
  logger().log_signature = sig


def set_per_thread_log_signatures(val: bool) -> None:
  logger().per_thread_log_signatures = val


def get_build_version(clip_length: Optional[int] = None) -> str:
  if clip_length is not None and clip_length >= 0:
    return logger().build_version[:clip_length]
  return logger().build_version


def set_build_version(build_version: str) -> None:
  logger().build_version = build_version


def set_log_trace_key(key: str) -> None:
  logger().trace_key = key


# Logging interfaces are used from processes which are forked (gunicorn,
# DICOM Proxy, Orchestrator, Refresher). In Python, forked processes do not
# copy threads running within parent processes or re-initalize global/module
# state. This can result in forked modules being executed with invalid global
# state, e.g., acquired locks that will not release or references to invalid
# state. The cloud logging library utilizes a background thread transporting
# logs to cloud. The background threading is not compatiable with forking and
# will seg-fault (python queue wait). This  can be avoided, by stoping and
# the background transport prior to forking and then restarting the transport
# following the fork.
os.register_at_fork(
    before=CloudLoggingClient._fork_shutdown,  # pylint: disable=protected-access
    after_in_child=CloudLoggingClient._init_fork_module_state,  # pylint: disable=protected-access
)
