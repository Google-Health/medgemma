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

"""Initializes Pete Cloud Logging."""

from typing import Any, Mapping, Optional

import ez_wsi_dicomweb.ez_wsi_logging_factory

from serving.logging_lib import cloud_logging_client


class _EZWSICloudLoggingInterface(
    ez_wsi_dicomweb.ez_wsi_logging_factory.AbstractLoggingInterface
):
  """EZ-WSI Cloud Logging Interface."""

  def __init__(self, signature: Optional[Mapping[str, Any]]):
    self._signature = signature

  def debug(
      self,
      msg: str,
      *args: ez_wsi_dicomweb.ez_wsi_logging_factory.OptionalStructureElements,
  ) -> None:
    cloud_logging_client.debug(msg, *args, self._signature, stack_frames_back=1)

  def info(
      self,
      msg: str,
      *args: ez_wsi_dicomweb.ez_wsi_logging_factory.OptionalStructureElements,
  ) -> None:
    cloud_logging_client.info(msg, *args, self._signature, stack_frames_back=1)

  def warning(
      self,
      msg: str,
      *args: ez_wsi_dicomweb.ez_wsi_logging_factory.OptionalStructureElements,
  ) -> None:
    cloud_logging_client.warning(
        msg, *args, self._signature, stack_frames_back=1
    )

  def error(
      self,
      msg: str,
      *args: ez_wsi_dicomweb.ez_wsi_logging_factory.OptionalStructureElements,
  ) -> None:
    cloud_logging_client.error(msg, *args, self._signature, stack_frames_back=1)

  def critical(
      self,
      msg: str,
      *args: ez_wsi_dicomweb.ez_wsi_logging_factory.OptionalStructureElements,
  ) -> None:
    cloud_logging_client.critical(
        msg, *args, self._signature, stack_frames_back=1
    )


class EZWSILoggingInterfaceFactory(
    ez_wsi_dicomweb.ez_wsi_logging_factory.AbstractLoggingInterfaceFactory
):
  """EZ-WSI Cloud Logging Interface Factory."""

  def __init__(self, signature: Mapping[str, Any]):
    self._signature = signature

  def create_logger(
      self, signature: Optional[Mapping[str, Any]] = None
  ) -> ez_wsi_dicomweb.ez_wsi_logging_factory.AbstractLoggingInterface:
    signature = {} if signature is None else dict(signature)
    signature.update(self._signature)
    return _EZWSICloudLoggingInterface(signature)
