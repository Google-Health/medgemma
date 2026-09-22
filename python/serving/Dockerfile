# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is used to build a Docker image that includes the necessary dependencies
# for running MedGemma as a microservice.

FROM python:3.12-slim AS prep

# Get pip requirements
COPY ./python/serving/requirements.txt requirements.txt
COPY ./python/serving/serving_framework/pip-install.txt pip-install.txt
RUN python3.12 -m venv --copies /payload/server-env && \
    /payload/server-env/bin/python3.12 -m pip install --require-hashes  \
        --no-input -r pip-install.txt && \
    /payload/server-env/bin/python3.12 -m pip install --require-hashes \
        --no-input -r requirements.txt

# Set up code directories
COPY ./python/data_accessors /payload/data_accessors
COPY ./python/data_processing /payload/data_processing
COPY ./python/serving /payload/serving
COPY ./LICENSE /payload/LICENSE
RUN mv /payload/serving/model_configuration /payload/model_repository && \
    chmod a+x /payload/serving/entrypoint.sh

# Install git and clone source code for mirroring
RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/certifi/python-certifi.git /source-mirror/python-certifi && \
    git clone https://github.com/tqdm/tqdm.git /source-mirror/tqdm && \
    git clone https://git.launchpad.net/launchpadlib /source-mirror/launchpadlib && \
    git clone https://git.launchpad.net/lazr.restfulclient /source-mirror/lazr.restfulclient && \
    git clone https://git.launchpad.net/lazr.uri /source-mirror/lazr.uri && \
    git clone https://git.launchpad.net/wadllib /source-mirror/wadllib && \
    git clone https://github.com/openslide/openslide-bin.git /source-mirror/openslide-bin && \
    git clone https://github.com/openslide/openslide-python.git /source-mirror/openslide-python && \
    git clone https://github.com/pycountry/pycountry.git /source-mirror/pycountry && \
    git clone https://gitlab.gnome.org/GNOME/pygobject.git /source-mirror/pygobject && \
    git clone https://git.launchpad.net/python-apt /source-mirror/python-apt

# ----------------------------------------------------------------------------

FROM google/cloud-sdk:stable AS gcloud_builder

# Remove vulnerable gcloud components (crc32c, bundled cryptography and setuptools)
RUN rm -f /usr/lib/google-cloud-sdk/bin/gcloud-crc32c && \
    rm -rf /usr/lib/google-cloud-sdk/platform/bundledpythonunix/lib/python*/site-packages/cryptography* && \
    rm -rf /usr/lib/google-cloud-sdk/platform/bundledpythonunix/lib/python*/site-packages/setuptools*

# ----------------------------------------------------------------------------

FROM nvcr.io/nvidia/tritonserver:26.08-vllm-python-py3

WORKDIR /

# Patch OS-level vulnerabilities while preserving CUDA/NVIDIA library
# compatibility. These packages are tightly coupled to PyTorch and vLLM;
# upgrading them breaks shared library linkage (e.g. libcusparseLt.so.0).
# Remove OS python3-pip/wheel packages — their dpkg metadata triggers
# vulnerability scanner findings with no upstream Ubuntu fix available.
# Remove vLLM code to eradicate uncompiled Rust Cargo lockfiles pinning vulnerable OpenSSL versions.
RUN dpkg -l | awk '/^ii/ && ($2 ~ /cuda|nccl|cublas|cusparse|cudnn|cutensor|cufft|curand|cusolver|nvjpeg|libnv|tensorrt/) {print $2}' | xargs apt-mark hold && \
    apt-get update && \
    apt-get -y upgrade && \
    (apt-get remove -y "^slurm.*" "^libslurm.*" "^libpmi.*" || true) && \
    apt-get remove -y python3-pip python3-pip-whl python3-wheel && \
    rm -rf /opt/vllm/vllm-src/rust/ && \
    rm -rf /var/lib/apt/lists/*

# Copy in minimal gcloud install
COPY --from=gcloud_builder /usr/lib/google-cloud-sdk /usr/lib/google-cloud-sdk
ENV PATH="/usr/lib/google-cloud-sdk/bin:$PATH"

# Copy the code and python environment from the prep image
COPY --from=prep /payload /

# Fix vulnerable packages in the base image system site-packages.
# pytest and virtualenv are not needed at runtime; removing virtualenv also
# eliminates its embedded pip/wheel wheels that trigger scanner findings.
# See model_requirements.txt for the full list of upgraded packages and CVEs.
COPY ./python/serving/model_requirements.txt /model_requirements.txt
RUN python3 -m pip install --require-hashes -r /model_requirements.txt && \
    python3 -m pip uninstall -y pytest virtualenv setuptools && \
    rm /model_requirements.txt

# Copy mirrored source code from the prep image
COPY --from=prep /source-mirror /source-mirror/

ENTRYPOINT ["/serving/entrypoint.sh"]
