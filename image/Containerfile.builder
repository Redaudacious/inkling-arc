# inkling-arc kernel builder: the base image's environment + Intel oneAPI DPC++ 2026.1.1 (icpx, ocloc) + cmake/ninja.
ARG ENV=localhost/vllm-xpu:b70-inkling2
FROM ${ENV}
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
RUN apt-get update -y && apt-get install -y --no-install-recommends gpg wget git && \
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list && \
    apt-get update -y && apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp-2026.1=2026.1.1-325 intel-oneapi-dpcpp-cpp-2026.1=2026.1.1-325 && \
    rm -rf /var/lib/apt/lists/* && \
    (uv pip install "cmake>=3.26" ninja || pip install "cmake>=3.26" ninja) && \
    . /opt/intel/oneapi/setvars.sh --force >/dev/null && icpx --version | grep -q 'Compiler 2026.1.1 '
ENTRYPOINT []
