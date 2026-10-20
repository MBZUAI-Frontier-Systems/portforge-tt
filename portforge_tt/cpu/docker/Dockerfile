# portforge-rvv-host: the x86-64 side of an RVV port.
#
# One image holds everything a campaign needs that is not the RISC-V board:
#   - native gcc 14 (x86 reference builds of XNNPACK scalar / SSE / AVX kernels)
#   - riscv64 and aarch64 cross gcc 14 (same compiler release as the X280 Debian)
#   - qemu-user (functional RVV at any VLEN; aarch64 to run the Neon source kernel)
#   - cvc5 1.3.2 and SALTyRN's bounded-equivalence verifier (Neon vs RVV, SMT)
#
# Build on the x86 host:
#   docker build -t portforge-rvv-host:0 targets/rvv/docker
FROM docker.io/library/debian@sha256:913f6706df59a68922d1dd08f78c2476560a8d367897200a6005b00e5f67c2d5

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates curl unzip xxd jq \
      python3 python3-venv python3-dev \
      gcc-14 g++-14 gcc g++ \
      gcc-riscv64-linux-gnu g++-riscv64-linux-gnu libc6-dev-riscv64-cross \
      gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross \
      qemu-user ccache time \
      libgmp-dev \
 && rm -rf /var/lib/apt/lists/*

# cvc5 1.3.2, static GPL build: SALTyRN's CMake links cln, glpk and CoCoA from it.
ARG CVC5_ZIP=cvc5-Linux-x86_64-static-gpl.zip
ARG CVC5_SHA256=30abf22d8042f3c4d3eb1120c595abd461f92f276a6d1264895fb4403b5fb3fd
RUN curl -fsSL -o /tmp/cvc5.zip https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.2/${CVC5_ZIP} \
 && echo "${CVC5_SHA256}  /tmp/cvc5.zip" | sha256sum -c - \
 && unzip -q /tmp/cvc5.zip -d /tmp/cvc5 \
 && mv /tmp/cvc5/* /opt/cvc5 \
 && rm -rf /tmp/cvc5 /tmp/cvc5.zip \
 && /opt/cvc5/bin/cvc5 --version | head -1

# SALTyRN at a pinned commit; only the verifier is used (no LLM keys, no Spike).
ARG SALTYRN_REV=6acdf7a522e2b97b96831f8e95b578e6edf42a83
RUN git clone -q https://github.com/keaganchern/SALTy-rn /opt/saltyrn \
 && git -C /opt/saltyrn checkout -q ${SALTYRN_REV} \
 # Upstream CMakeLists keeps a debug target whose source was never committed;
 # drop it so the orchestrator's HARNESS_SRC builds configure.
 && sed -i '/^add_executable(verify_f16_conv_hwc2chw_cvc5/,/^target_link_libraries(verify_f16_conv_hwc2chw_cvc5/d' \
      /opt/saltyrn/src/verification_cvc5/CMakeLists.txt \
 && ! grep -q verify_f16_conv_hwc2chw /opt/saltyrn/src/verification_cvc5/CMakeLists.txt \
 && mkdir -p /opt/saltyrn/third_party/cvc5 \
 && ln -s /opt/cvc5 /opt/saltyrn/third_party/cvc5/install \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install -q libclang==18.1.1 \
 && /opt/venv/bin/pip install -q --no-deps -e /opt/saltyrn

ENV PATH=/opt/venv/bin:/opt/cvc5/bin:$PATH \
    CC=gcc-14 CXX=g++-14 \
    SALTYRN=/opt/saltyrn
WORKDIR /work
