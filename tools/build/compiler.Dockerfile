# Minimal amd64 CUDA base; Python CUDA libraries are installed once by uv.
FROM nvidia/cuda@sha256:e869de8a68e95272e51c4c7071b1d39941aaab707f9bc051a99062a3791448b1
COPY tools/build/compiler-requirements.txt /tmp/orin-requirements.txt
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++ python3-venv ca-certificates \
      cuda-nvcc-12-8=12.8.93-1 cuda-cudart-dev-12-8=12.8.90-1 cuda-nvrtc-dev-12-8=12.8.93-1 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --no-cache-dir --index-url https://pypi.org/simple --no-deps uv==0.13.1 \
 && UV_CONSTRAINT=/dev/null UV_BUILD_CONSTRAINT=/dev/null /opt/venv/bin/uv pip sync --python /opt/venv/bin/python --no-cache --index-url https://pypi.org/simple /tmp/orin-requirements.txt \
 && /opt/venv/bin/uv pip check --python /opt/venv/bin/python \
 && /opt/venv/bin/python -c "import importlib.metadata as m, platform, torch; assert platform.machine() == 'x86_64'; assert torch.__version__.split('+')[0] == '2.9.1'; assert torch.version.cuda == '12.8'; assert m.version('tilelang') == '0.1.15'" \
 && rm /tmp/orin-requirements.txt
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /work
ENTRYPOINT ["/opt/venv/bin/python"]
