# Public Jetson CUDA 12.6 toolchain; Python and its complete environment are installed below.
FROM ghcr.io/nvidia-ai-iot/vllm@sha256:2a90817f4d760094a25c546126a48aa8e8ae4fae1ae41bf72445fbc2c8bc2a7f
COPY tools/build/compiler-requirements.txt /tmp/orin-requirements.txt
COPY tools/build/jetson_wheels.py tools/build/jetson_wheels.json /tmp/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_DEFAULT_INDEX=https://pypi.org/simple
ENV UV_CONSTRAINT=/dev/null UV_BUILD_CONSTRAINT=/dev/null PIP_CONSTRAINT=/dev/null
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple --no-deps --target /tmp/orinfer-bootstrap uv==0.13.4 \
 && cp /tmp/orinfer-bootstrap/bin/uv /usr/local/bin/uv \
 && rm -rf /tmp/orinfer-bootstrap
RUN /usr/local/bin/uv python install 3.12.15 \
 && /usr/local/bin/uv venv --clear --python 3.12.15 /opt/venv \
 && /opt/venv/bin/python /tmp/jetson_wheels.py /tmp/jetson-wheels \
 && /usr/local/bin/uv pip sync --python /opt/venv/bin/python --no-cache --find-links /tmp/jetson-wheels --index-url https://pypi.org/simple /tmp/orin-requirements.txt \
 && uv pip check --python /opt/venv/bin/python \
 && /opt/venv/bin/python -c "import sys, importlib.metadata as m, torch; assert sys.version_info[:2] == (3, 12); assert m.version('torch') == '2.14.1+orinfer1'; assert torch.version.cuda == '12.6'; assert 'sm_87' in torch._C._cuda_getArchFlags(); assert torch._C.has_lapack; assert m.version('torchvision') == '0.29.1+orinfer1'; assert m.version('tilelang') == '0.1.15'" \
 && rm -rf /tmp/orin-requirements.txt /tmp/jetson-wheels /tmp/jetson_wheels.py /tmp/jetson_wheels.json
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}
ENV PYTHONPATH="" TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
WORKDIR /work
ENTRYPOINT ["/opt/venv/bin/python"]
