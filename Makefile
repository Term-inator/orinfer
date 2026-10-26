.PHONY: check fmt lint build info python-env compiler-image check-offline check-gpu

PYTHON ?= .venv/bin/python

python-env:
	uv venv --allow-existing --python python3.12 .venv
	uv pip sync --python $(PYTHON) tools/build/cpu-requirements.txt

build:
	bash tools/build/native.sh

fmt:
	cargo fmt --all
	$(PYTHON) -m ruff format tools kernels

lint:
	cargo fmt --all -- --check
	$(PYTHON) -m ruff format --check tools kernels
	$(PYTHON) -m ruff check tools kernels
	cargo clippy --workspace --all-targets --locked --offline -- -D warnings

check: lint
	cargo check --workspace --locked --offline
	cargo test --workspace --locked --offline
	$(PYTHON) -m unittest tools.build.test_files tools.build.test_jetson_wheels tools.model.test_tuning -v
	$(PYTHON) -m unittest discover -s tools/api -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/eval -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/bench -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/operators -p 'test_abi.py' -v
	$(PYTHON) -m unittest discover -s tools/vision -p 'test_*.py' -v
	$(PYTHON) -m unittest discover -s tools/quantization -p 'test_*.py' -v
	$(PYTHON) -m unittest tools.model.test_safetensors_source -v
	$(PYTHON) -m unittest tools.model.test_compile tools.model.flash_next.tests.test_storage tools.model.flash_next.tests.test_precision tools.model.flash_next.tests.test_hc -v
	$(PYTHON) -m unittest tools.model.qwen35.test_build tools.model.qwen35.test_parallel tools.model.qwen35.test_attention tools.model.qwen35.test_q2 tools.model.qwen35.test_rebuild_attention tools.model.qwen35.test_rebuild_mtp -v
	$(PYTHON) -m unittest tools.model.nemotron_embed.test_config tools.model.qwen_reranker.test_config -v
	$(PYTHON) -m unittest tools.model.test_compact -v
	$(PYTHON) -m unittest tools.model.flash_next.tests.test_decode_package tools.model.flash_next.tests.test_decode_policy tools.model.flash_next.tests.test_kernel_policy tools.model.flash_next.tests.test_repack -v
	$(PYTHON) -m unittest tools.model.flash_next.tests.test_weights -v
	$(PYTHON) -m unittest tools.model.test_row_selection tools.model.flash_next.tests.test_convergence tools.model.flash_next.tests.test_build -v
	$(PYTHON) -m unittest tools.model.flash_next.tests.test_ple tools.model.flash_next.tests.test_chunks tools.model.flash_next.tests.test_validation tools.model.flash_next.tests.test_original tools.model.flash_next.tests.test_teacher_inputs tools.model.flash_next.tests.test_speculation tools.model.flash_next.tests.test_config -v
	$(PYTHON) -m unittest tools.model.test_attach_execution tools.model.test_package tools.model.test_catalog tools.model.test_publication tools.model.test_row_contract tools.model.test_vision_capacity tools.release.test_package tools.test_reference -v
	$(PYTHON) tools/bench/validate_plan.py

info:
	cargo run --offline -p orinfer-cli -- info

compiler-image:
	docker build $(DOCKER_BUILD_ARGS) -t orinfer-compiler:0.2.0-sm120 -f tools/build/compiler.Dockerfile .

check-offline:
	docker run --rm --runtime runc --network none -v "$(CURDIR):$(CURDIR):ro" -w "$(CURDIR)" -e PYTHONPATH="$(CURDIR):$(CURDIR)/tools/operators" --entrypoint /opt/venv/bin/python orinfer-compiler:0.2.0-sm120 -m tools.build.check_imports
	docker run --rm --runtime runc --network none -v "$(CURDIR):$(CURDIR):ro" -w "$(CURDIR)" -e PYTHONPATH="$(CURDIR)" --entrypoint /opt/venv/bin/python orinfer-compiler:0.2.0-sm120 -m unittest tools.model.test_frontend tools.model.qwen35.test_reference tools.model.flash_next.tests.test_scenes tools.model.flash_next.tests.test_reference_math tools.model.flash_next.tests.test_teacher tools.model.flash_next.tests.test_qsa_reference tools.model.flash_next.tests.test_native_contract tools.model.flash_next.tests.test_expert_pipeline tools.model.flash_next.tests.test_dynamic_batch -v

check-gpu:
	$(PYTHON) tools/bench/check_gpu.py --model "$(MODEL)" --output "$(OUTPUT)"
