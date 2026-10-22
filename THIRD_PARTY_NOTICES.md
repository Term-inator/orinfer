Selected test vectors and adapted test scenarios originate from
[Ollama](https://github.com/ollama/ollama), Copyright (c) Ollama, under the MIT
License. The pinned revision, source test names and adaptations are recorded in
`fixtures/ollama/cases.json`; the full license is in `fixtures/ollama/LICENSE`.

The offline compiler image uses PyTorch and TorchVision built for Jetson by
[our PyTorch fork](https://github.com/iMMIQ/pytorch). The build profile records the
upstream revisions, NVIDIA toolchain image, and pinned dependencies.
`tools/build/jetson_wheels.json` pins the release artifacts and SHA256 values.
The wheels retain their upstream BSD licenses and bundled third-party notices.
