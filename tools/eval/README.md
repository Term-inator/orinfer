# 质量评测工具

`quick_quality.py` 分析同一 teacher-forced 历史下的 baseline/candidate token 概率；概率按完整词表归一化，并显式查询 baseline top-3 IDs。token 身份差异用于诊断，质量结合概率和任务结果判定。

`scoring_common.py` 校验固定 seed、历史身份、概率字段及任务结果。`run_quality.py` 连接配置的评分接口，使用 `fixtures/quick-quality-scenarios.json` 生成评测请求。参考服务需自行提供。

```bash
python3 -m unittest discover -s tools/eval -p 'test_*.py' -v
python3 tools/eval/run_quality.py --help
```

官方 BF16 或同模型 FP8 用作量化质量参考；同 AWQ 权重对照只用于实现诊断。场景检查、target NLL 与长上下文任务需要分别评估。

翻译模型可在 `offline_quality prepare` 时指定 `--fixtures fixtures/translation-quality-scenarios.json`，覆盖中英日德翻译、术语、JSON 占位符、否定和商务语气。参考译文用于固定概率评分历史；自由生成允许其他准确译法。这组快速检查覆盖固定翻译场景；完整多语言 benchmark、图片和多图的视觉概率及 API 任务分别验收。

## 离线 BF16 对照

`reference_bf16.py` 使用支持 Qwen3_5 的 Transformers，从原始 safetensors 逐层加载和释放权重。参考执行完整序列 prefill；`score-model` 执行正常 prefill 及逐 token teacher-forced decode。报告明确标记两条路径，保留完整词表归一化的 top-3、指定 token 概率与实际历史 SHA256。

```bash
python3 -m tools.eval.offline_quality prepare --checkpoint MODEL_DIR --output artifacts/quality/requests.json
python3 -m tools.eval.reference_bf16 --checkpoint BF16_DIR --revision SOURCE_COMMIT \
  --requests artifacts/quality/requests.json --output artifacts/quality/bf16.json
python3 -m tools.eval.offline_quality queries --requests artifacts/quality/requests.json \
  --baseline artifacts/quality/bf16.json --output artifacts/quality/queries.json
target/release/orinfer score-model MODEL_DIR artifacts/quality/queries.json > artifacts/quality/candidate.json
python3 -m tools.eval.offline_quality compare --requests artifacts/quality/queries.json \
  --baseline artifacts/quality/bf16.json --candidate artifacts/quality/candidate.json \
  --output artifacts/quality/comparison.json
```

GPU 任务通过 `flock artifacts/gpu-experiment.lock` 串行执行。参考与 candidate 使用固定 seed20261002 和相同原始 token IDs。场景答案用于冻结评分历史，独立自由生成仍需验证任务结果；该概率比较验证固定历史下的分布；完整 LLM benchmark 和 BF16 autoregressive 路径另行验收。

磁盘空间有限时，可在保留原始配置、tokenizer 和权重 index 的目录上使用 `--remote-repo REPO --revision COMMIT`。COMMIT 必须是固定的 40 位提交，读取器校验 HTTP Range 边界，按文本组件并行读取原始 tensor，记录直接读取的原始权重字节及其 SHA256。`--read-workers` 控制网络并发，范围 1–16。可选的 `--tensor-cache DIR` 消费预取的标准 safetensors，校验来源提交和 payload hash，加载后删除已消费文件。该流式读取用于质量对照，计时包含网络读取；模型启动性能通过独立启动基准测量。

`prepare --long-context-tokens 512 2048 8192` 追加确定性的中部资料检索场景。保留完整 Chat 模板，实际 token 长度接近给定预算且在预算范围内；评分历史固定为正确资料答案。参考的 `--batch-size` 默认 1，可合批以减少读取权重的次数；BF16 舍入也会随 GEMM 形状变化，单流验收应使用单流参考。

图片评分在每个 case 增加 `images` 数组，每项为 `grid_height`、`grid_width` 及归一化 FP32 patch 数组 `pixels`，与在线 `ImageInput` 契约一致。`prompt_ids` 须包含原生模板和展开后的图片占位 token；参考和 candidate 共享相同 patch 输入。参考独立执行官方 BF16 视觉编码器、特征注入和 MRoPE。输入指纹包含图片尺寸、FP32 little-endian 像素 SHA256 和图片顺序，防止错误配对。该对照覆盖编码器至 token 概率；原始图片解码/resize 应另用预处理参考验证，任务结果仍需独立自由生成。

Transformers 的图像参考需要显式传入 `mm_token_type_ids`：展开的 image-pad 为 1，文本及界定符为 0；自由生成也必须保留该字段，确保使用 MRoPE。部署 tokenizer 应合并 `tokenizer_config.json` 的 added-token 配置，并与 Transformers 对照实际 token IDs；相同渲染字符串仍可能被切成不同 tokens。

## 本机 Q4/Q2 任务筛查

`task_quality.py` 生成固定 seed 的 200 条文本功能测试，每类 20 条：计算、数量应用题、JSON、指令更正、资料抽取、访问逻辑、列表变换、代码阅读、翻译约束及长文本检索。访问逻辑的允许和拒绝各占一半。该合成集用于筛查功能退化，真实业务质量需另用业务样本验收。

```bash
.venv/bin/python -m tools.eval.task_quality prepare --output artifacts/task-screen
# 用同一 requests.jsonl、采样参数和上下文预算分别执行各个模型的 orinfer batch。
.venv/bin/python -m tools.eval.task_quality score \
  --suite artifacts/task-screen/suite.json --results RESULTS.jsonl \
  --output artifacts/task-screen/scores.json
```

使用对应模型 tokenizer 预检每条原生 Chat 模板的实际长度与输出预算，保证总和处于执行包容量内。`score` 要求请求 ID 完整且唯一；错误与截断均不计为通过。`strict_tasks` 汇总 180 条固定答案任务，`translation_constraints` 单独报告 20 条翻译的标识符及关键词约束命中；约束命中不代表语义正确，保留原文和回答进行独立语义复核。重复文本标记仅作诊断。Q4/Q2 任务正确率按答案规则分别判定，Q4 不是绝对质量参考。
