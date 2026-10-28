# 离线量化

CPU 工具使用项目 uv 环境，GPU 编码使用固定的 x86_64/SM120 编译镜像 `orinfer-compiler:0.2.0-sm120`。编码、存储重排和执行包构建分别完成；量化产物通过独立质量验收后才能部署。

E8P Q2 编解码、GPU 算子、Q2_K 对比及两级 Q4 扩展见[量化技术 Wiki](../../docs/quantization.md)。

## 当前表示

| 用途 | 权重表示 | 缩放与计算 |
| --- | --- | --- |
| Flash Next 专家及 PLE | 整数 E8P Q2，U16 描述 8 个权重 | 输出行 FP16 scale，解码到 INT8 |
| 27B 主干与输出头 | 两级整数 E8P Q4，U32 描述 8 个权重 | 固定系数 `[4,1]`，输出行 FP16 scale，A8 计算 |
| Index-Translate-2B/9B 主干 | 两级整数 E8P Q4 | 固定系数 `[4,1]`，输出行 FP16 scale，A8 计算 |
| Qwen3-ASR-1.7B 文本主干与共享输出头 | 两级整数 E8P Q4 | `[4,1]`、输出行 FP16 scale、A8；前 4 层与 attention 输出保留 W8，音频编码器和 projector 保留 FP16 |
| Qwen3-Reranker-0.6B | 两级整数 E8P Q4 | `[4,1]`、A8；attention 输出保留 W8，norm 与原始 yes/no 双行分类头保留 FP16；见[重排说明](../../docs/reranking.md) |
| Flash Next 部分非专家投影 | 旋转标量码表 Q4 | signed-block128，输出行 FP16 scale，A8 计算 |

敏感部分保留 W8、FP16、BF16 或 FP32。scale、basis、旋转符号、padding 和例外权重均计入实际空间；权重常驻为单份压缩 bank，W8 展开限定用于临时计算。

## E8P 编码与布局

`e8p.py` 从格点几何生成整数 basis，`e8p_gpu.py` 使用 TileLang 最近邻编码器拟合原始浮点矩阵。basis 及其顺序属于格式身份；`vq.py` 提供 E8P 码字解码、selector 表、无损 K64 重排与 safetensors 数值夹具。

`e8p_q4.py` 是两级 Q4 的 CPU 参考与存储定义。`e8p_q4_gpu.Encoder.fit()` 分块拟合两个码字，`Encoder.refine(raw, original)` 以原始浮点权重执行交替更新及 beam8 联合搜索。当前 27B 使用 `[4,1]` 和 beam8；两个阶段共享 basis，每个输出行共用一个 FP16 scale。保存的码字为 U32 `[N,K/8]`，构建器无损重排为计算用的 K128 布局。

调用者按输出行分块持有原始权重，例如每次 128 行；`raw` 必须直接读取原始 BF16/FP8 权重。存储、FP16 scale、旋转和系数均须与执行包匹配。

```bash
uv run --locked python -m unittest tools.quantization.test_e8p tools.quantization.test_e8p_q4
bash tools/operators/run.sh tools/quantization/check_e8p_q4.py artifacts/check-e8p-q4
bash tools/operators/run.sh tools/quantization/check_e8p_search.py artifacts/check-e8p-search
```

`check_e8p_direct.py` 验证完整 A8 位平面和两个 INT4 Tensor Core 点积恢复完整 A8 的原始 INT32 点积。`evaluate_e8p_q4.py` 比较有原始权重来源的组件夹具，输出重构误差与投影时延；模型 NLL 和完整请求性能另外测量。

## 权重准备与参考

Flash Next 的固定来源转换、PLE、非专家权重准备和原始 BF16 teacher 见[模型构建](../model/flash_next/README.md)。

`row_q4.py` 与 `row_q4_gpu.py` 实现当前标量 Q4 组件，码表为 `±(3,10,19,31,46,67,93,127)`。`qwen35.py` 为 27B 分块准备标量 Q4、W8 例外及原始张量；两级 E8P 编码与 beam8 优化由 `qwen_e8p.py` 完成。27B E8P 部署向[原生构建器](../model/qwen35/README.md)提供完成两级编码的 checkpoint。

源信息需包含固定的 repository/revision、配置和 safetensors 索引。远端权重可按字节范围读取；已下载分片通过 `--source-dir` 复用。准备工具核对来源、编码实现、seed 和已完成分块，仅复用来源、身份及完整性均匹配的产物。

`qwen_e8p.py` 支持 Index-Translate-2B/9B/35B、Qwen3.5-4B、27B 和 Qwen3-ASR-1.7B，从官方 BF16 分片直接执行两级拟合和 beam8 搜索。输入目录包含原始 HF 格式文件（可从 ModelScope 下载）以及记录固定 `repo`、40 位 `revision` 的 `source.json`；`--shard FILE` 可逐片编码。已完成块保存校验和，完整分片写入后才清理其临时块；调用方可释放已编码源分片。embedding、共享输出头和视觉大矩阵也量化为 E8P Q4；敏感输出投影和 MTP 矩阵保留 W8，patch、位置表、norm 等较小敏感参数保留原始数据。2B 的 tied 输出头读取同一份压缩 embedding；9B 按官方配置分别编码 embedding 和输出头，保持原始权重共享语义。

`nemotron_embed.py` 复用同一两级编码器，支持 Nemotron-3.5-Embed-1B 的 Ministral3 文本和 Pixtral 视觉权重。attention 输出和最终 projector 保留 W8。其双向注意力、平均池化、图文输入与原生构建方法见[嵌入模型说明](../../docs/embeddings.md)。

```bash
ORINFER_CHECKPOINT_DIR=/path/to/original \
  bash tools/quantization/run.sh qwen_e8p --source-dir /path/to/original \
  --output artifacts/quantization/index-translate-2b-e8p-q4a8
```

极小 E8P 查询按原 FP64 评分的 norm/code 顺序直接返回 top-8，以加速等距候选处理；阈值分支保持原评分结果、权重值、beam 宽度和迭代次数。`check_e8p_search.py` 包含穷举、极小值和换输入 Graph 对照。

`rotation.py` 提供分块与完整 Hadamard 变换；`block_ldlq.py` 提供协方差因子及误差反馈的 CPU 数学参考。整模型校准或微调作为独立流程验收；旋转方案须分别验证激活量化、非线性和舍入边界。

量化质量使用同模型原始 BF16；资源不足时可用明确标注的 FP8。固定 seed 20261002 与 teacher-forced 历史，比较任务结果、目标 NLL、概率和长上下文行为。token 一致率和局部重构误差用于诊断，整模型质量结合概率、任务和长上下文评测判定。

## Qwen Q2 实验

`qwen_e8p --bits 2` 从原始 BF16 使用单级整数 E8P 编码，U16 保存八个权重，每个输出行保留一个 FP16 scale。它与 Q4 使用相同 signed-block128 旋转和精度例外策略；主干、专家、embedding/head 和可量化视觉矩阵使用 Q2，敏感输出投影与 MTP 保留 W8。实际平均位数包含这些例外和全部元数据。

Dense Qwen 可用 `qwen_promote` 将明确选中的主干投影升级为 W8。`--matrices` 指向 JSON 字符串数组，元素为完整原始权重名，例如 `model.language_model.layers.3.self_attn.k_proj.weight`。支持 Q/K/V、GDN QKV/Z 和 FFN gate/up/down；输出投影沿用现有 W8。工具从同一来源 BF16 分块重新量化，保留原 signed-block128 输入基与行 scale，替换选中压缩 bank，其余编码值保持不变。配置中的 `rotated_w8_matrices` 是转换、构建与原生校验共同使用的精度清单；层号、角色、dtype、shape 和单份权重均校验。

```bash
.venv/bin/python -m tools.quantization.qwen_promote \
  --source-dir ORIGINAL_BF16_DIR --upgrade-from E8P_CONVERTED_DIR \
  --matrices selected-projections.json --output PROMOTED_CONVERTED_DIR
```

将输出交给[原生构建器](../model/qwen35/README.md)生成独立执行包。选择清单需要通过该包的任务质量、固定历史概率和完整请求性能验证；局部投影重建误差用于筛选候选。

对于具有独立输出头的 dense 模型，`qwen_head_mix` 可组合 Q2 主干和同一来源的 Q4 输出头。工具校验完整源分片、模型来源、几何、码表与旋转符号，仅替换 `lm_head.weight` 的码字和 scale，其余编码值保持不变。`quantization_config.output_head=e8p_q4` 显式声明输出头精度；embedding 仍使用全局 Q2 bank。原生构建和校验按输出头声明选择 U32/K128 与 Q4 投影，复用相同 E8P 基表。该选项适用于 untied dense E8P 模型。

```bash
.venv/bin/python -m tools.quantization.qwen_head_mix \
  --q2 Q2_CONVERTED_DIR --q4 Q4_CONVERTED_DIR --output HEAD_Q4_CONVERTED_DIR
```

```bash
ORINFER_CHECKPOINT_DIR=/path/to/original \
  bash tools/quantization/run.sh qwen_e8p --source-dir /path/to/original \
  --bits 2 --output artifacts/quantization/qwen-e8p-q2
bash tools/operators/run.sh tools/quantization/check_e8p_q2.py artifacts/check-e8p-q2
```

Q2 输出使用独立目录，构建复用 `tools/model/qwen35/build.py`。生产基线以独立质量和完整请求验收结果为准。

## Qwen MoE 主干 Q4、专家 Q2

`qwen_moe_mix` 从同一原始 checkpoint 的已完成 Q4/Q2 转换中组合权重：普通主干、embedding/head 与视觉使用 Q4 转换，target 路由专家的码字和 scale 使用 Q2 转换，原有 W8/BF16 例外沿用 Q4 转换。工具验证完整源分片 hash、配置、来源版本、码表与旋转符号，流式复制张量字节并原子发布新目录；每个专家只保存一份编码，不读取原始浮点权重或重新量化。

```bash
.venv/bin/python -m tools.quantization.qwen_moe_mix \
  --q4 Q4_CONVERTED_DIR --q2 Q2_CONVERTED_DIR \
  --output artifacts/quantization/qwen-moe-dense-q4-experts-q2/converted
```

配置使用全局 `quant_method=orinfer_e8p_q4_int8` 与 `experts=e8p_q2`。共享 FFN 属于主干，路由专家单独声明精度；单级和两级 E8P 共享相同的基础码表与 selector 数据。原生构建及模型校验分别检查主干 U32 和专家 U16 的 K128 bank。质量恢复、显存和完整请求性能需要使用该新包独立测量。
