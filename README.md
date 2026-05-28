# 中文 RAG 位置重排小规模实验

本项目参考论文 *Do RAG Systems Really Suffer From Positional Bias?* 的真实 RAG 场景实验，基于中文 DuReader 2.0 构建小规模语料库，比较不同 passage 排序策略对回答准确率的影响。

当前版本使用：

- 数据集：DuReader 2.0
- 检索器：`jieba + BM25`
- 生成模型：本地 Hugging Face 格式的 `Qwen2.5-7B`
- 裁判模型：DeepSeek API
- 实验重点：论文第 5 节真实 RAG 场景下的 passage ordering strategies

> 注意：本项目是中文小规模复现实验，不是原论文的大规模严格复现。原论文使用英文 QA benchmark、完整 KILT Wikipedia、大规模检索库、BM25/BGE/reranker 等多条检索管线。本项目第一版只实现中文小规模 BM25 管线。

## 1. 环境准备

推荐使用 conda：

```bash
conda create -n nlp python=3.11 -y
conda activate nlp

pip install -r requirements.txt
pip install -e .
```

安装完成后确认 CLI 可用：

```bash
rag-zh --help
```

## 2. 下载 DuReader 2.0

DuReader 2.0 官方仓库位于 Baidu GitHub：<https://github.com/baidu/DuReader>。

在项目目录下执行：

```bash
git clone https://github.com/baidu/DuReader.git
cd DuReader/DuReader-2.0/data
bash download.sh
```

官方脚本会下载并解压 DuReader 2.0 的 raw 和 preprocessed 数据。下载完成后，通常可以使用：

```bash
DuReader/DuReader-2.0/data/preprocessed
```

作为本项目的 `--dureader-path`。

## 3. 下载 Qwen2.5-7B

本项目通过 `transformers` 从本地 checkpoint 加载生成模型。推荐模型为：

```text
Qwen/Qwen2.5-7B
```

官方 Hugging Face 页面：<https://huggingface.co/Qwen/Qwen2.5-7B>。

使用 Hugging Face 下载：

```bash
conda activate nlp

huggingface-cli download Qwen/Qwen2.5-7B \
  --local-dir models/Qwen2.5-7B \
  --local-dir-use-symlinks False
```

然后设置环境变量：

```bash
export GENERATOR_MODEL_PATH="models/Qwen2.5-7B"
```

### 显存提示

`Qwen2.5-7B` 是 7B 级别模型。完整精度或半精度推理需要较多显存/统一内存。若本机资源不足，可以：

- 换用更小的 Qwen2.5 模型做流程验证。
- 使用量化版本，但需要相应修改加载方式。
- 使用远程 GPU 机器运行 `rag-zh run-experiment`。

当前代码按 base 模型的 plain prompt completion 方式调用 `Qwen2.5-7B`，不使用 instruct chat template。如果你改用 `Qwen2.5-7B-Instruct`，需要另外调整 prompt 模式。

## 4. 配置 DeepSeek API

本项目用 DeepSeek API 做 LLM-as-a-judge，判断生成答案是否与参考答案语义一致。

设置 API key：

```bash
export DEEPSEEK_API_KEY="sk-..."
```

默认配置见 `configs/default.yaml`：

```yaml
judge:
  base_url: https://api.deepseek.com
  model: deepseek-v4-flash
```

如需更换模型，可以修改配置文件，或在运行前设置自己的配置。

## 5. 参数说明

默认参数位于 `configs/default.yaml`。命令行参数优先于配置文件；如果命令行没有传入对应参数，则读取配置文件默认值。

### 配置文件参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `data.dureader_path` | `data/raw` | DuReader 数据目录或 JSON/JSONL 文件路径。可被 `prepare-data --dureader-path` 覆盖。 |
| `data.prepared_path` | `data/processed/prepared.json` | 抽样后实验数据的保存路径。可被 `prepare-data --output` 覆盖。 |
| `data.sample_size` | `50` | 抽取的问题数量，也就是评测集大小。可被 `prepare-data --sample-size` 覆盖。 |
| `data.corpus_size` | `1000` | 构建检索库的 passage 数量。可被 `prepare-data --corpus-size` 覆盖。 |
| `data.seed` | `42` | 数据抽样和 `shuffle` 排序策略使用的随机种子。 |
| `data.min_answer_overlap` | `0.0` | 可选噪声过滤阈值。默认不清洗；大于 0 时过滤参考答案 token 与正例 passage 重合度过低的样本。 |
| `retrieval.top_k` | `5` | 每个问题检索的 passage 数量。可被 `retrieve --top-k` 覆盖。 |
| `generator.model_path` | `${GENERATOR_MODEL_PATH}` | 本地 Hugging Face 模型目录。通常通过环境变量 `GENERATOR_MODEL_PATH` 设置。 |
| `generator.max_new_tokens` | `128` | 生成答案的最大 token 数。过短可能截断答案，过长可能增加重复生成。 |
| `generator.temperature` | `0.0` | 生成采样温度。`0.0` 表示确定性生成。 |
| `generator.device` | `auto` | 模型加载设备。`auto` 会交给 `transformers` 自动分配。 |
| `generator.repetition_penalty` | `1.08` | 重复惩罚系数，用于降低重复输出。 |
| `generator.no_repeat_ngram_size` | `6` | 禁止重复生成指定长度的 n-gram，用于缓解“问题/答案”格式循环。 |
| `judge.base_url` | 见配置文件 | DeepSeek 兼容 API 的 base URL。 |
| `judge.model` | 见配置文件 | 裁判评估使用的模型名。 |
| `judge.api_key` | 见配置文件 | 裁判 API key 配置。 |
| `output.dir` | `results` | 实验输出目录。 |

### CLI 参数

| 命令 | 参数 | 默认值 | 含义 |
|---|---|---|---|
| 全局 | `--config` | `configs/default.yaml` | 指定配置文件路径。 |
| `prepare-data` | `--dureader-path` | `data.dureader_path` | 指定 DuReader 数据路径。 |
| `prepare-data` | `--output` | `data.prepared_path` | 指定 prepared JSON 输出路径。 |
| `prepare-data` | `--sample-size` | `data.sample_size` | 指定抽样问题数。 |
| `prepare-data` | `--corpus-size` | `data.corpus_size` | 指定检索库 passage 数。 |
| `prepare-data` | `--min-answer-overlap` | `data.min_answer_overlap` | 指定 DuReader 噪声过滤阈值。 |
| `retrieve` | `--top-k` | `retrieval.top_k` | 指定展示每个问题的检索条数。 |
| `retrieve` | `--limit` | `3` | 指定展示多少个问题的检索结果。 |
| `run-experiment` | `--config` | `configs/default.yaml` | 使用指定配置运行完整实验。 |

## 6. 准备实验数据

默认小规模配置：

- `sample_size=50`：抽取 50 个问题作为评测集。
- `corpus_size=1000`：构建约 1000 条 passage 的检索库。
- `top_k=5`：每个问题检索 5 条 passage。

运行：

```bash
conda activate nlp

rag-zh prepare-data \
  --dureader-path DuReader/DuReader-2.0/data/preprocessed \
  --sample-size 50 \
  --corpus-size 1000
```

如果样本噪声较多，可以启用轻量过滤：

```bash
rag-zh prepare-data \
  --dureader-path DuReader/DuReader-2.0/data/preprocessed \
  --sample-size 50 \
  --corpus-size 1000 \
  --min-answer-overlap 0.3
```

输出文件：

```text
data/processed/prepared.json
```

`sample_size` 和 `corpus_size` 是两个不同概念：`sample_size` 控制问题数量，`corpus_size` 控制检索库 passage 数量。因此 `sample_size=50` 不代表只有 50 条 passage。

## 7. 检查检索结果

先用少量样本检查 BM25 是否能检索到合理 passage：

```bash
rag-zh retrieve --limit 3
```

如果输出的前几条 passage 与问题明显无关，优先检查：

- `--dureader-path` 是否指向正确数据目录。
- DuReader 数据是否已完整解压。
- `data/processed/prepared.json` 是否是用当前数据重新生成的。

## 8. 运行完整实验

确认以下环境变量已设置：

```bash
export GENERATOR_MODEL_PATH="models/Qwen2.5-7B"
export DEEPSEEK_API_KEY="sk-..."
```

运行：

```bash
rag-zh run-experiment
```

输出位于：

```text
results/
```

包括：

- `results/details.jsonl`：逐问题、逐排序策略的明细结果。
- `results/summary.csv`：各排序策略准确率汇总。
- `results/summary.md`：Markdown 表格版汇总。

`details.jsonl` 的 `retrieved` 字段会记录每条检索 passage 的 `rank`、`score`、`passage_id`、`title` 和正文前 800 字 `text`，用于定位检索证据是否足够。

## 9. 调试流程

建议先按小样本调通流程，再扩大到默认规模。

### 9.1 数据调试

先生成 3 个问题和 50 条 passage：

```bash
rag-zh prepare-data \
  --dureader-path DuReader/DuReader-2.0/data/preprocessed \
  --sample-size 3 \
  --corpus-size 50
```

检查 `data/processed/prepared.json` 中是否包含：

- `examples`：问题样本列表，数量应接近 `sample_size`。
- `passages`：检索库 passage 列表，数量应接近 `corpus_size`。

如果数量明显不对，优先检查 `--dureader-path` 是否指向 DuReader 2.0 的 `preprocessed` 或可解析的 JSON/JSONL 数据。

### 9.2 检索调试

查看少量问题的 BM25 检索结果：

```bash
rag-zh retrieve --limit 3
```

重点看 top-k 的 `title` 是否和问题相关。如果标题明显无关，通常说明数据路径、抽样语料或中文分词检索效果需要检查。

### 9.3 生成与裁判调试

确认模型和裁判环境变量已设置后，先在小样本上运行：

```bash
rag-zh run-experiment
```

查看 `results/details.jsonl`：

- `prediction`：Qwen 生成的答案。
- `correct`：裁判判断是否正确。
- `judge_rationale`：裁判理由。
- `retrieved`：该策略下使用的检索 passage 排序，其中 `text` 可用于检查正文是否真的包含答案证据。

如果 `prediction` 频繁重复、过长或包含多个“问题/答案”片段，可以优先降低 `generator.max_new_tokens`，例如改为 `64`。

### 9.4 结果分析

先看总体表格：

```bash
cat results/summary.md
```

再看失败样本：

```bash
grep '\"correct\": false' results/details.jsonl | head
```

如果五种排序策略差异很小，这是符合原论文第 5 节的核心观察之一：真实 RAG 场景中，相关 passage 和干扰 passage 往往同时出现在靠前位置，单纯调整 passage 顺序未必显著提升准确率。

## 10. 排序策略

本项目实现 5 种 passage 排序策略：

- `sequential`：保持 BM25 检索排序。
- `inverse`：反转 BM25 检索排序。
- `shuffle`：固定随机种子乱序。
- `max_relevance`：参考论文中 Qwen2.5-7B 的 k=5 位置偏好 `[5, 1, 4, 3, 2]`，把高排名 passage 放到模型更偏好的位置。
- `min_distraction`：参考论文中 Qwen2.5-7B 的 k=5 干扰规避顺序 `[3, 2, 4, 1, 5]`，尽量避免把潜在干扰 passage 放到高影响位置。

## 11. 运行测试

测试不依赖真实 DuReader、Qwen 模型或 DeepSeek API：

```bash
conda activate nlp
pytest
```

当前测试覆盖：

- DuReader 风格 JSON/JSONL 解析
- passage 抽样
- BM25 检索
- 五种排序策略
- DeepSeek 裁判响应解析

## 12. 常见问题

### `Prepared dataset not found`

先运行：

```bash
rag-zh prepare-data --dureader-path /path/to/dureader
```

### `generator.model_path is required`

设置：

```bash
export GENERATOR_MODEL_PATH=/path/to/models/Qwen2.5-7B
```

### `judge.api_key is required`

设置：

```bash
export DEEPSEEK_API_KEY=sk-...
```

### 显存或内存不足

先用更小样本验证流程：

```bash
rag-zh prepare-data \
  --dureader-path data/raw/DuReader/DuReader-2.0/data/preprocessed \
  --sample-size 3 \
  --corpus-size 50
```

也可以临时换成更小的 Qwen2.5 checkpoint 测试完整流程。

### 为什么会出现“无法确定”

当前 prompt 要求模型只根据检索文档回答。如果模型判断文档证据不足，就可能输出“无法确定”。常见原因包括：

- BM25 检索到的是主题相关但不能直接回答问题的 passage。
- DuReader 样本中存在问题和参考答案不完全匹配的噪声，可尝试用 `--min-answer-overlap 0.3` 重新准备数据。
- base 模型对指令遵循能力弱于 instruct 模型，可能更容易保守拒答或重复生成。

排查时先看 `results/details.jsonl` 中对应样本的 `prediction`、`judge_rationale`、`retrieved` 标题和 `retrieved[*].text` 正文片段。

### 为什么小规模结果不能直接等同于原论文结论

本项目默认只用 50 个问题和 1000 条 passage，且第一版只实现 BM25。原论文使用更大的英文 benchmark、完整 Wikipedia 级检索库、多种检索器和 reranker。因此本项目适合做中文小规模验证和课程实验，不能直接声称严格复现原论文全部结论。

### 为什么 Qwen2.5-7B 不使用 instruct chat template

当前使用的是 `Qwen2.5-7B` base 模型，代码采用普通 prompt completion。Chat template 主要适用于 instruct/chat 模型。如果换成 `Qwen2.5-7B-Instruct`，再考虑引入 chat template 更合理。

## 13. 项目结构

```text
.
├── configs/default.yaml
├── data/
├── results/
├── src/rag_zh/
│   ├── cli.py
│   ├── data.py
│   ├── experiment.py
│   ├── generation.py
│   ├── judge.py
│   ├── reorder.py
│   └── retrieval.py
├── tests/
├── requirements.txt
└── pyproject.toml
```
