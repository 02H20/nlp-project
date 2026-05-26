# 中文 RAG 位置重排小规模实验

本项目参考论文 *Do RAG Systems Really Suffer From Positional Bias?* 的真实 RAG 场景实验，基于中文 DuReader 2.0 构建小规模语料库，比较不同 passage 排序策略对回答准确率的影响。

当前版本使用：

- 数据集：DuReader 2.0
- 检索器：`jieba + BM25`
- 生成模型：本地 Hugging Face 格式的 `Qwen2.5-7B-Instruct`
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
git clone https://github.com/baidu/DuReader.git data/raw/DuReader
cd data/raw/DuReader/DuReader-2.0/data
bash download.sh
```

官方脚本会下载并解压 DuReader 2.0 的 raw 和 preprocessed 数据。下载完成后，通常可以使用：

```bash
data/raw/DuReader/DuReader-2.0/data/preprocessed
```

作为本项目的 `--dureader-path`。

## 3. 下载 Qwen2.5-7B-Instruct

本项目通过 `transformers` 从本地 checkpoint 加载生成模型。推荐模型为：

```text
Qwen/Qwen2.5-7B-Instruct
```

官方 Hugging Face 页面：<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>。

使用 Hugging Face 下载：

```bash
conda activate nlp

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct \
  --local-dir-use-symlinks False
```

然后设置环境变量：

```bash
export GENERATOR_MODEL_PATH="models/Qwen2.5-7B-Instruct"
```

### 显存提示

`Qwen2.5-7B-Instruct` 是 7B 级别模型。完整精度或半精度推理需要较多显存/统一内存。若本机资源不足，可以：

- 换用更小的 Qwen2.5 Instruct 模型做流程验证。
- 使用量化版本，但需要相应修改加载方式。
- 使用远程 GPU 机器运行 `rag-zh run-experiment`。

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

## 5. 准备实验数据

默认小规模配置：

- `sample_size=50`：抽取 50 个问题作为评测集。
- `corpus_size=1000`：构建约 1000 条 passage 的检索库。
- `top_k=5`：每个问题检索 5 条 passage。

运行：

```bash
conda activate nlp

rag-zh prepare-data \
  --dureader-path data/raw/DuReader/DuReader-2.0/data/preprocessed \
  --sample-size 50 \
  --corpus-size 1000
```

输出文件：

```text
data/processed/prepared.json
```

## 6. 检查检索结果

先用少量样本检查 BM25 是否能检索到合理 passage：

```bash
rag-zh retrieve --limit 3
```

如果输出的前几条 passage 与问题明显无关，优先检查：

- `--dureader-path` 是否指向正确数据目录。
- DuReader 数据是否已完整解压。
- `data/processed/prepared.json` 是否是用当前数据重新生成的。

## 7. 运行完整实验

确认以下环境变量已设置：

```bash
export GENERATOR_MODEL_PATH="models/Qwen2.5-7B-Instruct"
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

## 8. 排序策略

本项目实现 5 种 passage 排序策略：

- `sequential`：保持 BM25 检索排序。
- `inverse`：反转 BM25 检索排序。
- `shuffle`：固定随机种子乱序。
- `max_relevance`：参考论文中 Qwen2.5-7B 的 k=5 位置偏好 `[5, 1, 4, 3, 2]`，把高排名 passage 放到模型更偏好的位置。
- `min_distraction`：参考论文中 Qwen2.5-7B 的 k=5 干扰规避顺序 `[3, 2, 4, 1, 5]`，尽量避免把潜在干扰 passage 放到高影响位置。

## 9. 运行测试

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

## 10. 常见问题

### `Prepared dataset not found`

先运行：

```bash
rag-zh prepare-data --dureader-path /path/to/dureader
```

### `generator.model_path is required`

设置：

```bash
export GENERATOR_MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct
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

也可以临时换成更小的 Qwen2.5 Instruct checkpoint 测试完整流程。

## 11. 项目结构

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
