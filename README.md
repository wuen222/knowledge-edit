# 知识编辑崩溃可视化诊断实验

这个仓库是 AutoDL A800 实验的轻量控制项目，只保存代码、配置、结果摘要和画图脚本。模型权重、完整 checkpoint、原始大数据集和大规模 hidden-state 文件不应放进 Git。

## 项目目标

第一版实验围绕“连续知识编辑是否会造成模型内部扰动并导致行为崩溃”展开：

- 模型：`meta-llama/Meta-Llama-3-8B-Instruct`
- 数据：优先使用 ZsRE，后续补 CounterFact
- 方法矩阵：`FT-L`、`ROME`、`MEMIT`、`PMET`、`SCR-LITE`
- 连续编辑检查点：`0, 1, 10, 50, 100`
- 证据链：行为指标下降、参数扰动、hidden-state drift、token trajectory、t-SNE/PCA/UMAP 静态图

方法选择的详细理由见 [知识编辑方法分类与实验选择](docs/知识编辑方法分类与实验选择.md)。

## AutoDL 目录约定

在 AutoDL 上统一使用这些路径：

```bash
/root/autodl-tmp/models
/root/autodl-tmp/data
/root/autodl-tmp/runs
/root/autodl-tmp/artifacts
/root/autodl-tmp/cache
/root/autodl-tmp/logs
```

只把结果摘要和图表下载回本地。不要同步模型权重、完整 checkpoint 或大体积 `hidden_states.npz`。

## AutoDL 快速开始

推荐配置：

- GPU：A800 80GB
- 镜像：PyTorch + CUDA + Python 3.10
- 数据盘：至少 200GB，推荐 300GB
- Hugging Face：提前申请 `meta-llama/Meta-Llama-3-8B-Instruct` 访问权限

进入 AutoDL 实例后：

```bash
source /root/.bashrc
nvidia-smi
mkdir -p /root/autodl-tmp/{models,data,runs,artifacts,cache,logs}
echo 'export HF_HOME=/root/autodl-tmp/cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/transformers' >> ~/.bashrc
source ~/.bashrc
```

安装 EasyEdit：

```bash
cd /root
git clone https://github.com/zjunlp/EasyEdit.git
cd /root/EasyEdit
pip install -r requirements.txt
pip install -e .
pip install "transformers>=4.40.0"
```

上传或 clone 本项目：

```bash
cd /root
git clone <你的仓库地址> knowledge-editing-drift
cd /root/knowledge-editing-drift
pip install -r requirements.txt
pip install -e .
```

如果没有 Git 仓库，可以在本地 PowerShell 使用：

```powershell
scp -rP <端口> "E:\New project" root@<AutoDL地址>:/root/knowledge-editing-drift
```

登录 Hugging Face 并验证 Llama 3 权限：

```bash
huggingface-cli login
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
print("Llama 3 权限和缓存路径正常")
PY
```

更详细的逐步说明见 [A800实验运行指南](docs/A800实验运行指南.md)。

## 数据准备

把原始 ZsRE 文件放到任一位置：

```bash
/root/autodl-tmp/data/editing-data/zsre/zsre_mend_eval.json
/root/autodl-tmp/data/zsre_mend_eval.json
```

然后运行：

```bash
cd /root/knowledge-editing-drift
python scripts/prepare_data.py --dataset zsre --n-edits 100 --seed 42
```

输出位置：

```bash
/root/autodl-tmp/data/prepared/zsre/
```

## 冒烟测试

正式跑 100 次连续编辑前，先跑一个极小实验：

```bash
python scripts/run_sequential_edit.py \
  --model /root/autodl-tmp/models/Meta-Llama-3-8B-Instruct \
  --method ROME \
  --hparams /root/autodl-tmp/EasyEdit/hparams/ROME/llama3-8b.yaml \
  --checkpoints 0 1 2 \
  --max-edits 2 \
  --probe-limit 20 \
  --batch-size 1 \
  --hidden-batch-size 1 \
  --easyedit-root /root/autodl-tmp/EasyEdit \
  2>&1 | tee /root/autodl-tmp/logs/smoke_rome.log
```

如果 EasyEdit 自动找不到 hparams，先查：

```bash
find /root/EasyEdit/hparams -maxdepth 2 -iname "*llama*"
```

然后显式传入：

```bash
--hparams /root/EasyEdit/hparams/ROME/<实际配置文件>.yaml
```

## 正式实验

建议在 `tmux` 里跑，避免断开 SSH 后任务中断：

```bash
tmux new -s ke
cd /root/knowledge-editing-drift

python scripts/run_sequential_edit.py --model meta-llama/Meta-Llama-3-8B-Instruct --method FT-L --checkpoints 0 1 10 50 100 --easyedit-root /root/EasyEdit 2>&1 | tee /root/autodl-tmp/logs/ftl.log
python scripts/run_sequential_edit.py --model meta-llama/Meta-Llama-3-8B-Instruct --method ROME --checkpoints 0 1 10 50 100 --easyedit-root /root/EasyEdit 2>&1 | tee /root/autodl-tmp/logs/rome.log
python scripts/run_sequential_edit.py --model meta-llama/Meta-Llama-3-8B-Instruct --method MEMIT --checkpoints 0 1 10 50 100 --easyedit-root /root/EasyEdit 2>&1 | tee /root/autodl-tmp/logs/memit.log
python scripts/run_sequential_edit.py --model meta-llama/Meta-Llama-3-8B-Instruct --method PMET --checkpoints 0 1 10 50 100 --easyedit-root /root/EasyEdit 2>&1 | tee /root/autodl-tmp/logs/pmet.log
python scripts/run_sequential_edit.py --model meta-llama/Meta-Llama-3-8B-Instruct --method SCR-LITE --checkpoints 0 1 10 50 100 2>&1 | tee /root/autodl-tmp/logs/scr_lite.log

python scripts/make_figures.py --runs /root/autodl-tmp/runs --out /root/autodl-tmp/artifacts
```

监控命令：

```bash
watch -n 5 nvidia-smi
tail -f /root/autodl-tmp/logs/rome.log
du -sh /root/autodl-tmp/*
```

## 下载结果

Windows 本地可用：

```powershell
scp -rP <端口> root@<AutoDL地址>:/root/autodl-tmp/artifacts "E:\New project\artifacts"
scp -rP <端口> root@<AutoDL地址>:/root/autodl-tmp/runs "E:\New project\runs"
```

如果结果目录过大，只下载轻量文件：

```bash
rsync -av --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.png' --include='*.pdf' --exclude='*' \
  root@<AutoDL地址>:/root/autodl-tmp/runs/ ./runs/
rsync -av root@<AutoDL地址>:/root/autodl-tmp/artifacts/ ./artifacts/
```

## 脚本说明

- `scripts/prepare_data.py`：把 ZsRE 或 CounterFact 规范化为 edit stream 和固定 probe set。
- `scripts/run_sequential_edit.py`：运行连续编辑、检查点评估、hidden-state drift、参数扰动和 token trajectory。
- `scripts/evaluate_generation.py`：独立的自回归生成评估脚本。
- `scripts/extract_internal_signals.py`：独立的 hidden-state 抽取和比较脚本。
- `scripts/make_figures.py`：生成静态图表和相关性汇总。

## 常见问题

- Llama 3 下载失败：先确认 Hugging Face 账号已同意模型协议，并在 AutoDL 上执行过 `huggingface-cli login`。
- EasyEdit 找不到配置：使用 `find /root/EasyEdit/hparams -maxdepth 2 -iname "*llama*"` 查实际 hparams 文件，再传 `--hparams`。
- 显存不足：冒烟测试先用 `--batch-size 1 --hidden-batch-size 1 --probe-limit 20`；FT-L 最吃显存，建议最后跑。
- 本地没有 GPU：本地只做代码、文档、结果整理和画图检查；真实编辑实验放在 AutoDL A800 上跑。
- t-SNE 不稳定：不要单独用 t-SNE 下结论，必须结合原始高维 cosine drift 或 CKA。
