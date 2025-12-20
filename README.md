# Multi-Target-Speaker-Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
![Vibe Coding](https://img.shields.io/badge/Vibe%20Coding-AI%20Assisted-blueviolet)

A turnkey solution for **lossless batch extracting** multiple specific speakers from mixed audio. Powered by TitanNet & Silero VAD.

基于参考音频的**多目标说话人**自动提取与无损切分工具。支持批量处理、GPU加速与高保真导出。

---

## Features

- **Multi-Target Extraction**: Extract multiple speakers from mixed audio simultaneously
- **Batch Processing**: Process entire audio datasets in one run
- **GPU Acceleration**: Batch inference with CUDA support
- **High Accuracy**: NVIDIA TitanNet-Large for speaker embeddings
- **Fast VAD**: Silero VAD with ONNX acceleration
- **Lossless Export**: Preserves original sample rate and audio channels

---

## Quick Start | 快速开始

### 1. Installation | 安装

```bash
# Clone the repository | 克隆仓库
git clone https://github.com/YOUR_USERNAME/Multi-Target-Speaker-Extraction.git
cd Multi-Target-Speaker-Extraction

# Install dependencies | 安装依赖
pip install -r requirements.txt
```

> **Note**: For GPU support, install PyTorch with CUDA first:
> https://pytorch.org/get-started/locally/
>
> **注意**：如需GPU加速，请先安装CUDA版本的PyTorch

### 2. Prepare Reference Audio | 准备参考音频

```
enrollment_audio/
├── SpeakerA/
│   ├── sample1.wav
│   └── sample2.wav
└── SpeakerB/
    └── sample1.wav
```

- Create a folder for each speaker
- Add clean audio samples (`.wav` format recommended)
- More samples = better accuracy (3-10 samples recommended)

---

- 每个说话人创建一个文件夹
- 添加纯净的语音样本（推荐 `.wav` 格式）
- 更多样本 = 更高准确率（推荐3-10个样本）

### 3. Add Input Audio | 添加待处理音频

Place audio files to process in `input_audio/`:

```
input_audio/
├── audio1.wav
└── audio2.wav
```

### 4. Run | 运行

**Windows:**
```bash
run_windows.bat
# Or | 或者
python run.py
```

**Linux / macOS:** (Without testing)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python run.py
```

### 5. View Results | 查看结果

```
output/
├── SpeakerA/
│   └── segments/
│       └── 0.85_audio1_seg_0_1.23_4.56.wav
├── SpeakerB/
│   └── segments/
├── metadata/
│   └── audio1.json
└── summary.json
```

- Filename format: `{similarity}_{source}_{segment_info}.wav`
- Sort by filename (descending) to view by similarity score
- 文件名格式：`{相似度}_{源文件}_{片段信息}.wav`
- 按文件名降序排列可按相似度从高到低查看

---

## Configuration | 配置

Edit `config.yaml` to customize:

```yaml
verification:
  similarity_threshold: 0.70  # 0.65-0.80 recommended | 推荐范围

performance:
  batch_size: 32              # Increase for faster processing | 增大可提速
  prefetch_workers: 2         # Audio prefetch threads | 预加载线程数

speaker_management:
  skip_speakers: []           # Speakers to skip | 要跳过的说话人
  include_only: []            # Process only these | 仅处理这些说话人
```

### Key Parameters | 关键参数

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.70 | Cosine similarity threshold (0.65-0.80) |
| `min_duration` | 0.5s | Minimum segment duration |
| `merge_gap` | 0.3s | Gap for merging adjacent segments |
| `batch_size` | 32 | Batch size for GPU inference |
| `prefetch_workers` | 2 | Audio prefetch thread count |

---

## Technical Details | 技术实现

### Models Used

| Model | Purpose | Device |
|-------|---------|--------|
| NVIDIA TitanNet-Large | Speaker embedding extraction | GPU |
| Silero VAD (ONNX) | Voice activity detection | CPU |

### Processing Pipeline

```
1. Extract reference embeddings → Compute average speaker vectors (L2 normalized)
2. VAD detection → Locate speech segments in input audio
3. Speaker identification → Extract segment embeddings, compute cosine similarity
4. Filtering → Keep segments above threshold
5. Merging → Merge adjacent segments from same speaker
6. Export → Save segments with original quality
```

---

## Project Structure | 项目结构

```
Multi-Target-Speaker-Extraction/
├── run.py                    # Entry point | 启动入口
├── speaker_verification.py   # Core logic | 核心逻辑
├── speaker_state_manager.py  # Speaker filtering | 说话人过滤
├── config.yaml               # Configuration | 配置文件
├── requirements.txt          # Dependencies | 依赖列表
├── run_windows.bat           # Windows launcher | Windows启动脚本
├── LICENSE                   # MIT License
├── enrollment_audio/         # Reference audio | 参考音频
├── input_audio/              # Input files | 待处理音频
└── output/                   # Results | 处理结果
```

---

## Performance Tips | 性能优化建议

1. **GPU Memory**: Adjust `batch_size` based on your VRAM
   - 8GB VRAM: `batch_size: 32`
   - 16GB VRAM: `batch_size: 64-96`

2. **Speed**: Enable prefetching with `prefetch_workers: 2-4`

3. **Accuracy**: Use more reference samples per speaker

---

## Troubleshooting | 常见问题

**Q: CUDA out of memory**
A: Reduce `batch_size` in config.yaml

**Q: No speakers detected**
A: Lower `similarity_threshold` (try 0.65)

**Q: Poor accuracy**
A: Add more clean reference samples

---

## License | 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project uses components under the following licenses:

- **NVIDIA NeMo**: Apache License 2.0
- **Silero VAD**: MIT License
- **PyTorch**: BSD-style License

---

## Acknowledgments | 致谢

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for TitanNet speaker verification model
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection
- Sample audio files are from the LibriSpeech corpus (CC BY 4.0).

---

## 中文说明

### 简介

**多目标说话人提取工具 (MTSE)** - 从混合音频中批量识别并提取多个指定说话人的语音片段。

### 功能特点

- 支持多说话人同时处理
- GPU批量推理加速
- 高精度NVIDIA TitanNet-Large模型
- Silero VAD + ONNX快速语音检测
- 保持原始音频质量（采样率、声道）
- 跨平台支持（Windows/Linux/macOS）

### 使用流程

1. **准备参考音频**：在 `enrollment_audio/` 下为每个说话人创建文件夹，放入纯净语音样本
2. **放置待处理音频**：将待处理的音频文件放入 `input_audio/`
3. **运行**：双击 `run_windows.bat` 或运行 `python run.py`
4. **查看结果**：输出在 `output/` 目录，按相似度排序的片段

### 参数调优

- `similarity_threshold`: 相似度阈值，推荐0.65-0.80
- `batch_size`: 批量大小，显存越大可设越高
- `prefetch_workers`: 预加载线程，推荐2-4

### 常见问题

- **显存不足**：降低 `batch_size`
- **漏检**：降低 `similarity_threshold`
- **误检**：增加参考样本数量或提高阈值
