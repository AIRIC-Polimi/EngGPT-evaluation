# EngGPT2-16B-A3B Evaluation Framework

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2605.07731)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](engineering-group/EngGPT2-16B-A3B)

Official evaluation repository for the paper *"Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs"*. 

This repository contains the lightweight, portable framework used to evaluate **EngGPT2-16B-A3B** and comparing models across various benchmarks. It builds on top of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [vLLM](https://github.com/vllm-project/vllm) for accelerated inference, featuring custom tasks and post-processing scripts.

## ⚠️ Hardware Disclaimer

> **Note on Hardware:** The code and evaluation configurations in this repository were run and validated exclusively on an **NVIDIA A100 GPU (40GB memory)** with **CUDA Version 13.1**. 
> 
> The codebase has **not** been tested for CPU usage or on GPUs with different memory constraints or architectures. You may need to adapt parameters like `batch_size` or max sequence lengths depending on your hardware limits.

## 🚀 Setup & Installation

This project is entirely managed using [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager. Dependencies are declared inline via [PEP 723](https://peps.python.org/pep-0723/) directly inside the scripts. 

You **do not** need to manually create virtual environments or install `requirements.txt`. `uv` will automatically provision isolated environments on the fly.

### 1. Install `uv`
On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(For Windows or other installation methods, refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)).*

### 2. Clone the Repository
```bash
git clone https://github.com/AIRIC-Polimi/EngGPT-evaluation.git
cd EngGPT-evaluation
```

## 🏃‍♂️ Running Evaluations

To run any script, simply use `uv run`. 

Because utility functions are shared across the repository in the `shared_utils` directory, execute the scripts from the repository root and pass `PYTHONPATH=.` so Python can locate the common utilities:

```bash
# Run the main vLLM parametric evaluation script
PYTHONPATH=. uv run vllm_model_evaluation/scripts/run_evaluations_parametric.py --models engineering-group/EngGPT2-16B-A3B --tasks arc_challenge_chat
```

*Note: The first time you run a script, `uv` will automatically download the correct Python version and all required libraries (e.g., `torch`, `vllm`, `lm-eval`) into a cached, isolated environment.*

### Custom Configurations

Evaluation configurations (like formatting or generation kwargs) are defined using YAML files. The scripts will automatically look for custom configurations in the `model_configs` folder matching the model's name.

For instance, `vllm_model_evaluation/model_configs/EngGPT2-16B-A3B/template_config_vllm.yaml` is used automatically when `EngGPT2-16B-A3B` is evaluated.

## 📁 Repository Structure

```text
├── lm_eval_custom_polimi/         # Custom tasks configurations for lm-eval
├── utils/                         # Shared python utilities across the repository
├── evalscope_evaluation/          # Scripts for EvalScope pipeline
├── italic_evaluation/             # Scripts and configs for ITALIC benchmark (in its original setting, from https://github.com/Crisp-Unimib/ITALIC)
├── hf_model_evaluation/           # Standard HF Transformers evaluations
└── vllm_model_evaluation/         # vLLM-accelerated evaluation pipelines
    ├── model_configs/             # Model-specific YAML configurations
    └── scripts/                   # Scripts for lm-eval pipeline
```

## 📝 Citation

If you use this evaluation framework, the custom benchmark implementations, or our models in your research, please cite our work:

```bibtex
@misc{sassella2026benchmarkingenggpt216ba3bcomparableitalian,
      title={Benchmarking EngGPT2-16B-A3B against Comparable Italian and International Open-source LLMs}, 
      author={Andrea Sassella and Andrea Chizzola and Tommaso Bianchi and Luca Alessandrelli and Mark James Carman},
      year={2026},
      eprint={2605.07731},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.07731}, 
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
