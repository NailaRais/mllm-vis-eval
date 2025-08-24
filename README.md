# MLLM Visual Evaluator (MLLM-VisEval)

An open-source tool to evaluate Multimodal LLMs (MLLMs) on the [Do-You-See-Me](https://huggingface.co/datasets/microsoft/Do-You-See-Me) benchmark for visual perception.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/mllm-vis-eval/blob/main/notebooks/demo.ipynb)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Evaluate any Hugging Face Vision-Language model on a standardized visual perception test
- Provides overall accuracy score and detailed per-question breakdown
- Category-wise performance analysis (Figure-Ground, Spatial Reasoning, etc.)
- Export results to CSV and JSON for further analysis
- Easy to use command-line interface

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mllm-vis-eval.git
   cd mllm-vis-eval
   ```
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Evaluate a model on 50 samples from the dataset:
```bash
python evaluate.py --model_name llava-hf/llava-1.5-7b-hf --num_samples 50
```
Evaluate on specific categories only:
```bash
python evaluate.py --model_name llava-hf/llava-1.5-7b-hf --categories visual_figure_ground visual_spatial
```
Run a comprehensive evaluation on all samples:
```bash
python evaluate.py --model_name llava-hf/llava-1.5-7b-hf --num_samples -1
```

## Supported Models

This tool should work with any Hugging Face model that supports visual question answering, including:

- LLaVA
- InstructBLIP
- Fuyu
- GPT-4V (via API)
- Florence-2

## Results Format

The tool generates two output files:

- `detailed_[model_name].csv` - Contains each question, the model's response, and correctness
- `summary_[model_name].json` - Contains overall accuracy and category-wise breakdown

Example summary:
```json
{
  "model": "llava-hf/llava-1.5-7b-hf",
  "overall_accuracy": 62.5,
  "total_samples": 80,
  "correct_predictions": 50,
  "category_stats": {
    "visual_figure_ground": {
      "accuracy": 65.2,
      "total": 23,
      "correct": 15
    },
    "visual_spatial": {
      "accuracy": 58.8,
      "total": 17,
      "correct": 10
    }
  }
}
```

## Advanced Usage

Evaluating with Different Precision:
```bash
# Use float32 precision (more memory, potentially more accurate)
python evaluate.py --model_name llava-hf/llava-1.5-7b-hf --precision float32
```
Specifying Output Directory:
```bash
python evaluate.py --model_name llava-hf/llava-1.5-7b-hf --output_dir my_results
```
Forcing CPU Usage:
```bash
python evaluate.py --model_name llava-hf/llava-1.5-7b-hf --device cpu
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add some amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Citation
If you use this tool in your research, please cite the original Do-You-See-Me dataset:

```bibtex
@misc{kanade2025multidimensionalbenchmarkevaluating,
      title={Do You See Me : A Multidimensional Benchmark for Evaluating Visual Perception in Multimodal LLMs}, 
      author={Aditya Kanade and Tanuja Ganu},
      year={2025},
      eprint={2506.02022},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.02022}, 
}
```

# Acknowledgments
Thanks to Microsoft for creating the Do-You-See-Me dataset.

- Built with Hugging Face Transformers
- Inspired by the need for better evaluation of multimodal AI systems.
