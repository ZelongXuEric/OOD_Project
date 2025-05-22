# LLM Hidden State Analysis for Sequence Pattern Tasks

This project explores the internal representations of Large Language Models (LLMs), specifically Llama 3 8B Instruct, when processing sequence pattern tasks such as word reversal. It aims to understand how the model encodes and transforms information агрегацияhidden states from different layers.

## Project Goal

The primary goal is to analyze the hidden states extracted from various layers of the Llama 3 8B Instruct model while it performs a word reversal task. We use techniques like Principal Component Analysis (PCA), linear probing, and cosine similarity to investigate:

*   How word identity and positional information are represented across layers.
*   How the model distinguishes between input prompts and generated outputs.
*   The transformation of representations as the model processes the sequence and generates the reversed output.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/YourRepositoryName.git
    cd YourRepositoryName
    ```

2.  **Create a Python environment and install dependencies:**
    It's recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    # Using conda
    conda create -n ood_llama3 python=3.10  # Or your preferred Python version
    conda activate ood_llama3
    
    pip install -r requirements.txt
    ```
    Ensure you have PyTorch installed compatible with your CUDA version if using a GPU.

3.  **Configuration:**
    *   Modify `config.py` to set parameters such as `MODEL_NAME`, `PATTERN_TYPE`, `SEQUENCE_LENGTH`, `NUM_SHOTS`, `LAYER_INDICES`, etc.
    *   The default configuration uses `meta-llama/Meta-Llama-3-8B-Instruct` for a word reversal task.
    *   Ensure you have access to the Llama 3 model (e.g., by logging into Hugging Face CLI: `huggingface-cli login`).

## Running the Analysis

Execute the main script:

```bash
python run.py