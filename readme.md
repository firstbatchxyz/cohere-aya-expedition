# Cohere Aya Expedition 
## Curriculum-Guided Decentralized Synthethic Data Generation with History-Aware Classifier-Free Guidance

This project explores a method for diversifying instruction datasets for LLMs. The core idea is to iteratively generate augmented instructions, identify dense (semantically similar) regions in the generated data using embeddings, and then use these dense regions as negative prompts for Classifier-Free Guidance (CFG) to encourage the generation of novel, diverse instructions.

## Directory Structure
└── cohere-aya-expedition/
├── controller.py # Manages embeddings, vector DB, and density selection
├── dria.py # Orchestrates the DRIA process (bootstrap & guided generation)
├── generate_dataset.py # Script to generate datasets using DRIA and a baseline
├── inst2gen.py # Converts augmented instructions into a question-answer dataset format
├── lm_eval.py # Wrapper for lm-evaluation-harness
├── metrics.py # Computes diversity metrics for generated datasets
├── node.py # LLMNode class for text generation (standard & CFG)
├── prompts.py # Stores various prompt templates
├── requirements.txt # Python dependencies
├── run_model.py # Handles model loading and fine-tuning (SFT)
└── test_pipe.py # Example pipeline for data generation, fine-tuning, and evaluation

## Core Components

*   **`node.py` (LLMNode):** Encapsulates an LLM, providing methods for standard text generation and generation with Classifier-Free Guidance (CFG) using positive and negative prompts.
*   **`controller.py` (Controller, E5Embedder, VectorDB):**
    *   `E5Embedder`: Generates text embeddings.
    *   `VectorDB`: Stores and searches text embeddings.
    *   `Controller`: Uses the `VectorDB` to `select()` texts from dense regions in the embedding space. These selected texts are used as negative prompts.
*   **`dria.py` (Dria):** The main orchestrator.
    1.  Initializes multiple `LLMNode` instances.
    2.  **Bootstrap Phase:** Generates initial augmented instructions from a base instruction.
    3.  **Guided Generation Phase (Iterative):**
        *   Uses the `Controller` to `select()` dense instructions.
        *   Each `LLMNode` generates a new augmentation using its previous output (or base instruction) as a positive prompt and the selected dense instructions as negative prompts (CFG).
        *   If the new augmentation is still too similar to existing ones, the guidance scale is increased, and regeneration is attempted.
*   **`generate_dataset.py`:** Uses `Dria` to create instruction datasets (`dria` and `bl` - baseline) and computes diversity metrics.
*   **`inst2gen.py`:** Processes the augmented instructions from `generate_dataset.py` and uses an `LLMNode` to convert them into a final question-answer pair dataset format.
*   **`metrics.py`:** Calculates semantic diversity, MST diversity, and convex hull area from embeddings.
*   **`run_model.py` & `test_pipe.py`:** Support fine-tuning a model on the generated datasets and evaluating it using `lm-evaluation-harness`.

## Workflow

1.  **Dataset Generation (`generate_dataset.py`):**
    *   The `Dria` class is used to generate a set of augmented instructions using the iterative guided process.
    *   A baseline dataset is also generated (likely without the guided diversification).
    *   Diversity metrics are computed for the generated instruction embeddings.
2.  **Instruction to Q/A Conversion (`inst2gen.py`):**
    *   The augmented instructions are further processed to create a final dataset, typically in a question-answer format suitable for supervised fine-tuning.
3.  **Model Fine-tuning & Evaluation (`test_pipe.py`):**
    *   The generated dataset is used to fine-tune a base LLM (e.g., `microsoft/Phi-4-mini-instruct`).
    *   The fine-tuned model is evaluated on downstream tasks (e.g., `tinyGSM8k`) using `lm-evaluation-harness`.

## Setup

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure `lm-evaluation-harness` is cloned and accessible (as suggested by `lm_eval.py`).

## Running

*   Execute `generate_dataset.py` to create the instruction datasets.
*   Execute `inst2gen.py` to convert these instructions into final dataset format.
*   Execute `test_pipe.py` to run an example fine-tuning and evaluation pipeline. (Modify paths and model names in `test_pipe.py` as needed).