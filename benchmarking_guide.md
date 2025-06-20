# Guide to Benchmarking Models with `benchmark_csbench.py`

## 1. Overview

This guide explains how to use the `benchmark_csbench.py` script to evaluate the performance of Hugging Face language models on tasks from the CS-Bench benchmark. The script is designed to assess models based on their ability to generate correct answers to specific computer science and related domain questions, using an "exact match" criterion against ground truth solutions.

The primary goal is to compare:
*   The original large **teacher model** (e.g., `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`).
*   The smaller **base student model** (e.g., `google/gemma-2b-it`) before any distillation.
*   The **distilled student model** (produced by `distill_demo.py`).

## 2. Prerequisites

*   **Python Environment:** A Python 3.x environment with the necessary libraries installed. These typically include:
    *   `torch`
    *   `transformers`
    *   `datasets` (though not strictly required by `benchmark_csbench.py` for its file-based approach, it's good to have if exploring CS-Bench data further)
    *   `argparse` (standard library)
    *   `logging` (standard library)
    The libraries are generally the same as those required for `distill_demo.py`.
*   **Git:** Git must be installed on your system, as the script uses it to clone the CS-Bench repository.

## 3. Setting up CS-Bench

The `benchmark_csbench.py` script can automatically clone the CS-Bench repository for you.

*   **Default Clone URL:** The script uses `https://github.com/evalplus/csbench.git` (Note: The original subtask prompt mentioned `csbench/csbench`, but `evalplus/csbench` is the more common repository for "CSBench: A Comprehensive Benchmark for Code Swapping and Benchmarking Large Language Models in Computer Science"). Please verify the correct URL for your specific needs. The script currently uses the URL defined in its `CSBENCH_GIT_URL` variable.
*   **Automatic Cloning:** By default, the script will clone CS-Bench into a temporary directory that is cleaned up after execution.
*   **Persistent Clone (Optional):** If you prefer to keep a persistent clone of CS-Bench (e.g., to avoid re-cloning or to explore it manually), you can use the `--csbench_clone_parent_dir /path/to/your/preferred/parent_directory` argument. The repository will be cloned into a subdirectory named `csbench_repo` within the path you provide.

**Critically Important: Identifying Task and Ground Truth Files**

The `benchmark_csbench.py` script requires you to provide **direct paths** to:
1.  A **task file**: This JSON Lines (`.jsonl`) file contains the prompts or questions for the model.
2.  A **ground truth file**: This JSON Lines (`.jsonl`) file contains the corresponding correct answers or solutions.

**You must explore your local clone of the CS-Bench repository to find these files.** The script does *not* automatically discover tasks.

*   **How to find relevant files:**
    *   Navigate into the cloned `csbench_repo` directory.
    *   Look for subdirectories related to your areas of interest, particularly those relevant to **computer science and circuits design knowledge**. Keywords to look for in directory or file names might include:
        *   `circuit_design`, `electronics`, `hardware_description`, `verilog`, `vhdl`, `fpga`
        *   `algorithms`, `data_structures`, `computer_organization`, `architecture`
        *   `multiple_choice`, `question_answering`, `code_generation` (if applicable to your evaluation goals)
    *   Task files often contain terms like "questions", "prompts", or the specific task name (e.g., `computer_organization_definition_multiple_choice.jsonl`).
    *   Ground truth files are usually in a corresponding `ground_truth` subdirectory or have `_ground_truth` in their name.

    For example, you might find a task file like:
    `csbench_repo/computer_organization/comporg_pipelining_concepts.jsonl`
    And its corresponding ground truth:
    `csbench_repo/computer_organization/ground_truth/comporg_pipelining_concepts_ground_truth.jsonl`

    **It is your responsibility to identify valid pairs of task and ground truth files for the specific knowledge areas you want to benchmark.**

## 4. Running the Benchmark

Once you have identified the task and ground truth files, you can run the benchmark script. Below are example command-line calls.

**Remember to replace placeholder paths with the actual paths you found in your CS-Bench clone.**

### a. Evaluating the Teacher Model

*   **Model:** `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (or your chosen teacher model)

```bash
python benchmark_csbench.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --csbench_task_file /path/to/your/csbench_repo/domain/task_file.jsonl \
    --csbench_ground_truth_file /path/to/your/csbench_repo/domain/ground_truth/corresponding_gt_file.jsonl \
    --output_dir ./benchmark_results/teacher_deepseek \
    --trust_remote_code_teacher
```

### b. Evaluating the Base Student Model (before distillation)

*   **Model:** `google/gemma-2b-it` (or your chosen base student model)

```bash
python benchmark_csbench.py \
    --model_name_or_path google/gemma-2b-it \
    --csbench_task_file /path/to/your/csbench_repo/domain/task_file.jsonl \
    --csbench_ground_truth_file /path/to/your/csbench_repo/domain/ground_truth/corresponding_gt_file.jsonl \
    --output_dir ./benchmark_results/student_gemma_base
```

### c. Evaluating the Distilled Student Model

First, ensure you have run `distill_demo.py` (or your distillation script) and know where the distilled student model was saved. The `distill_demo.py` script provided saves to `./distilled_student_model` by default.

*   **Model:** Path to your saved distilled student model (e.g., `./distilled_student_model`).

```bash
# Example: Assuming distill_demo.py saves the model to ./distilled_student_model
# If you ran distill_demo.py and it saved the model elsewhere, adjust the path.
# (No need to run distill_demo.py here if the model is already saved)

python benchmark_csbench.py \
    --model_name_or_path ./distilled_student_model \
    --csbench_task_file /path/to/your/csbench_repo/domain/task_file.jsonl \
    --csbench_ground_truth_file /path/to/your/csbench_repo/domain/ground_truth/corresponding_gt_file.jsonl \
    --output_dir ./benchmark_results/student_gemma_distilled
```

**Note on `--trust_remote_code_teacher`:** Only use this flag if the model you are evaluating (like some teacher models) explicitly requires `trust_remote_code=True` for loading with Hugging Face Transformers. For standard models like Gemma, it's typically not needed.

## 5. Interpreting Results

The script will perform the following:
1.  Load the specified model and tokenizer.
2.  Load the prompts from your `--csbench_task_file`.
3.  Load the references from your `--csbench_ground_truth_file`.
4.  Generate predictions for each prompt using the model.
5.  Compare the predictions against the references to calculate an `exact_match` score (and `accuracy`, which is currently equivalent).

**Output:**
*   **Console Output:** The script will log its progress, including model loading, data loading, prediction generation, and the final evaluation scores.
*   **Output Directory (`--output_dir`):**
    *   A JSON Lines file named `{model_name}_predictions.jsonl` (e.g., `google_gemma-2b-it_predictions.jsonl`) containing the prompts, the model's predictions, and the ground truth references. This is useful for manual inspection of model outputs.
    *   A JSON file named `{model_name}_results.json` (e.g., `google_gemma-2b-it_results.json`) containing the evaluation metrics, such as `exact_match`, `accuracy`, `total_samples`, and `matches`.

By comparing the results from the teacher, base student, and distilled student models on the same CS-Bench tasks, you can quantify the effectiveness of your distillation process in preserving relevant knowledge while reducing model size.
```
