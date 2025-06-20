import os
import subprocess
import json
import tempfile
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset # Not strictly needed for this file-based approach
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Variables/Configuration
CSBENCH_GIT_URL = "https://github.com/csbench/csbench.git"

def setup_csbench_repo(clone_dir_parent):
    """
    Clones the CS-Bench repository into a subdirectory within clone_dir_parent.
    If it already exists and is the correct repo, it uses the existing one.
    """
    repo_path = os.path.join(clone_dir_parent, "csbench_repo")
    try:
        logging.info(f"Setting up CS-Bench repository in {repo_path}...")
        if os.path.exists(repo_path) and os.path.isdir(os.path.join(repo_path, '.git')):
            result = subprocess.run(
                ["git", "-C", repo_path, "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip() == CSBENCH_GIT_URL:
                logging.info(f"CS-Bench repository already exists and seems correct in {repo_path}.")
                # Optional: git pull to update if needed, for now assume it's fine.
                # logging.info(f"Attempting to update repository at {repo_path}...")
                # subprocess.run(["git", "-C", repo_path, "pull"], check=True, capture_output=True, text=True)
                return repo_path
            else:
                logging.warning(f"Found a directory at {repo_path}, but it's not the correct CS-Bench repo. Re-cloning.")
                shutil.rmtree(repo_path)

        logging.info(f"Cloning CS-Bench repository from {CSBENCH_GIT_URL} into {repo_path}...")
        subprocess.run(
            ["git", "clone", CSBENCH_GIT_URL, repo_path],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info(f"CS-Bench repository cloned successfully into {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during CS-Bench repository setup: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during CS-Bench repository setup: {e}")
        return None

def prepare_model_and_tokenizer(model_name_or_path, trust_remote_code=False):
    """
    Loads the model and tokenizer.
    Sets model to eval mode and moves to device.
    """
    logging.info(f"Loading model and tokenizer for: {model_name_or_path}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if device.type == 'cuda' else None # Optimize for GPU if available
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                logging.info("Tokenizer missing pad_token_id, using eos_token_id as pad_token_id.")
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                logging.warning("Tokenizer missing pad_token_id and eos_token_id. Adding a new pad token: '[PAD]'.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))


        model.eval()
        model.to(device)
        logging.info(f"Model {model_name_or_path} loaded successfully on {device}.")
        return model, tokenizer, device
    except Exception as e:
        logging.error(f"Error loading model {model_name_or_path}: {e}", exc_info=True)
        return None, None, None

def load_benchmark_data(task_file_path, ground_truth_file_path):
    """
    Reads the task file (prompts) and ground truth file (references).
    Assumes JSONL format. Task file needs a "prompt" key (or "question", "instruction").
    Ground truth file needs a "reference" key (or "solution", "answer").
    """
    logging.info(f"Loading benchmark data from task file: {task_file_path}")
    logging.info(f"Loading ground truth from: {ground_truth_file_path}")

    prompts = []
    references = []

    try:
        with open(task_file_path, 'r', encoding='utf-8') as f_task:
            for line_num, line in enumerate(f_task):
                try:
                    data = json.loads(line)
                    # Adapt these keys based on CS-Bench's actual format
                    prompt = data.get("prompt") or data.get("question") or data.get("problem") or data.get("instruction") or data.get("input")
                    if prompt is None:
                        logging.warning(f"Missing 'prompt' (or similar key) in task file {task_file_path} at line {line_num+1}")
                        continue
                    prompts.append(prompt)
                except json.JSONDecodeError:
                    logging.error(f"JSON decode error in task file {task_file_path} at line {line_num+1}: {line.strip()}")
                    return None, None

        with open(ground_truth_file_path, 'r', encoding='utf-8') as f_gt:
            for line_num, line in enumerate(f_gt):
                try:
                    data = json.loads(line)
                     # Adapt these keys based on CS-Bench's actual format
                    reference = data.get("reference") or data.get("solution") or data.get("answer") or data.get("output")
                    if reference is None:
                        logging.warning(f"Missing 'reference' (or similar key) in ground truth file {ground_truth_file_path} at line {line_num+1}")
                        continue
                    references.append(reference)
                except json.JSONDecodeError:
                    logging.error(f"JSON decode error in ground truth file {ground_truth_file_path} at line {line_num+1}: {line.strip()}")
                    return None, None

        if len(prompts) != len(references):
            logging.error(f"Mismatch in number of prompts ({len(prompts)}) and references ({len(references)}). Please check files.")
            return None, None

        if not prompts:
            logging.error("No prompts loaded. Please check task file and keys.")
            return None, None

        logging.info(f"Successfully loaded {len(prompts)} prompts and {len(references)} references.")
        return prompts, references

    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading benchmark data: {e}", exc_info=True)
        return None, None

def generate_predictions(model, tokenizer, prompts, device):
    """
    Generates predictions for the given prompts using the model.
    """
    logging.info(f"Starting prediction generation for {len(prompts)} prompts...")
    predictions = []

    # It's crucial that tokenizer.pad_token_id is set.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # This case should have been handled in prepare_model_and_tokenizer, but double check.
            logging.error("tokenizer.pad_token_id is None and eos_token_id is also None. Cannot proceed with generation.")
            return None

    for i, prompt_text in enumerate(prompts):
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device) # Max length can be adjusted

            # Generate output
            # Adjust generation parameters as needed for the specific model/task
            # Common parameters: max_new_tokens, do_sample, temperature, top_k, top_p
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256, # Max tokens to generate after the prompt
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False # For deterministic output; set True for diverse answers
                    # temperature=0.7, # Example if do_sample=True
                    # top_k=50,        # Example if do_sample=True
                )

            # Decode the generated tokens, skipping special tokens (like padding, eos)
            # The slicing inputs["input_ids"].shape[-1]: removes the prompt part from the output
            generated_text = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            predictions.append(generated_text.strip())
            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                logging.info(f"Generated prediction for {i+1}/{len(prompts)} prompts.")
        except Exception as e:
            logging.error(f"Error generating prediction for prompt {i+1} ('{prompt_text[:50]}...'): {e}", exc_info=True)
            predictions.append(f"Error: Could not generate prediction. {e}")

    logging.info("Prediction generation finished.")
    return predictions

def run_evaluation(predictions, references):
    """
    Calculates exact match score between predictions and references.
    """
    if not predictions or not references or len(predictions) != len(references):
        logging.error("Invalid input for evaluation. Predictions and references must be non-empty and of the same length.")
        return {"exact_match": 0.0, "accuracy": 0.0, "error": "Invalid input"}

    exact_match_count = 0
    for pred, ref in zip(predictions, references):
        if str(pred).strip() == str(ref).strip():
            exact_match_count += 1

    total_samples = len(references)
    exact_match_score = (exact_match_count / total_samples) if total_samples > 0 else 0.0

    logging.info(f"Evaluation complete: Exact Match = {exact_match_count}/{total_samples} ({exact_match_score:.4f})")
    return {"exact_match": exact_match_score, "accuracy": exact_match_score, "total_samples": total_samples, "matches": exact_match_count}

def main():
    parser = argparse.ArgumentParser(description="""
Run CS-Bench benchmark tasks on a given Hugging Face model.
You need to specify the model, and the paths to the task prompt file and
the ground truth file from a local clone of the CS-Bench repository.

Example CS-Bench file structure (explore your CS-Bench clone for actual paths):
csbench_repo/
├── README.md
├── assertion           # Example domain
│   ├── assertion_definition_multiple_choice.jsonl (task_file_path)
│   └── ground_truth
│       └── assertion_definition_multiple_choice_ground_truth.jsonl (ground_truth_file_path)
├── ... (other domains like basic_programming, computer_organization etc.)
└── scripts
    ├── create_input.py
    ├── gen_judgment.py
    └── show_result.py

To run this script, first clone CS-Bench. Then, identify a task file (e.g., a .jsonl file with prompts)
and its corresponding ground truth file. Provide their full paths to this script.
    """)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model name or local path.")
    parser.add_argument("--csbench_task_file", type=str, required=True, help="Path to the CS-Bench task file (e.g., a .jsonl file containing prompts/questions). This path should be relative to your CS-Bench clone or an absolute path.")
    parser.add_argument("--csbench_ground_truth_file", type=str, required=True, help="Path to the corresponding CS-Bench ground truth file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions and results.")
    parser.add_argument("--csbench_clone_parent_dir", type=str, default=None, help="Optional: Parent directory to clone CS-Bench into. If None, a temporary directory will be used.")
    parser.add_argument("--trust_remote_code_teacher", action='store_true', help="Boolean flag for models requiring trust_remote_code=True (e.g. some teacher models).")

    args = parser.parse_args()

    logging.info(f"Starting benchmark for model: {args.model_name_or_path}")
    logging.info(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine where to clone CS-Bench
    if args.csbench_clone_parent_dir:
        os.makedirs(args.csbench_clone_parent_dir, exist_ok=True)
        csbench_base_dir = args.csbench_clone_parent_dir
        temp_dir_manager = None # Not using tempfile.TemporaryDirectory
    else:
        temp_dir_manager = tempfile.TemporaryDirectory()
        csbench_base_dir = temp_dir_manager.name

    try:
        # 1. Setup CS-Bench (clone or verify existing)
        # Note: The script itself doesn't use the cloned repo directly for file loading in this version,
        # as task_file and ground_truth_file are full paths provided by user.
        # However, setup_csbench_repo can be useful if future versions integrate more deeply.
        csbench_repo_path = setup_csbench_repo(csbench_base_dir)
        if not csbench_repo_path:
            logging.error("Failed to setup CS-Bench repository. Aborting.")
            return
        logging.info(f"CS-Bench repository is available at: {csbench_repo_path}")

        # 2. Prepare model and tokenizer
        model, tokenizer, device = prepare_model_and_tokenizer(args.model_name_or_path, args.trust_remote_code_teacher)
        if not model or not tokenizer:
            logging.error(f"Failed to load model or tokenizer for {args.model_name_or_path}. Aborting.")
            return

        # 3. Load benchmark data
        prompts, references = load_benchmark_data(args.csbench_task_file, args.csbench_ground_truth_file)
        if prompts is None or references is None:
            logging.error("Failed to load benchmark data. Aborting.")
            return

        # 4. Generate predictions
        predictions = generate_predictions(model, tokenizer, prompts, device)
        if predictions is None:
            logging.error("Failed to generate predictions. Aborting.")
            return

        # Save predictions
        predictions_save_path = os.path.join(args.output_dir, f"{os.path.basename(args.model_name_or_path).replace('/','_')}_predictions.jsonl")
        with open(predictions_save_path, 'w', encoding='utf-8') as f_pred:
            for i in range(len(prompts)):
                json.dump({"id": i, "prompt": prompts[i], "prediction": predictions[i], "reference": references[i]}, f_pred)
                f_pred.write('\n')
        logging.info(f"Predictions saved to: {predictions_save_path}")

        # 5. Run evaluation
        evaluation_results = run_evaluation(predictions, references)

        # Save results
        results_save_path = os.path.join(args.output_dir, f"{os.path.basename(args.model_name_or_path).replace('/','_')}_results.json")
        with open(results_save_path, 'w', encoding='utf-8') as f_res:
            json.dump(evaluation_results, f_res, indent=2)
        logging.info(f"Evaluation results saved to: {results_save_path}")

        logging.info("\n--- CS-Bench Evaluation Results ---")
        logging.info(json.dumps(evaluation_results, indent=2))

    finally:
        if temp_dir_manager: # Cleanup if tempfile.TemporaryDirectory was used
            temp_dir_manager.cleanup()
            logging.info(f"Temporary directory {csbench_base_dir} cleaned up.")
        elif args.csbench_clone_parent_dir:
             logging.info(f"CS-Bench repository cloned/verified in {os.path.join(args.csbench_clone_parent_dir, 'csbench_repo')}. Not removing as a persistent path was provided.")


if __name__ == "__main__":
    main()
