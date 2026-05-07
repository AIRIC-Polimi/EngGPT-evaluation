# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "jupyterlab>=4.2.4",
#     "lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git",
#     "torch>=2.9.1",
#     "transformers>=5.0.0",
#     "accelerate>=1.12.0",
#     "openai>=2.15.0",
#     "PyJWT>=2.10.1",
#     "dotenv>=0.9.9",
#     "tabulate>=0.9.0",
#     "wonderwords>=3.0.1",
#     "nltk>=3.9.4",
# ]
# ///

import sys
from pathlib import Path
# Add the root directory to path so we can import shared utils
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import gc
import multiprocessing as mp

# NOTE: this is needed to run the humaneval task, which relies on sandboxed code execution
import os
import time
import traceback
from copy import deepcopy

import lm_eval
import torch
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.utils import setup_logging
from utils import save_evaluation_results

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Try to increase logging verbosity to spot deadlocks
import logging

import datasets
import transformers

logging.basicConfig(
    level=logging.DEBUG,
)

datasets.logging.set_verbosity_info()
transformers.logging.set_verbosity_info()


def _get_model_custom_config(model_name: str) -> EvaluatorConfig | None:
    """Get the custom configuration for a specific model.

    Args:
        model_name (str): The name of the model.

    Returns:
        EvaluatorConfig | None: The custom configuration if it exists, otherwise None.
    """
    config_base_path = Path(__file__).parent.parent / "model_configs"

    # Support both HuggingFace repo IDs (e.g. "Qwen/Qwen3-8B") and local paths
    # (e.g. "/path/to/MyModel"). We use the final path segment as key.
    model_key = Path(model_name).name if Path(model_name).exists() else model_name.split("/")[-1]

    model_config_path = config_base_path / model_key / "template_config_vllm.yaml"
    if Path(model_config_path).exists():
        return EvaluatorConfig.from_config(str(model_config_path))
    return None


def _evaluate_model_task_pair(model: str, task: str, template_config_path: str, skip_already_existing: bool) -> None:
    setup_logging("DEBUG")

    base_template_config = EvaluatorConfig.from_config(str(template_config_path))
    model_custom_config = _get_model_custom_config(model)
    model_template_config = model_custom_config or base_template_config

    if model_custom_config is not None:
        print(f"Using custom configuration for {model}")  # noqa: T201

    print(f"Evaluating {task} on {model}")  # noqa: T201

    config = deepcopy(model_template_config)
    config.model_args["pretrained"] = model
    config.tasks = [task]
    config.output_path = f"{config.output_path}/{model.split('/')[-1]}/{task}"

    if config.output_path is None:
        raise ValueError("output_path must be set in evaluator config")

    output_path = Path(config.output_path)

    # Make sure output_path exists
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    elif skip_already_existing and any(output_path.iterdir()):
        print(f"Skipping evaluation for {task} on {model} as results already exist.")  # noqa: T201
        return
    print(config)  # noqa: T201
    # Process tasks
    if task == "ruler":
        task_manager = config.process_tasks(metadata={"max_seq_lengths": [4096, 16384, 32768], "pretrained": model})
    else:
        task_manager = config.process_tasks()

    # Run evaluation
    start_time = time.time()
    results = lm_eval.simple_evaluate(
        model=config.model,
        model_args=config.model_args,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        task_manager=task_manager,
        log_samples=False if task == "ruler" else config.log_samples,
        gen_kwargs=config.gen_kwargs,
        apply_chat_template=config.apply_chat_template,
        system_instruction=config.system_instruction,
        # NOTE: this is needed to run the humaneval task, which relies on sandboxed code execution
        confirm_run_unsafe_code=True,
    )

    # Record evaluation time
    end_time = time.time()
    elapsed_time = end_time - start_time
    results["evaluation_time_seconds"] = elapsed_time

    # Save results
    save_evaluation_results(results, config.output_path, is_serializable=False)

    # Best effort cleanup in child process; process exit fully releases CUDA context.
    gc.collect()
    torch.cuda.empty_cache()


def main():
    default_models_list = [
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    ]
    default_tasks_list = [
        # "arc_challenge_chat",
        # "arc_challenge_chat_openai",
        # "aime24",
        # "aime25",
        # "mmlu_redux_stem_generative",
        # "mmlu_redux_other_generative",
        # "mmlu_redux_social_sciences_generative",
        # "mmlu_redux_humanities_generative",
        # "mmlu_generative",
        # "mmlu_cot_llama",
        # "gsm8k_llama",
        # "humaneval", # to run it needs the env variable HF_ALLOW_CODE_EVAL to be set to 1
        # "humaneval_instruct",
        # Custom tasks
        # "arc_challenge_cot_it_fixed",
        # "arc_challenge_cot_fixed",
        # "aime24_pass_avg_8",
        # "aime25_pass_avg_8",
        # "arc_challenge_cot_airic",
        # "arc_challenge_cot_it_airic",
        # "mmlu_cot_fixed_llama",
        # "humaneval_8_instruct_cot",
        # "ruler",
        # "italic_from_mmlu_cot",
        # "italic_from_arc_chat",
        # "italic_cot",
        "italic_fast",
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", nargs="+", default=default_models_list, help="List of HuggingFace model IDs to use."
    )
    parser.add_argument("--tasks", nargs="+", default=default_tasks_list, help="List of evaluation tasks to run.")
    parser.add_argument(
        "--config",
        type=str,
        dest="template_config_path",  # Maps the argument to this variable name
        default=str(Path(__file__).parent.parent / "model_configs" / "template_config_base_vllm.yaml"),
        help="Path to the template configuration YAML file.",
    )
    parser.add_argument(
        "--skip-already-existing",
        action="store_true",
        default=True,
        help="If set, skip evaluation for model-task pairs that already have results saved.",
    )

    args = parser.parse_args()
    models = args.models
    tasks = args.tasks
    template_config_path = args.template_config_path
    skip_already_existing = args.skip_already_existing

    spawn_context = mp.get_context("spawn")

    for model in models:
        for task in tasks:
            print(f"Launching subprocess for {task} on {model}")  # noqa: T201
            try:
                process = spawn_context.Process(
                    target=_evaluate_model_task_pair,
                    args=(model, task, template_config_path, skip_already_existing),
                )
                process.start()
                process.join()

                if process.exitcode != 0:
                    print(f"ERROR while evaluating {task} on {model}, subprocess exited with code {process.exitcode}")  # noqa: T201

            except Exception:
                print(traceback.format_exc())  # noqa: T201
                print(f"ERROR while evaluating {task} on {model}, continuing with the next one")  # noqa: T201

            # Small delay to smooth GPU teardown/startup transitions.
            time.sleep(30)
            # Best effort cleanup in child process; process exit fully releases CUDA context.
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(30)


if __name__ == "__main__":
    main()
