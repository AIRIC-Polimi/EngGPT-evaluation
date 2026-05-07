# /// script
# requires-python = "<3.13,>=3.12"
# dependencies = ["evalscope>=1.4.2", "bfcl-eval>=2026.2.9", "soundfile>=0.13.1", "requests>=2.32.5", "vllm>=0.15.1"]
# ///


import logging
import subprocess  # noqa: S404 # nosec
import time
from dataclasses import asdict
from pathlib import Path

import requests
from evalscope import run_task
from evalscope.config import TaskConfig
from evalscope_evaluation.core import ALL_EVALS_CONFIG, GenerationConfig

logger = logging.getLogger(__name__)


def wait_until_healthcheck(healthcheck_url: str, timeout_seconds: int) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(healthcheck_url, timeout=1)
            if response.status_code == 200:
                logger.info("Server is up and running in %s seconds", time.time() - start_time)
                return True
        except requests.ConnectionError:
            # Server isn't listening yet, keep looping
            pass

        # Wait a few seconds before checking again to avoid spamming
        time.sleep(5)
    return False


def eval_bfcl_v4(
    model: str,
    work_dir: str,
    api_url: str,
    api_key: str,
    is_fc_model: bool = False,
    batch_size: int = 32,
    limit: int | None = None,
    generation_config: GenerationConfig = GenerationConfig(),
    underscore_to_dots: bool = False,
):
    task_cfg = TaskConfig(
        model=model,
        eval_type="openai_api",
        eval_batch_size=batch_size,
        api_url=api_url,
        api_key=api_key,
        generation_config=asdict(generation_config),
        work_dir=work_dir,
        datasets=["bfcl_v4"],
        dataset_args={
            "bfcl_v4": {
                "subset_list": [
                    "irrelevance",  # samples=240,  mean=85.98,   min=55,  max=203  # noqa: ERA001, RUF100
                    "live_irrelevance",  # samples=884,  mean=198.72,  min=36,  max=10805  # noqa: ERA001, RUF100
                    "live_multiple",  # samples=1053, mean=149.16,  min=42,  max=4828  # noqa: ERA001, RUF100
                    "live_parallel",  # samples=16,   mean=163.88,  min=88,  max=365  # noqa: ERA001, RUF100
                    "live_parallel_multiple",  # samples=24,   mean=217.67,  min=72,  max=650  # noqa: ERA001, RUF100
                    "live_relevance",  # samples=16,   mean=385.94,  min=71,  max=1830  # noqa: ERA001, RUF100
                    "live_simple",  # samples=258,  mean=171.91,  min=47,  max=1247  # noqa: ERA001, RUF100
                    "memory_kv",  # samples=155,  mean=646.46,  min=585, max=826  # noqa: ERA001, RUF100
                    "memory_rec_sum",  # samples=155,  mean=646.46,  min=585, max=826  # noqa: ERA001, RUF100
                    "memory_vector",  # samples=155,  mean=646.46,  min=585, max=826  # noqa: ERA001, RUF100
                    "multi_turn_base",  # samples=200,  mean=808.43,  min=106, max=1756  # noqa: ERA001, RUF100
                    # "multi_turn_long_context",     # samples=200,  mean=808.41,  min=106, max=1756  # noqa: ERA001, RUF100
                    "multi_turn_miss_func",  # samples=200,  mean=811.88,  min=110, max=1760  # noqa: ERA001, RUF100
                    "multi_turn_miss_param",  # samples=200,  mean=863.88,  min=167, max=1791  # noqa: ERA001, RUF100
                    "multiple",  # samples=200,  mean=120.75,  min=65,  max=229  # noqa: ERA001, RUF100
                    "parallel",  # samples=200,  mean=297.93,  min=69,  max=781  # noqa: ERA001, RUF100
                    "parallel_multiple",  # samples=200,  mean=432.17,  min=88,  max=1249  # noqa: ERA001, RUF100
                    "simple_java",  # samples=100,  mean=228.93,  min=138, max=531  # noqa: ERA001, RUF100
                    "simple_javascript",  # samples=50,   mean=222.34,  min=118, max=637  # noqa: ERA001, RUF100
                    "simple_python",  # samples=400,  mean=119.66,  min=57,  max=303  # noqa: ERA001, RUF100
                    "web_search_base",  # samples=100,  mean=831.01,  min=748, max=963  # noqa: ERA001, RUF100
                    "web_search_no_snippet",  # samples=100,  mean=831.01,  min=748, max=963  # noqa: ERA001, RUF100
                ],
                "extra_params": {
                    "is_fc_model": is_fc_model,
                    # Model refuses to use dots (`.`) in function names; set this to automatically
                    # convert dots to underscores during evaluation.
                    "underscore_to_dot": underscore_to_dots,
                },
            }
        },
        limit=limit,
    )

    run_task(task_cfg=task_cfg)


def _serve_model_vllm(
    model_name_hub: str,
    model_name_served: str | None = None,
    port: int = 8000,
    start_timeout: int = 600,
    vllm_log_path: str | None = None,
    is_fc_model: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
) -> subprocess.Popen[bytes]:
    command = [
        "vllm",
        "serve",
        model_name_hub,
        "--served-model-name",
        model_name_served,
        "--trust-remote-code",
        "--max-model-len",
        "auto",  # NOTE: this will autoselect the largest model len that fits in GPU memory
        # NOTE: this is to use the python instead of the Rust tokenizer, which is a little slower but handles better concurrent requests
        "--tokenizer-mode",
        "mistral" if "mistral" in model_name_hub else ("auto" if "Minerva" in model_name_hub else "slow"),
        "--port",
        str(port),
    ]
    if is_fc_model:
        if tool_call_parser is None:
            logger.error(
                "Unable to run model %s in FC mode without a tool call parser for vLLM. Skipping eval.",
                model_name_served,
            )
            return None
        command.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser])
    if reasoning_parser is not None:
        command.extend(["--reasoning-parser", reasoning_parser])
    if chat_template is not None:
        command.extend(["--chat-template", chat_template])
    logger.info("Starting vLLM server: %s", " ".join(command))

    # Setup the logging destination for the subprocess
    if vllm_log_path:
        logger.info("[Main] Redirecting vLLM logs to: %s", vllm_log_path)
        vllm_log_dest = Path.open(vllm_log_path, "w")
    else:
        logger.info("[Main] Discarding vLLM logs (redirecting to /dev/null)")
        vllm_log_dest = subprocess.DEVNULL

    # Start the vLLM subprocess
    # Note: server logs are re-routed to subprocess.PIPE or to a custom file to avoid polluting stdout and eval results
    process = subprocess.Popen(  # noqa: S603 # nosec
        command,
        stdout=vllm_log_dest,
        stderr=vllm_log_dest,  # Catch errors too so they don't bleed into the terminal
    )

    health_url = f"http://localhost:{port}/health"

    logger.info("Waiting for vLLM server to be ready (this may take a few minutes)...")

    # Poll the server until it responds or times out
    server_ready = wait_until_healthcheck(healthcheck_url=health_url, timeout_seconds=start_timeout)

    if not server_ready:
        logger.info("Error: Server failed to start within the %s timeout.", start_timeout)
        process.terminate()
        if vllm_log_path:
            vllm_log_dest.close()
        return None

    return process


def serve_model_and_eval_bfcl_v4(
    model_name_hub: str,
    eval_work_dir: str,
    model_name_served: str | None = None,
    port: int = 8000,
    start_timeout: int = 600,
    vllm_log_path: str | None = None,
    is_fc_model: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
    batch_size: int = 32,
    limit: int | None = None,
    generation_config: GenerationConfig = GenerationConfig(),
    skip_vllm_serve: bool = False,
    cloud_model_url: str | None = None,
    cloud_model_api_key: str | None = None,
    underscore_to_dots: bool = False,
):
    """
    Starts vLLM, waits for it to be ready, runs evaluation, and cleans up.
    """
    model_name_served = model_name_served or model_name_hub.split("/")[-1]
    vllm_process = None
    if not skip_vllm_serve and cloud_model_url is None:
        vllm_process = _serve_model_vllm(
            model_name_hub=model_name_hub,
            model_name_served=model_name_served,
            port=port,
            start_timeout=start_timeout,
            vllm_log_path=vllm_log_path,
            is_fc_model=is_fc_model,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            chat_template=chat_template,
        )

    if vllm_process is None and cloud_model_url is None:
        # We are relying on a local vllm server that we did not start as a subprocess: wait for it to be ready
        health_url = f"http://localhost:{port}/health"
        logger.info("Waiting for external vLLM server to be ready (this may take a few minutes)...")
        server_ready = wait_until_healthcheck(healthcheck_url=health_url, timeout_seconds=start_timeout)
        if not server_ready:
            logger.info("Error: external server failed to start within the %s timeout.", start_timeout)
            return

    try:
        # Run the evaluation now that the server is definitely ready
        if cloud_model_url is not None:
            assert cloud_model_api_key is not None and len(cloud_model_api_key) > 0, (
                f"Missing API KEY for cloud model {model_name_served}"
            )
        eval_bfcl_v4(
            model=model_name_served,
            is_fc_model=is_fc_model,
            batch_size=batch_size,
            limit=limit,
            generation_config=generation_config,
            work_dir=eval_work_dir,
            api_url=cloud_model_url or f"http://127.0.0.1:{port}/v1",
            api_key=cloud_model_api_key or "EMPTY",
            underscore_to_dots=underscore_to_dots,
        )

    except Exception as e:
        logger.info("An error occurred during evaluation: %s", e)

    finally:
        if vllm_process is not None:
            # Guarantee that the vLLM server shuts down regardless of success or failure
            logger.info("Shutting down vLLM server subprocess...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=10)  # Give it 10 seconds to shut down gracefully
                logger.info("Server shut down cleanly.")
            except subprocess.TimeoutExpired:
                logger.info("Server taking too long to shut down. Killing process.")
                vllm_process.kill()  # Force kill if it hangs


# Link to public leaderboard to compare results: https://gorilla.cs.berkeley.edu/leaderboard.html
if __name__ == "__main__":
    for model, config in ALL_EVALS_CONFIG:
        if config.skip_eval:
            logger.info("Skipping %s since its config has skip_eval set to True", model)
            continue
        serve_model_and_eval_bfcl_v4(
            model_name_hub=model,
            vllm_log_path=f"{model.split('/')[-1][:15]}_bfcl_vllm_logs.log",
            eval_work_dir=str(Path(__file__).parent.parent.parent / "data" / "bfcl-v4"),
            batch_size=config.batch_size,
            generation_config=config.generation_config,
            is_fc_model=config.is_fc_model,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            chat_template=config.chat_template,
            skip_vllm_serve=config.skip_vllm_serve or config.cloud_model_url is not None,
            cloud_model_url=config.cloud_model_url,
            cloud_model_api_key=config.cloud_model_api_key,
            underscore_to_dots=config.underscore_to_dots,
        )
