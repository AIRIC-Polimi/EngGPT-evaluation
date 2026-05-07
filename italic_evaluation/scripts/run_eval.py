# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "accelerate==1.13.0",
#     "aiohappyeyeballs==2.6.1",
#     "aiohttp==3.13.3",
#     "aiosignal==1.4.0",
#     "annotated-doc==0.0.4",
#     "annotated-types==0.7.0",
#     "anthropic==0.84.0",
#     "antlr4-python3-runtime==4.9.3",
#     "anyio==4.12.1",
#     "apache-tvm-ffi==0.1.9",
#     "astor==0.8.1",
#     "attrs==25.4.0",
#     "blake3==1.0.8",
#     "cachetools==7.0.4",
#     "cbor2==5.8.0",
#     "certifi==2026.2.25",
#     "cffi==2.0.0",
#     "charset-normalizer==3.4.5",
#     "click==8.3.1",
#     "cloudpickle==3.1.2",
#     "compressed-tensors==0.13.0",
#     "cryptography==46.0.5",
#     "cuda-bindings==12.9.4",
#     "cuda-pathfinder==1.4.1",
#     "cuda-python==12.9.4",
#     "cupy-cuda12x==14.0.1",
#     "depyf==0.20.0",
#     "dill==0.4.1",
#     "diskcache==5.6.3",
#     "distro==1.9.0",
#     "dnspython==2.8.0",
#     "docstring-parser==0.17.0",
#     "einops==0.8.2",
#     "email-validator==2.3.0",
#     "fastapi==0.135.1",
#     "fastapi-cli==0.0.24",
#     "fastapi-cloud-cli==0.14.1",
#     "fastar==0.8.0",
#     "filelock==3.25.0",
#     "flashinfer-python==0.6.4",
#     "frozenlist==1.8.0",
#     "fsspec==2026.2.0",
#     "gguf==0.18.0",
#     "googleapis-common-protos==1.73.0",
#     "grpcio==1.78.0",
#     "grpcio-reflection==1.78.0",
#     "h11==0.16.0",
#     "hf-xet==1.3.2",
#     "httpcore==1.0.9",
#     "httptools==0.7.1",
#     "httpx==0.28.1",
#     "httpx-sse==0.4.3",
#     "huggingface-hub==0.36.2",
#     "hydra-core==1.3.2",
#     "idna==3.11",
#     "ijson==3.5.0",
#     "importlib-metadata==8.7.1",
#     "interegular==0.3.3",
#     "jinja2==3.1.6",
#     "jiter==0.13.0",
#     "jmespath==1.1.0",
#     "jsonschema==4.26.0",
#     "jsonschema-specifications==2025.9.1",
#     "kaldi-native-fbank==1.22.3",
#     "lark==1.2.2",
#     "llguidance==1.3.0",
#     "llvmlite==0.44.0",
#     "lm-format-enforcer==0.11.3",
#     "loguru==0.7.3",
#     "markdown-it-py==4.0.0",
#     "markupsafe==3.0.3",
#     "mcp==1.26.0",
#     "mdurl==0.1.2",
#     "mistral-common==1.9.1",
#     "model-hosting-container-standards==0.1.13",
#     "mpmath==1.3.0",
#     "msgpack==1.1.2",
#     "msgspec==0.20.0",
#     "multidict==6.7.1",
#     "networkx==3.6.1",
#     "ninja==1.13.0",
#     "numba==0.61.2",
#     "numpy==2.2.6",
#     "nvidia-cublas-cu12==12.8.4.1",
#     "nvidia-cuda-cupti-cu12==12.8.90",
#     "nvidia-cuda-nvrtc-cu12==12.8.93",
#     "nvidia-cuda-runtime-cu12==12.8.90",
#     "nvidia-cudnn-cu12==9.10.2.21",
#     "nvidia-cudnn-frontend==1.18.0",
#     "nvidia-cufft-cu12==11.3.3.83",
#     "nvidia-cufile-cu12==1.13.1.3",
#     "nvidia-curand-cu12==10.3.9.90",
#     "nvidia-cusolver-cu12==11.7.3.90",
#     "nvidia-cusparse-cu12==12.5.8.93",
#     "nvidia-cusparselt-cu12==0.7.1",
#     "nvidia-cutlass-dsl==4.4.1",
#     "nvidia-cutlass-dsl-libs-base==4.4.1",
#     "nvidia-ml-py==13.590.48",
#     "nvidia-nccl-cu12==2.27.5",
#     "nvidia-nvjitlink-cu12==12.8.93",
#     "nvidia-nvshmem-cu12==3.4.5",
#     "nvidia-nvtx-cu12==12.8.90",
#     "omegaconf==2.3.0",
#     "openai==2.24.0",
#     "openai-harmony==0.0.8",
#     "opencv-python-headless==4.13.0.92",
#     "opentelemetry-api==1.40.0",
#     "opentelemetry-exporter-otlp==1.40.0",
#     "opentelemetry-exporter-otlp-proto-common==1.40.0",
#     "opentelemetry-exporter-otlp-proto-grpc==1.40.0",
#     "opentelemetry-exporter-otlp-proto-http==1.40.0",
#     "opentelemetry-proto==1.40.0",
#     "opentelemetry-sdk==1.40.0",
#     "opentelemetry-semantic-conventions==0.61b0",
#     "opentelemetry-semantic-conventions-ai==0.4.15",
#     "outlines-core==0.2.11",
#     "packaging==26.0",
#     "pandas==3.0.1",
#     "partial-json-parser==0.2.1.1.post7",
#     "pillow==12.1.1",
#     "prometheus-client==0.24.1",
#     "prometheus-fastapi-instrumentator==7.1.0",
#     "propcache==0.4.1",
#     "protobuf==6.33.5",
#     "psutil==7.2.2",
#     "py-cpuinfo==9.0.0",
#     "pybase64==1.4.3",
#     "pycountry==26.2.16",
#     "pycparser==3.0",
#     "pydantic==2.12.5",
#     "pydantic-core==2.41.5",
#     "pydantic-extra-types==2.11.0",
#     "pydantic-settings==2.13.1",
#     "pygments==2.19.2",
#     "pyjwt==2.11.0",
#     "python-dateutil==2.9.0.post0",
#     "python-dotenv==1.2.2",
#     "python-json-logger==4.0.0",
#     "python-multipart==0.0.22",
#     "pyyaml==6.0.3",
#     "pyzmq==27.1.0",
#     "quack-kernels==0.2.10",
#     "ray==2.54.0",
#     "referencing==0.37.0",
#     "regex==2026.2.28",
#     "requests==2.32.5",
#     "rich==14.3.3",
#     "rich-toolkit==0.19.7",
#     "rignore==0.7.6",
#     "rpds-py==0.30.0",
#     "safetensors==0.7.0",
#     "sentencepiece==0.2.1",
#     "sentry-sdk==2.54.0",
#     "setproctitle==1.3.7",
#     "setuptools==80.10.2",
#     "shellingham==1.5.4",
#     "six==1.17.0",
#     "sniffio==1.3.1",
#     "sse-starlette==3.3.2",
#     "starlette==0.52.1",
#     "supervisor==4.3.0",
#     "sympy==1.14.0",
#     "tabulate==0.10.0",
#     "tenacity==9.1.4",
#     "tiktoken==0.12.0",
#     "tokenizers==0.22.2",
#     "torch==2.10.0",
#     "torch-c-dlpack-ext==0.1.5",
#     "torchaudio==2.10.0",
#     "torchvision==0.25.0",
#     "tqdm==4.67.3",
#     "transformers==4.57.6",
#     "triton==3.6.0",
#     "typer==0.24.1",
#     "typing-extensions==4.15.0",
#     "typing-inspection==0.4.2",
#     "urllib3==2.6.3",
#     "uvicorn==0.41.0",
#     "uvloop==0.22.1",
#     "vllm==0.17.0",
#     "watchfiles==1.1.1",
#     "websockets==16.0",
#     "xgrammar==0.1.29",
#     "yarl==1.23.0",
#     "zipp==3.23.0",
# ]
# ///

import sys
from pathlib import Path
# Add the root directory to path so we can import shared utils
sys.path.append(str(Path(__file__).parent.parent.parent))

import gc
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hydra
import numpy as np
import pandas as pd
import tenacity
import torch
from core.enggpt_vllm import register_enggpt2moe
from loguru import logger
from omegaconf import DictConfig, open_dict
from pydantic import BaseModel
from tqdm import tqdm
from utils import save_evaluation_results

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logger.opt(colors=True)

DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."

QUERY_TEMPLATE_MULTICHOICE = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. L'ultima riga della tua risposta deve essere nel seguente formato: 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Ragiona brevemente prima di rispondere.

{question}

{options}
""".strip()

QUERY_TEMPLATE_MULTICHOICE_FAST = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

{question}

{options}

Risposta:
""".strip()


class ProviderEnum(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    VLLM = "vllm"


class Sample(BaseModel):
    messages: List[Dict[str, str]]
    answer: str


class ChatCompletionRequest(BaseModel):
    index: int
    provider: ProviderEnum
    model: str
    messages: List[Dict[str, str]]
    answer: str
    temperature: float = 0.7
    max_tokens: int = 150
    fast: bool = False


class ChatCompletionResponse(ChatCompletionRequest):
    output: str


class RateLimiter(object):
    """Easy peasy rate limiter to throttle requests."""

    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.request_times = deque()

    def wait(self):
        current_time = time.time()
        with self.lock:
            # remove old ones
            while self.request_times and current_time - self.request_times[0] >= 60:
                self.request_times.popleft()

            if len(self.request_times) < self.rate:
                self.request_times.append(current_time)
                return True
            return False

    def throttle_requests(self):
        """Wait until it can proceed."""
        while not self.wait():
            time.sleep(0.1)

    def get_total_requests(self):
        with self.lock:
            return len(self.request_times)


class BaseProvider(ABC):
    @abstractmethod
    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        pass


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        import google.generativeai as genai  # type: ignore

        self.genai = genai
        self.genai.configure(api_key=api_key)

    @staticmethod
    def _change_role(message: Dict[str, str]) -> Dict[str, str]:
        return {"role": message["role"], "parts": message["content"]}

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        mq = self.genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=DEFAULT_SYSTEM_MESSAGE,
        )

        history = [self.change_role(message) for message in messages[1:-1]]
        chat_session = mq.start_chat(history=history)

        response = chat_session.send_message(messages[-1]["content"])

        return response.text.strip()


class AzureOpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        from openai import AzureOpenAI

        def _is_missing(value: Optional[str]) -> bool:
            return not value or value.upper() in {"EMPTY", "NONE", "NULL"}

        azure_endpoint = kwargs.pop("azure_endpoint", None)
        azure_api_version = kwargs.pop("azure_api_version", None) or kwargs.pop("api_version", None)
        azure_deployment = kwargs.pop("azure_deployment", None) or kwargs.pop("deployment_name", None)

        if _is_missing(api_key):
            raise ValueError("Missing Azure OpenAI API key. Set `models[].api_key` in config.")
        if _is_missing(azure_endpoint):
            raise ValueError("Missing Azure endpoint. Set `models[].provider_kwargs.azure_endpoint` in config.")
        if _is_missing(azure_api_version):
            raise ValueError("Missing Azure API version. Set `models[].provider_kwargs.azure_api_version` in config.")
        if _is_missing(azure_deployment):
            raise ValueError("Missing Azure deployment. Set `models[].provider_kwargs.azure_deployment` in config.")

        self.azure_deployment = azure_deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            **kwargs,
        )

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=messages,
            max_completion_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key, **kwargs)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic  # type: ignore

        self.client = Anthropic(api_key=api_key)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        response = self.client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system=messages[0]["content"],
            messages=messages[1:],
        )

        return response.content[0].text.strip()


def model_factory(provider: ProviderEnum, api_key: str, **kwargs) -> BaseProvider:
    match provider:
        case ProviderEnum.OPENAI:
            return OpenAIProvider(api_key=api_key, **kwargs)
        case ProviderEnum.ANTHROPIC:
            return AnthropicProvider(api_key=api_key)
        case ProviderEnum.GOOGLE:
            return GoogleProvider(api_key=api_key)
        case ProviderEnum.AZURE_OPENAI:
            return AzureOpenAIProvider(api_key=api_key, **kwargs)
        case ProviderEnum.VLLM:
            model = kwargs.pop("model")
            return VLLMProvider(model=model, **kwargs)
        case _:
            raise ValueError(f"Provider {provider} not supported.")


class Provider(BaseProvider):
    def __init__(self, api_key: str, provider: ProviderEnum, **kwargs):
        self.provider = model_factory(provider, api_key, **kwargs)

    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionResponse:
        return self.provider.complete(model, messages, temperature, max_tokens)


def configure_payload(
    topic: str,
    question: str,
    options: List[Dict[str, str]],
    answer: str,
    fast: bool = False,
    system_message: str | None = None,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> Sample:
    def format_options(options: List[Dict[str, str]]) -> str:
        formatted_options = "\n".join([f"{list(item.keys())[0]}) {list(item.values())[0]}" for item in options])  # noqa: RUF015
        keys = "".join([list(item.keys())[0] for item in options])  # noqa: RUF015

        return formatted_options, keys

    USER_QUERY_TEMPLATE = QUERY_TEMPLATE_MULTICHOICE_FAST if fast else QUERY_TEMPLATE_MULTICHOICE

    options_str, merged_letters = format_options(options)
    messages = [
        {
            "role": "system",
            "content": system_message or DEFAULT_SYSTEM_MESSAGE,
        },
    ]

    if few_shots:
        for shot in few_shots:
            shot_options, shot_merged_letters = format_options(shot["options"])
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": USER_QUERY_TEMPLATE.format(
                            topic=shot["category"],
                            question=shot["question"],
                            options=shot_options,
                            merged_letters=shot_merged_letters,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": shot["answer"] if fast else f"Risposta: {shot['answer']}",
                    },
                ]
            )

    messages.append(
        {
            "role": "user",
            "content": USER_QUERY_TEMPLATE.format(
                topic=topic,
                question=question,
                options=options_str,
                merged_letters=merged_letters,
            ),
        }
    )

    return Sample(messages=messages, answer=answer)


def _should_retry_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        return True
    if status_code in {408, 409, 429}:
        return True
    return status_code >= 500


@tenacity.retry(
    retry=tenacity.retry_if_exception(_should_retry_exception),
    wait=tenacity.wait_exponential(multiplier=1, max=5),
    stop=tenacity.stop_after_attempt(3),
)
def process_request(
    request: ChatCompletionRequest,
    client: BaseProvider,
    rate_limiter: Optional[RateLimiter] = None,
) -> ChatCompletionResponse:
    if rate_limiter:
        rate_limiter.throttle_requests()
    completion = client.complete(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    return ChatCompletionResponse(**request.dict(), output=completion)


def load_few_shots(file_path: str) -> List[Dict[str, Any]]:
    few_shots = pd.read_json(file_path, lines=True)
    return few_shots.to_dict(orient="records")


def extract_answer_fast(output: str) -> str:
    # truncate at the first USER: string to avoid hallucinations
    output = re.split(r"\n?USER:", output)[0]

    # First try: ASSISTANT: X
    match = re.search(r"ASSISTANT:\s*([A-Z])", output)
    if match:
        return match.group(1)

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    min_index = min([output.find(letter) for letter in LETTERS if letter in output], default=-1)
    if min_index == -1:
        return ""
    return output[min_index]


def extract_answer(output: str) -> str:
    # truncate at the first USER: string to avoid hallucinations
    output = re.split(r"\n?USER:", output)[0]

    def _find(pattern: str, text: str, ignore_case: bool = True) -> str:
        flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
        match = re.search(pattern, text, flags)
        if match:
            answer = re.sub(r"[è:)(?+-,;.]", "", match.group(1)).strip()
            answer = re.sub(r"^(?:sarà\s+la\s+|la\s+)?", "", answer).strip()
            return extract_answer_fast(answer) if answer else ""
        return ""

    def_pattern = r"Risposta:\s*(.*?)\s*(?=\n[A-Z]\)|\Z)"

    fallback_patterns = [
        r"quindi, la risposta è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*(?:corretta|giusta|appropriata|esatta|migliore|ottimale|finale|definitiva)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*più\s*(?:corretta|appropriata)\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:soluzione|opzione|scelta|alternativa)\s*(?:corretta)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:quindi|in\s*conclusione,?)?\s*(?:la\s*)?risposta\s*è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:la\s*)?(?:risposta|opzione|scelta)\s*(?:corretta|giusta|esatta)\s*è\s*(?:la\s*)?(?:lettera\s*)?([A-Z])",
    ]

    answer = _find(def_pattern, output, ignore_case=False)
    if answer:
        return answer

    if "nessuna delle opzioni" in output.lower():
        return ""

    for pattern in fallback_patterns:
        answer = _find(pattern, output)
        if answer:
            return answer

    # Last resort try: ASSISTANT: X (only if X is basically alone or at the very end/beginning)
    match = re.search(r"ASSISTANT:\s*([A-Z])\b", output)
    if match:
        return match.group(1)

    return ""


def _compute_stat(values: list, stat: str):
    stat_functions = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }
    if stat not in stat_functions:
        raise ValueError(f"Unknown {stat =}")
    return stat_functions[stat](values)


def _evaluate_response(response: ChatCompletionResponse) -> Tuple[str, bool]:
    parsed_answer = extract_answer_fast(response.output) if response.fast else extract_answer(response.output)
    is_correct = parsed_answer == response.answer
    return parsed_answer, is_correct


def aggregate_results(
    responses: List[ChatCompletionResponse],
    default_stats: Tuple[str, ...] = ("mean", "std"),
    name2stats: Dict[str, Tuple[str]] | None = None,
) -> Dict[str, float]:
    """Aggregate results from multiple evaluations into a single result dictionary.
    Similar to OpenAI simple_eval.

    Args:
        responses (List[ChatCompletionResponse]): List of responses to aggregate.
        default_stats (Tuple[str, ...], optional): Default statistics to compute for each metric. Defaults to ("mean", "std").
        name2stats (Dict[str, Tuple[str]], optional): Optional mapping from metric name to specific statistics to compute. If not provided, default_stats will be used for all metrics. Defaults to None.

    Returns:
        Dict[str, float]: A dictionary containing the aggregated metrics.
    """
    name2stats = name2stats or {}
    name2values = defaultdict(list)

    for resp in responses:
        _, correct = _evaluate_response(resp)
        name2values["accuracy"].append(float(correct))

    final_metrics = {}
    for name, values in name2values.items():
        stats = name2stats.get(name, default_stats)
        for stat in stats:
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = _compute_stat(values, stat)

    return final_metrics


def save_intermediate_results(responses: List[ChatCompletionResponse], output_file: Path):
    results = []
    for resp in sorted(responses, key=lambda x: x.index):
        parsed_answer, is_correct = _evaluate_response(resp)
        result = resp.dict()
        result["parsed_answer"] = parsed_answer
        result["is_correct"] = is_correct
        results.append(result)

    with Path.open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def load_intermediate_results(
    output_file: Path,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    if not output_file.exists():
        logger.info("Checkpoint file not found, starting from scratch.")
        return [], set()

    logger.info(f"Loading checkpoint file {output_file}")  # noqa: G004
    with Path.open(output_file, "r") as f:
        results = json.load(f)
        results = [ChatCompletionResponse(**r) for r in results]

    ids = {r.index for r in results}
    return results, ids


def process(requests: List[ChatCompletionRequest], client: Provider, config: DictConfig):
    if config.limit:
        requests = requests[: config.limit]

    if not config.data.output_dir:
        raise ValueError("config.data.output_dir must be set")

    model_name = Path(config.model).name

    output_dir = Path(config.data.output_dir) / model_name / "ITALIC"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n\nmodel: {config.model}\nthreads: {config.num_threads}\nfile: {config.data.data_file}\n")  # noqa: G004

    all_responses = []

    # -------------------------
    # FAST PATH FOR VLLM
    # -------------------------
    if config.provider == "vllm":
        logger.info("Using vLLM batched inference")

        outputs = client.provider.batch_complete(
            requests,
            config.temperature,
            config.max_tokens,
        )

        for req, text in zip(requests, outputs):
            resp = ChatCompletionResponse(**req.dict(), output=text)
            all_responses.append(resp)

    # -------------------------
    # NORMAL PATH (API MODELS)
    # -------------------------
    else:
        rate_limiter = None
        if config.rate_limiting.enabled:
            rate_limiter = RateLimiter(config.rate_limiting.requests_per_minute)

        pbar = tqdm(total=len(requests), desc="Processing responses")

        with ThreadPoolExecutor(max_workers=min(config.num_threads, len(requests))) as executor:
            futures = [executor.submit(process_request, req, client, rate_limiter) for req in requests]

            for future in as_completed(futures):
                resp = future.result()
                all_responses.append(resp)
                pbar.update(1)

    metrics = aggregate_results(all_responses)

    logger.info(f"Metrics: {metrics}")  # noqa: G004

    results = []
    for resp in sorted(all_responses, key=lambda x: x.index):
        parsed_answer, is_correct = _evaluate_response(resp)
        result = resp.dict()
        result["parsed_answer"] = parsed_answer
        result["is_correct"] = is_correct
        results.append(result)

    num_few_shots = (
        config.data.get("num_few_shots", 0) if hasattr(config.data, "get") else getattr(config.data, "num_few_shots", 0)
    )

    payload = {"metrics": metrics, "num_few_shots": num_few_shots, "results": results}

    save_evaluation_results(payload, str(output_dir))

    logger.info(f"Results saved to {output_dir}")  # noqa: G004


def _resolve_input_file(path_value: str) -> Path:
    raw_path = Path(path_value).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [(Path(__file__).resolve().parent / raw_path).resolve(), (Path.cwd() / raw_path).resolve()]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"File {path_value} does not exist. Checked: {checked_paths}")


def load_requests(config: DictConfig) -> List[ChatCompletionRequest]:
    data_file = _resolve_input_file(config.data.data_file)
    df_config = pd.read_json(data_file, lines=True)
    data = df_config.to_dict(orient="records")

    few_shots = (
        load_few_shots(str(_resolve_input_file(config.data.few_shot_file))) if config.data.few_shot_file else None
    )

    num_few_shots = (
        config.data.get("num_few_shots", 0) if hasattr(config.data, "get") else getattr(config.data, "num_few_shots", 0)
    )
    if num_few_shots > 0 and few_shots:
        num_few_shots = min(num_few_shots, 5)
        few_shots = few_shots[:num_few_shots]
    else:
        few_shots = None

    requests = []
    for i, item in tqdm(enumerate(data), total=len(data), desc="Preparing requests"):
        sample = configure_payload(
            topic=item["category"],
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            fast=config.fast,
            system_message=config.system_message,
            few_shots=few_shots,
        )
        requests.append(
            ChatCompletionRequest(
                index=i,
                provider=ProviderEnum(config.provider),
                model=config.model,
                messages=sample.messages,
                answer=sample.answer,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                fast=config.fast,
            )
        )
    return requests


class VLLMProvider(BaseProvider):
    def __init__(self, model: str, **kwargs):
        from vllm import LLM

        self.model_name = model
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            **kwargs,
        )

    # This is required by the abstract base class
    def complete(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Wrap single request in batch_complete
        from types import SimpleNamespace

        dummy_req = SimpleNamespace(messages=messages)  # noqa: F841

        # batch_complete expects a list of ChatCompletionRequest objects
        # we can wrap a minimal object for compatibility
        class DummyRequest:
            def __init__(self, messages):
                self.messages = messages

        out = self.batch_complete([DummyRequest(messages)], temperature, max_tokens)
        return out[0]

    def batch_complete(
        self,
        requests: list,
        temperature: float,
        max_tokens: int,
    ) -> list:
        from vllm import SamplingParams

        prompts = []
        for req in requests:
            prompt = ""
            for m in req.messages:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        return [o.outputs[0].text.strip() for o in outputs]


@hydra.main(version_base=None, config_path="./eval_config", config_name="config")
def run(config: DictConfig):
    register_enggpt2moe()

    def _is_missing(value: Optional[str]) -> bool:
        return not value or value.upper() in {"EMPTY", "NONE", "NULL"}

    def _sanitize_provider_kwargs(provider_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        clean_kwargs = dict(kwargs)
        if provider_name in {ProviderEnum.OPENAI.value, ProviderEnum.AZURE_OPENAI.value}:
            clean_kwargs.pop("max_model_len", None)
        return clean_kwargs

    for model_cfg in config.models:
        try:
            provider_str = model_cfg["provider"]
            model_name = model_cfg["model"]
            api_key = model_cfg.get("api_key", "")
            print(f"Found model config: provider={provider_str}, model={model_name}, api_key={api_key}")  # noqa: T201
            requires_api_key = provider_str in {
                ProviderEnum.OPENAI.value,
                ProviderEnum.AZURE_OPENAI.value,
                ProviderEnum.ANTHROPIC.value,
                ProviderEnum.GOOGLE.value,
            }
            if requires_api_key and _is_missing(api_key):
                raise ValueError(
                    f"Missing api_key for provider={provider_str}. Set `models[].api_key` in config (you can use Hydra env interpolation)."
                )

            provider_kwargs = _sanitize_provider_kwargs(provider_str, dict(model_cfg.get("provider_kwargs", {})))
            if provider_str == "vllm":
                provider_kwargs["model"] = model_name

            with open_dict(config):
                config.provider = provider_str
                config.model = model_name
                config.api_key = api_key

            logger.info(f"<bold>🔎 Running evaluation: {provider_str}:{model_name} 🔎</bold>")  # noqa: G004

            client = Provider(
                api_key=config.api_key,
                provider=ProviderEnum(config.provider),
                **provider_kwargs,
            )

            requests = load_requests(config)
            process(requests=requests, client=client, config=config)

        except Exception:
            logger.error(f"Error while evaluating {model_cfg}. Skipping to the next one.")  # noqa: G004
            logger.exception("Exception details:")

        finally:
            # ---- free GPU memory ----
            if provider_str == "vllm":
                time.sleep(30)
                del client.provider.llm
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(30)
                logger.info(f"Cleared GPU memory for model {model_name}")  # noqa: G004


if __name__ == "__main__":
    run()
