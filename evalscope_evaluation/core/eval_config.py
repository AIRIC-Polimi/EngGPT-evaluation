import os
from dataclasses import dataclass

from .generation_config import GenerationConfig


@dataclass
class EvalConfiguration:
    generation_config: GenerationConfig
    batch_size: int = 32
    is_fc_model: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    chat_template: str | None = None
    skip_vllm_serve: bool = False
    skip_eval: bool = False
    cloud_model_url: str | None = None
    cloud_model_api_key: str | None = None
    # The model refuses to use a dot ('.') in function names; set this to automatically convert dots to
    # underscores during evaluation. NOTE: this is particularly important for OpenAI models, which are trained to avoid
    # producing dots in function names (and produce underscores instead).
    underscore_to_dots: bool = False


# For reference about with tool call parser to use for which model, see https://docs.vllm.ai/en/v0.10.2/features/tool_calling.html#none-function-calling
# For reference about with reasoning parser to use for which model, see https://docs.vllm.ai/en/v0.10.2/features/reasoning_outputs.html
ALL_EVALS_CONFIG = [
    # EngGPT2Moe model served externally (requires custom Dockerfile), in native function calling mode
    (
        "engineering-group/EngGPT2-16B-A3B",
        EvalConfiguration(
            generation_config=GenerationConfig(max_tokens=35000),
            batch_size=16,
            is_fc_model=True,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            skip_vllm_serve=True,
            skip_eval=True,
            underscore_to_dots=True,
        ),
    ),
    (
        "Qwen/Qwen3-4B",
        EvalConfiguration(
            generation_config=GenerationConfig(max_tokens=35000),
            batch_size=32,
            is_fc_model=True,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",  # NOTE: for first eval run was using deepseek_r1
            skip_eval=True,
            underscore_to_dots=True,
        ),
    ),
    (
        "Qwen/Qwen3-8B",
        EvalConfiguration(
            generation_config=GenerationConfig(max_tokens=35000),
            batch_size=32,
            is_fc_model=True,
            tool_call_parser="hermes",
            reasoning_parser="qwen3",
            skip_eval=True,
            underscore_to_dots=True,
        ),
    ),
    (
        "Almawave/Velvet-14B",
        EvalConfiguration(generation_config=GenerationConfig(), batch_size=16, is_fc_model=False, skip_eval=True),
    ),
    (
        "deepseek-ai/deepseek-moe-16b-chat",
        EvalConfiguration(generation_config=GenerationConfig(), batch_size=16, is_fc_model=False, skip_eval=True),
    ),
    (
        "sapienzanlp/Minerva-7B-instruct-v1.0",
        EvalConfiguration(generation_config=GenerationConfig(), batch_size=32, is_fc_model=False, skip_eval=True),
    ),
    (
        "Fastweb/FastwebMIIA-7B",
        EvalConfiguration(generation_config=GenerationConfig(), batch_size=32, is_fc_model=False, skip_eval=True),
    ),
    (
        "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
        EvalConfiguration(generation_config=GenerationConfig(), batch_size=32, is_fc_model=False, skip_eval=True),
    ),
    (
        "openai/gpt-oss-20b",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=16,
            is_fc_model=True,
            tool_call_parser="openai",
            skip_eval=True,
            underscore_to_dots=True,
        ),
    ),
    (
        "moonshotai/Moonlight-16B-A3B-Instruct",
        EvalConfiguration(generation_config=GenerationConfig(), batch_size=16, is_fc_model=False, skip_eval=True),
    ),
    (
        "meta-llama/Llama-3.1-8B-Instruct",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=32,
            is_fc_model=True,
            tool_call_parser="llama3_json",
            skip_eval=True,
            chat_template="evalscope_evaluation/vllm_chat_templates/tool_chat_template_llama3.1_json.jinja",
            underscore_to_dots=True,
        ),
    ),
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=64,
            is_fc_model=True,
            tool_call_parser="llama3_json",
            skip_eval=True,
            chat_template="evalscope_evaluation/vllm_chat_templates/tool_chat_template_llama3.2_json.jinja",
            underscore_to_dots=True,
        ),
    ),
    # pythonic parser with corresponding chat template
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=64,
            is_fc_model=True,
            tool_call_parser="pythonic",
            skip_eval=True,
            chat_template="evalscope_evaluation/vllm_chat_templates/tool_chat_template_llama3.2_pythonic.jinja",
            underscore_to_dots=True,
        ),
    ),
    # Llama models in prompt mode, which have been observed to produce better results in some tasks
    (
        "meta-llama/Llama-3.1-8B-Instruct",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=32,
            is_fc_model=False,
            skip_eval=True,
            underscore_to_dots=False,
        ),
    ),
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=64,
            is_fc_model=False,
            skip_eval=True,
            underscore_to_dots=False,
        ),
    ),
    (
        "google/gemma-3-4b-it",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=64,
            is_fc_model=False,
            skip_eval=True,
        ),
    ),
    (
        "google/gemma-3-12b-it",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=16,
            is_fc_model=False,
            skip_eval=True,
        ),
    ),
    # Gemma3 models with pythonic tool parser
    # NOTE: this does not seem to work properly
    (
        "google/gemma-3-4b-it",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=64,
            is_fc_model=True,
            tool_call_parser="pythonic",
            skip_eval=True,
            chat_template="evalscope_evaluation/vllm_chat_templates/tool_chat_template_gemma3_pythonic.jinja",
        ),
    ),
    (
        "google/gemma-3-12b-it",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=16,
            is_fc_model=True,
            tool_call_parser="pythonic",
            skip_eval=True,
            chat_template="evalscope_evaluation/vllm_chat_templates/tool_chat_template_gemma3_pythonic.jinja",
        ),
    ),
    (
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        EvalConfiguration(
            generation_config=GenerationConfig(),
            batch_size=32,
            is_fc_model=True,
            tool_call_parser="mistral",
            skip_eval=True,
            underscore_to_dots=True,
        ),
    ),
    (
        "gpt-5-nano",
        EvalConfiguration(
            # NOTE: this model does only support its default temperature of 1 (original message "Unsupported value:
            # 'temperature' does not support 0.01 with this model. Only the default (1) value is supported.")
            generation_config=GenerationConfig(temperature=1),
            batch_size=64,
            is_fc_model=True,
            skip_eval=True,
            skip_vllm_serve=True,
            cloud_model_url="https://swedencentral.api.cognitive.microsoft.com/openai/v1",
            cloud_model_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            underscore_to_dots=True,
        ),
    ),
    # gpt 5 nano in PROMPT mode
    (
        "gpt-5-nano",
        EvalConfiguration(
            # NOTE: this model does only support its default temperature of 1 (original message "Unsupported value:
            # 'temperature' does not support 0.01 with this model. Only the default (1) value is supported.")
            generation_config=GenerationConfig(temperature=1),
            batch_size=16,
            is_fc_model=False,
            skip_eval=True,
            skip_vllm_serve=True,
            cloud_model_url="https://swedencentral.api.cognitive.microsoft.com/openai/v1",
            cloud_model_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            underscore_to_dots=False,
        ),
    ),
]
