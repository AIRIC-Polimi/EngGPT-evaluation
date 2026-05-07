from dataclasses import dataclass


@dataclass
class GenerationConfig:
    max_tokens: int = 4096
    top_p: float | None = None
    top_k: int | None = None
    temperature: float = 0
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int = 42
    do_sample: bool = False
    max_tool_output: int = 16 * 1024  # default of the evalscope library
