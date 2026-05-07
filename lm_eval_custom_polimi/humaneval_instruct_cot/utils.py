import re

import evaluate as hf_evaluate

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions_instruct_reason(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    INVALID_TOKEN_REPLACEMENTS = {
        "\u2581": " ",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u279e": "->",
        "\u2192": "->",
    }

    TRANSLATION_TABLE = str.maketrans(INVALID_TOKEN_REPLACEMENTS)

    def sanitize_python_text(text: str) -> str:
        return text.translate(TRANSLATION_TABLE)

    def normalize_indentation(text: str) -> str:
        return re.sub(
            r"(?m)^(?P<prefix>[ \t]*)(?P<spm>\u2581+)",
            lambda m: m.group("prefix") + (" " * len(m.group("spm"))),
            text,
        )

    def strip_doctest_prompts(text: str) -> str:
        return re.sub(r"(?m)^(?P<indent>[ \t]*)>>>\s?", r"\g<indent>", text)

    def strip_markdown_fences(text: str) -> str:
        text = text.strip()
        fenced = re.search(r"```\s*(?:python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        if text.startswith("```"):
            text = re.sub(r"^```\s*(?:python)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()

    def extract_code(r: str) -> str:
        code = strip_markdown_fences(r)
        code = sanitize_python_text(code)
        code = normalize_indentation(code)
        code = strip_doctest_prompts(code)
        return code.strip()

    return [[extract_code(r) for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")]) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
