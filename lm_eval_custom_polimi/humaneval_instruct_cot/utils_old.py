import evaluate as hf_evaluate
import re

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
    def normalize_indentation(text: str) -> str:
        return re.sub(
            r"(?m)^(?P<prefix>[ \t]*)(?P<spm>\u2581+)",
            lambda m: m.group("prefix") + (" " * len(m.group("spm"))),
            text,
        )

    def extract_code(r: str) -> str:
        # Match ```python ... ``` oppure ``` ... ```
        m = re.search(r"```(?:python)?\n(.*?)```", r, re.DOTALL)
        if m:
            return normalize_indentation(m.group(1).strip())
        return normalize_indentation(r.strip())

    return [[extract_code(r) for r in resp] for resp, doc in zip(resps, docs)]

def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
