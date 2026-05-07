import json
from datetime import datetime
from pathlib import Path

from lm_eval.utils import handle_non_serializable


def _get_current_timestamp() -> str:
    """Get the current timestamp in the format YYYY-MM-DD_HH-MM-SS.

    Returns:
        str: current timestamp formatted as a string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_evaluation_results(
    results: dict,
    output_dir: str,
    filename_prefix: str = "evaluation_results",
    is_serializable: bool = True,
) -> None:
    """Save evaluation results to a JSON file with a timestamped filename.

    Args:
        results (dict): The evaluation results to save.
        output_dir (str): The directory where the results file will be saved.
        filename_prefix (str, optional): Prefix for the results filename. Defaults to "evaluation_results".
        is_serializable (bool, optional): Whether the results are JSON serializable. Defaults to True.
    """
    # Ensure the output directory exists
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create a timestamped filename
    timestamp = _get_current_timestamp()
    filename = f"{filename_prefix}_{timestamp}.json"
    file_path = Path(output_dir) / filename

    # Save results to the JSON file
    with file_path.open("w") as f:
        json.dump(results, f, default=handle_non_serializable if not is_serializable else None, indent=2)


def load_evaluation_results(file_path: str) -> dict | None:
    """Load evaluation results from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing evaluation results.

    Returns:
        dict | None: The loaded evaluation results, or None if loading fails.
    """
    if not Path(file_path).exists():
        return None
    with Path(file_path).open("r") as f:
        return json.load(f)
