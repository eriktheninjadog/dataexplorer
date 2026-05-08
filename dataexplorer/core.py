from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SAFE_BUILTINS = {
    "Exception": Exception,
    "NameError": NameError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "abs": abs,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


def default_script(data_path: str) -> str:
    """Create a starter script for time-series price-data exploration."""
    return (
        f"data_path = {data_path!r}\n"
        "df = pd.read_csv(data_path)\n"
        "print('Rows:', len(df))\n"
        "print('Columns:', list(df.columns))\n"
        "print('\\nHead:')\n"
        "print(df.head())\n"
        "if 'close' in df.columns:\n"
        "    print('\\nClose summary:')\n"
        "    print(df['close'].describe())\n"
    )


def build_prompt(user_request: str, current_code: str, data_path: str) -> str:
    """Build the code-generation prompt for the local LLM."""
    return (
        "You are helping with a local, interactive pandas time-series explorer.\n"
        "Return only valid Python code (no markdown fences) that can be executed directly.\n"
        "Use pandas (already available as 'pd') and keep prints concise and useful for iterative analysis.\n"
        "If the user asks for a trading simulation/backtest, generate code that:\n"
        "- creates any requested indicator columns (for example moving averages).\n"
        "- creates the requested signal column values (for example 1 and -1).\n"
        "- runs a strategy simulation on the requested price column (default to 'close' if not specified).\n"
        "- reports practical performance metrics, including at least pnl, Sharpe, and Sortino.\n"
        "- also include max drawdown and basic trade stats when possible.\n"
        f"The CSV path to use is: {data_path}\n\n"
        "Current code:\n"
        f"{current_code}\n\n"
        "User request:\n"
        f"{user_request}\n"
    )


def extract_python_code(text: str) -> str:
    """Extract Python from a fenced block when present, otherwise return raw text."""
    match = re.search(r"```python", text, flags=re.IGNORECASE)
    if match:
        closing = text.find("```", match.end())
        if closing != -1:
            return text[match.end() : closing].strip()
    return text.strip()


def _execute_llm_command(*, prompt: str, model: str, command: str, timeout: int) -> str:
    """Execute local LLM command and return stripped stdout, or raise a readable error."""
    if not command.strip():
        raise RuntimeError("LLM command cannot be empty.")
    if not model.strip():
        raise RuntimeError("LLM model cannot be empty.")

    process = subprocess.run(
        [command, "run", model, prompt],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip() or "No stderr output."
        raise RuntimeError(f"LLM command failed: {stderr}")

    output = process.stdout.strip()
    if not output:
        raise RuntimeError("LLM returned no output.")
    return output


def request_llm_update(
    user_request: str,
    current_code: str,
    data_path: str,
    *,
    model: str = "llama3.1",
    command: str = "ollama",
    timeout: int = 60,
) -> str:
    """Request updated session code from the local LLM and normalize to Python source."""
    prompt = build_prompt(user_request=user_request, current_code=current_code, data_path=data_path)
    output = _execute_llm_command(
        prompt=prompt,
        model=model,
        command=command,
        timeout=timeout,
    )
    return extract_python_code(output)


def build_chat_prompt(user_message: str, data_path: str, data_summary: str = "") -> str:
    """Build a non-code chat prompt for Q&A about the current dataset."""
    context = f"Data file: {data_path}\n"
    if data_summary:
        context += f"Data summary:\n{data_summary}\n"
    return (
        "You are a helpful data analyst assistant. The user is exploring a CSV dataset.\n"
        "Answer their questions about the data, provide insights, and suggest analyses.\n"
        "Do NOT write Python code unless specifically asked. Focus on data interpretation.\n"
        f"{context}\n"
        "User question:\n"
        f"{user_message}\n"
    )


def request_llm_chat(
    user_message: str,
    data_path: str,
    data_summary: str = "",
    *,
    model: str = "llama3.1",
    command: str = "ollama",
    timeout: int = 60,
) -> str:
    """Request a plain-language data analysis response from the local LLM."""
    prompt = build_chat_prompt(
        user_message=user_message, data_path=data_path, data_summary=data_summary
    )
    return _execute_llm_command(
        prompt=prompt,
        model=model,
        command=command,
        timeout=timeout,
    )


def get_data_summary(data_path: str) -> str:
    """Return a short textual summary of the CSV for LLM context."""
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(data_path)
        lines = [
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            f"Columns: {list(df.columns)}",
            "Head:",
            str(df.head()),
        ]
        return "\n".join(lines)
    except Exception as error:
        return f"(Could not load data summary: {error})"


def run_user_code_with_plots(code: str, data_path: str) -> tuple[str, list[str]]:
    """Like run_user_code but pre-injects matplotlib and captures plt.show() calls as PNG files.

    ``plt`` (matplotlib.pyplot with the Agg backend) is injected into the execution namespace so
    user code can call ``plt.plot(...)`` / ``plt.show()`` without import statements.
    Note: ``plt.show`` is replaced globally on the injected pyplot module for the duration of this
    call; this is a side effect of matplotlib's shared module state.

    Each ``plt.show()`` call saves the current figure to a temporary directory and closes all
    figures. The temporary directory is not automatically deleted so that callers can open the
    files with an external viewer; callers are responsible for cleanup if desired.

    Returns (text_output, list_of_figure_paths).
    """
    # Track all saved figures so the UI can report and open them after execution.
    figure_paths: list[str] = []
    stdout = StringIO()
    stderr = StringIO()
    namespace: dict[str, object] = {
        "__builtins__": SAFE_BUILTINS,
        "data_path": str(Path(data_path)),
    }

    pandas_import_error: Exception | None = None
    try:
        import pandas as pd  # type: ignore

        namespace["pd"] = pd
    except Exception as error:
        pandas_import_error = error
    if pandas_import_error and _references_pandas(code):
        return "Error: pandas is required to execute this script but is not installed.", []

    try:
        import matplotlib  # type: ignore

        # Force a non-interactive backend so plot export works in terminal environments.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        tmp_dir = tempfile.mkdtemp(prefix="dataexplorer_plots_")
        plot_counter = [0]

        def _show() -> None:
            plot_counter[0] += 1
            path = os.path.join(tmp_dir, f"plot_{plot_counter[0]}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close("all")
            figure_paths.append(path)

        plt.show = _show  # type: ignore[method-assign]
        namespace["plt"] = plt
        namespace["matplotlib"] = matplotlib
    except Exception:
        pass

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<session_code>", "exec"), namespace, namespace)
    except (NameError, ImportError) as error:
        if "__import__" in str(error):
            return (
                "Error: import statements are disabled in session code. "
                "Use the preloaded 'pd' pandas object instead.",
                figure_paths,
            )
        err_text = stderr.getvalue().strip()
        if err_text:
            return f"{stdout.getvalue()}\n{err_text}\n{error}".strip(), figure_paths
        return f"{stdout.getvalue()}\nError: {error}".strip(), figure_paths
    except Exception as error:
        err_text = stderr.getvalue().strip()
        if err_text:
            return f"{stdout.getvalue()}\n{err_text}\n{error}".strip(), figure_paths
        return f"{stdout.getvalue()}\nError: {error}".strip(), figure_paths
    err_text = stderr.getvalue().strip()
    if err_text:
        return f"{stdout.getvalue()}\n{err_text}".strip(), figure_paths
    return (stdout.getvalue().strip() or "No output produced."), figure_paths


def open_figure(path: str) -> bool:
    """Open a saved figure file with the system default image viewer.

    Returns True if the viewer was launched successfully, False otherwise.
    """
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False


def run_user_code(code: str, data_path: str) -> str:
    """Run session code with pandas preloaded as `pd` and restricted builtins.

    This limits imports but is not a hardened sandbox.
    """
    stdout = StringIO()
    stderr = StringIO()
    namespace: dict[str, object] = {
        "__builtins__": SAFE_BUILTINS,
        "data_path": str(Path(data_path)),
    }
    pandas_import_error: Exception | None = None
    try:
        import pandas as pd  # type: ignore

        namespace["pd"] = pd
    except Exception as error:
        pandas_import_error = error
    if pandas_import_error and _references_pandas(code):
        return "Error: pandas is required to execute this script but is not installed."
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<session_code>", "exec"), namespace, namespace)
    except (NameError, ImportError) as error:
        if "__import__" in str(error):
            return (
                "Error: import statements are disabled in session code. "
                "Use the preloaded 'pd' pandas object instead."
            )
        err_text = stderr.getvalue().strip()
        if err_text:
            return f"{stdout.getvalue()}\n{err_text}\n{error}".strip()
        return f"{stdout.getvalue()}\nError: {error}".strip()
    except Exception as error:
        err_text = stderr.getvalue().strip()
        if err_text:
            return f"{stdout.getvalue()}\n{err_text}\n{error}".strip()
        return f"{stdout.getvalue()}\nError: {error}".strip()
    err_text = stderr.getvalue().strip()
    if err_text:
        return f"{stdout.getvalue()}\n{err_text}".strip()
    return stdout.getvalue().strip() or "No output produced."


def _references_pandas(code: str) -> bool:
    """Return True when code references pandas directly or through imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"pd", "pandas"}:
            return True
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "pandas" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "pandas":
                return True
    return False
