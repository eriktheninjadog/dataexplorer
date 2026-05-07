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
    return (
        "You are helping with a local, interactive pandas time-series explorer.\n"
        "Return only valid Python code (no markdown fences) that can be executed directly.\n"
        "Use pandas (already available as 'pd') and keep prints concise and useful for iterative analysis.\n"
        f"The CSV path to use is: {data_path}\n\n"
        "Current code:\n"
        f"{current_code}\n\n"
        "User request:\n"
        f"{user_request}\n"
    )


def extract_python_code(text: str) -> str:
    match = re.search(r"```python", text, flags=re.IGNORECASE)
    if match:
        closing = text.find("```", match.end())
        if closing != -1:
            return text[match.end() : closing].strip()
    return text.strip()


def request_llm_update(
    user_request: str,
    current_code: str,
    data_path: str,
    *,
    model: str = "llama3.1",
    command: str = "ollama",
    timeout: int = 60,
) -> str:
    if not command.strip():
        raise RuntimeError("LLM command cannot be empty.")
    if not model.strip():
        raise RuntimeError("LLM model cannot be empty.")
    prompt = build_prompt(user_request=user_request, current_code=current_code, data_path=data_path)
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
    return extract_python_code(output)


def build_chat_prompt(user_message: str, data_path: str, data_summary: str = "") -> str:
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
    if not command.strip():
        raise RuntimeError("LLM command cannot be empty.")
    if not model.strip():
        raise RuntimeError("LLM model cannot be empty.")
    prompt = build_chat_prompt(
        user_message=user_message, data_path=data_path, data_summary=data_summary
    )
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

    Returns (text_output, list_of_figure_paths).
    """
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


def open_figure(path: str) -> None:
    """Open a saved figure file with the system default image viewer."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


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
