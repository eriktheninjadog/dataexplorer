from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import re
import subprocess


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
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
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


def run_user_code(code: str, data_path: str) -> str:
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
    if pandas_import_error and ("pd" in code or "pandas" in code):
        return "Error: pandas is required to execute this script but is not installed."
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(code, "<session_code>", "exec"), namespace, namespace)
    except Exception as error:
        if "__import__ not found" in str(error):
            return (
                "Error: import statements are disabled in session code. "
                "Use the preloaded 'pd' pandas object instead."
            )
        err_text = stderr.getvalue().strip()
        if err_text:
            return f"{stdout.getvalue()}\n{err_text}\n{error}".strip()
        return f"{stdout.getvalue()}\nError: {error}".strip()
    err_text = stderr.getvalue().strip()
    if err_text:
        return f"{stdout.getvalue()}\n{err_text}".strip()
    return stdout.getvalue().strip() or "Script ran successfully with no output."
