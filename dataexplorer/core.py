from __future__ import annotations

import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from io import StringIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


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

# Assumes a standard equities-style annualization factor.
TRADING_DAYS_PER_YEAR = 252
# Matches filename characters to replace: anything not alphanumeric, dot, underscore, or hyphen.
SAFE_FILENAME_PATTERN = r"[^A-Za-z0-9._-]"


def sanitize_script_text(text: str) -> str:
    """Remove non-ASCII characters from script text."""
    return "".join(char for char in text if ord(char) < 128)


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
        "Always return a complete, runnable script.\n"
        "Do not return only a diff, patch, fragment, or just the newly added lines.\n"
        "Keep all existing behavior unless the user explicitly asks to change or remove it.\n"
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
            return text[match.end() : closing].strip("\r\n")
    return text.strip("\r\n")


def _execute_llm_command(*, prompt: str, model: str, command: str, timeout: int) -> str:
    """Execute local LLM command and return stripped stdout, or raise a readable error."""
    normalized_command = command.strip()
    normalized_model = model.strip()

    if not normalized_command:
        raise RuntimeError("LLM command cannot be empty.")
    if not normalized_model:
        raise RuntimeError("LLM model cannot be empty.")

    process = subprocess.run(
        [normalized_command, "run", normalized_model, prompt],
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


def list_ollama_models(*, command: str = "ollama", timeout: int = 20) -> list[str]:
    """Return available model names from ``ollama list`` output."""
    normalized_command = command.strip()
    if not normalized_command:
        raise RuntimeError('LLM command cannot be empty. Please provide a valid command (e.g., "ollama").')

    process = subprocess.run(
        [normalized_command, "list"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip() or "No stderr output."
        raise RuntimeError(f"Could not list models: {stderr}")

    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        return []

    models: list[str] = []
    for index, line in enumerate(lines):
        if index == 0 and line.lower().startswith("name"):
            continue
        parts = line.split()
        if not parts:
            continue
        model_name = parts[0]
        if model_name:
            models.append(model_name)
    return models


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
    return textwrap.dedent(extract_python_code(output))


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


def download_eodhd_csv(
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    output_path: str | None = None,
    api_key_env: str = "EODHD_API_KEY",
    timeout: int = 30,
) -> str:
    """Download EODHD CSV data and return the written absolute CSV path."""
    normalized_symbol = symbol.strip()
    normalized_timeframe = timeframe.strip().lower()
    normalized_start = start_date.strip()
    normalized_end = end_date.strip()
    if not normalized_symbol:
        raise ValueError("EODHD symbol cannot be empty.")
    if not normalized_timeframe:
        raise ValueError("EODHD timeframe cannot be empty.")
    if not normalized_start or not normalized_end:
        raise ValueError("EODHD start and end dates are required.")

    try:
        start_value = date.fromisoformat(normalized_start)
        end_value = date.fromisoformat(normalized_end)
    except ValueError as error:
        raise ValueError("Dates must use YYYY-MM-DD format.") from error
    if start_value > end_value:
        raise ValueError("Start date must be less than or equal to end date.")

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing EODHD API key in environment variable: {api_key_env}")

    if normalized_timeframe in {"d", "1d", "day", "daily"}:
        endpoint = f"https://eodhd.com/api/eod/{urllib_parse.quote(normalized_symbol)}"
        query_params = {
            "api_token": api_key,
            "fmt": "csv",
            "period": "d",
            "from": normalized_start,
            "to": normalized_end,
        }
    else:
        endpoint = f"https://eodhd.com/api/intraday/{urllib_parse.quote(normalized_symbol)}"
        query_params = {
            "api_token": api_key,
            "fmt": "csv",
            "interval": normalized_timeframe,
            "from": normalized_start,
            "to": normalized_end,
        }

    url = f"{endpoint}?{urllib_parse.urlencode(query_params)}"
    request = urllib_request.Request(url, headers={"User-Agent": "dataexplorer/1.0"})
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib_error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"EODHD request failed ({error.code}): {detail or error.reason}") from error
    except urllib_error.URLError as error:
        raise RuntimeError(f"EODHD request failed: {error.reason}") from error

    if not payload:
        raise RuntimeError("EODHD returned an empty response.")

    if output_path:
        target = Path(output_path).expanduser().resolve()
    else:
        safe_symbol = re.sub(SAFE_FILENAME_PATTERN, "_", normalized_symbol)
        safe_tf = re.sub(SAFE_FILENAME_PATTERN, "_", normalized_timeframe)
        file_name = f"{safe_symbol}_{safe_tf}_{normalized_start}_{normalized_end}.csv"
        target = Path(file_name).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return str(target)


def list_available_csv_files(base_path: str = ".") -> list[str]:
    """Return sorted relative CSV file paths under ``base_path``."""
    root = Path(base_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files: list[str] = []
    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [name for name in dirs if not name.startswith(".") and name != "__pycache__"]
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                full_path = Path(current_root) / filename
                files.append(str(full_path.relative_to(root)))
    return sorted(files)


def save_session_file(session_path: str, payload: dict[str, object]) -> str:
    """Persist session payload to JSON and return absolute file path."""
    target = Path(session_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target)


def load_session_file(session_path: str) -> dict[str, object]:
    """Load and parse a saved session JSON payload."""
    source = Path(session_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Session file not found: {source}")
    content = source.read_text(encoding="utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise RuntimeError("Session file is invalid: expected a JSON object.")
    return payload


def export_session_html(session_path: str, payload: dict[str, object]) -> str:
    """Export session payload to a standalone HTML document."""
    from html import escape

    output_path = Path(session_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = escape(str(payload.get("title", "Data Explorer Session")))
    csv_path = escape(str(payload.get("csv_path", "")))
    model = escape(str(payload.get("model", "")))
    llm_command = escape(str(payload.get("llm_command", "")))
    code = escape(str(payload.get("code", "")))

    events_html: list[str] = []
    output_events = payload.get("output_events", [])
    if isinstance(output_events, list):
        for event in output_events:
            if not isinstance(event, dict):
                continue
            kind = escape(str(event.get("kind", "system")))
            text = escape(str(event.get("text", "")))
            events_html.append(
                "<div class='event'>"
                f"<div class='event-kind'>{kind}</div>"
                f"<pre>{text}</pre>"
                "</div>"
            )

    images_html: list[str] = []
    generated_plots = payload.get("generated_plots", [])
    if isinstance(generated_plots, list):
        for image_path in generated_plots:
            if not isinstance(image_path, str):
                continue
            uri = _encode_image_data_uri(image_path)
            if not uri:
                continue
            images_html.append(
                "<figure>"
                f"<img src='{uri}' alt='Generated plot' />"
                f"<figcaption>{escape(image_path)}</figcaption>"
                "</figure>"
            )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.4; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    .meta dt {{ font-weight: 600; }}
    .meta dd {{ margin: 0 0 0.5rem 0; }}
    pre {{ white-space: pre-wrap; background: #f5f5f5; padding: 0.75rem; border-radius: 6px; }}
    .event {{ border: 1px solid #ddd; border-radius: 6px; margin-bottom: 0.75rem; }}
    .event-kind {{ padding: 0.4rem 0.75rem; background: #fafafa; border-bottom: 1px solid #eee; font-size: 0.85rem; color: #555; text-transform: uppercase; }}
    .event pre {{ margin: 0; border-radius: 0 0 6px 6px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
    figure {{ margin: 0 0 1rem 0; }}
    figcaption {{ font-size: 0.85rem; color: #555; margin-top: 0.4rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <h2>Session Settings</h2>
  <dl class="meta">
    <dt>CSV path</dt><dd>{csv_path}</dd>
    <dt>Model</dt><dd>{model}</dd>
    <dt>LLM command</dt><dd>{llm_command}</dd>
  </dl>
  <h2>Code</h2>
  <pre>{code}</pre>
  <h2>Chat / Output History</h2>
  {''.join(events_html) or '<p>No history captured.</p>'}
  <h2>Generated Graphs</h2>
  {''.join(images_html) or '<p>No generated graphs captured.</p>'}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def _encode_image_data_uri(path: str) -> str:
    """Encode existing image file to a data URI, or return empty string."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        return ""
    suffix = candidate.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        mime = ""
    if not mime:
        return ""
    encoded = base64.b64encode(candidate.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def run_signal_trading_simulation(
    code: str,
    data_path: str,
    *,
    price_column: str = "close",
    signal_column: str = "signal",
) -> str:
    """Run session code, then simulate trading using next-row execution from ``signal``.

    Risk metrics are annualized with a 252 trading-days factor.
    """
    stdout = StringIO()
    stderr = StringIO()
    namespace: dict[str, object] = {
        "__builtins__": SAFE_BUILTINS,
        "data_path": str(Path(data_path)),
    }
    pandas_import_error: Exception | None = None
    pandas_module = None
    try:
        import pandas as pd  # type: ignore

        namespace["pd"] = pd
        pandas_module = pd
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

    if pandas_module is None:
        return "Error: pandas is required to run /ts simulation."

    df_obj = namespace.get("df")
    if not isinstance(df_obj, pandas_module.DataFrame):
        return "Error: /ts requires the active script to define a pandas DataFrame named 'df'."

    missing_columns = [name for name in (price_column, signal_column) if name not in df_obj.columns]
    if missing_columns:
        return "Error: /ts missing required column(s): " + ", ".join(missing_columns)

    frame = df_obj[[price_column, signal_column]].copy()
    frame[price_column] = pandas_module.to_numeric(frame[price_column], errors="coerce")
    frame[signal_column] = pandas_module.to_numeric(frame[signal_column], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=[price_column]).reset_index(drop=True)
    if len(frame) < 2:
        return "Error: /ts requires at least 2 valid price rows."

    signal = frame[signal_column].apply(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
    effective_signal = signal.shift(1).fillna(0.0)
    price_diff = frame[price_column].diff().fillna(0.0)
    step_pnl = effective_signal * price_diff

    equity = step_pnl.cumsum()
    mean_step_pnl = float(step_pnl.mean())
    step_std = float(step_pnl.std(ddof=0))
    sharpe = 0.0 if step_std == 0 else mean_step_pnl / step_std * (TRADING_DAYS_PER_YEAR ** 0.5)

    downside = step_pnl[step_pnl < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
    sortino = 0.0 if downside_std == 0 else mean_step_pnl / downside_std * (TRADING_DAYS_PER_YEAR ** 0.5)

    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    trades = int(((effective_signal != effective_signal.shift(1).fillna(0.0)) & (effective_signal != 0.0)).sum())

    summary_lines = [
        f"Trading simulation ({signal_column} executes on next {price_column} row)",
        f"Rows: {len(frame)}",
        f"Trades: {trades}",
        f"Total PnL: {float(step_pnl.sum()):.6f}",
        f"Sharpe: {sharpe:.6f}",
        f"Sortino: {sortino:.6f}",
        f"Max Drawdown: {max_drawdown:.6f}",
    ]
    prior_output = stdout.getvalue().strip()
    if prior_output:
        return f"{prior_output}\n" + "\n".join(summary_lines)
    return "\n".join(summary_lines)


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
