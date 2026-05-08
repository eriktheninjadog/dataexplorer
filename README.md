# dataexplorer

A pandas-driven Textual app for interactive time-series exploration with local LLM help, command shortcuts, and CSV workflows.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Start the app

```bash
python main.py --csv prices.csv --model llama3.1
```

Optional CLI flags:

- `--csv` default CSV path shown in the app
- `--model` default local model
- `--llm-command` command used for local LLM calls (default: `ollama`)

## App layout

- **CSV path**: the file used by script execution and chat context.
- **Model** and **LLM command**: editable at runtime.
- **Prompt input**: used for both natural language prompts and slash commands.
- **Code editor**: live Python script.
- **Output panel**: logs, command responses, and execution output.

## Main actions

- **Ask LLM**: asks your local model to update script code.
- **Run Code**: executes current script against current CSV path.
- **Chat Mode**: toggles analyst Q&A mode (no code generation by default).
- **Refresh Models**: runs `ollama list` and shows available model names.
- **Save Session / Load Session**: persists and restores app state.
- **Export HTML**: exports a standalone session report with output/code/plots.

## Slash commands

Slash commands are entered in the prompt box (they are not sent to the LLM).

### `/ts`

Runs a trading simulation using the active script `df` with `close` and `signal`.

### `/csvs [directory]`

Lists available `.csv` files under a directory.

- If directory is omitted, `.` is used.
- Hidden folders are skipped.

Examples:

```text
/csvs
/csvs data
```

### `/eodhd <symbol> <timeframe> <start-date> <end-date> [output.csv]`

Downloads market data from EODHD and writes it as CSV.

- `symbol`: e.g. `AAPL.US`
- `timeframe`: e.g. `1d`, `5m`, `1h`
- dates must be `YYYY-MM-DD`
- optional `output.csv` lets you choose file name/path
- after download, the app updates **CSV path** to the new file

Examples:

```text
/eodhd AAPL.US 1d 2024-01-01 2024-12-31
/eodhd MSFT.US 5m 2024-04-01 2024-04-05 data/msft_5m.csv
```

## EODHD API key setup

Set your API key before launching the app:

```bash
export EODHD_API_KEY="your_api_key_here"
```

If `EODHD_API_KEY` is missing, `/eodhd` returns an error.

## Session code safety

Session code runs with restricted builtins and preloaded `pd`, but this is not a hardened sandbox.
