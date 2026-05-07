# dataexplorer

A pandas-driven Textual app for interactively exploring time-series price files with help from a local LLM.

## Run

```bash
python -m pip install -r requirements.txt
python main.py --csv prices.csv --model llama3.1
```

## How it works

- The center editor shows the **live Python script** and is always editable.
- Use **Ask LLM** to request script updates from your local LLM (`ollama run <model>` by default).
- Use **Run Code** to execute the current script with pandas against the selected CSV path.
- Script output appears in the right-side output panel for iterative analysis and tests.

## Notes

- Session code runs with restricted builtins and preloaded `pd`, but this is not a hardened sandbox.
