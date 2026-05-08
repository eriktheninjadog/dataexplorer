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
- Prompt strings that start with `/` are treated as app commands (not sent to the LLM).
- Use `/ts` to run a trading simulation on the active script output using `close` and `signal`; signals execute on the next row's price.
- Change **Model** and **LLM command** directly inside the app at any time.
- Use **Refresh Models** to query `ollama list`, then select one of the returned model names.
- Use **Run Code** to execute the current script with pandas against the selected CSV path.
- Script output appears in the right-side output panel for iterative analysis and tests.
- Use **Save Session** / **Load Session** to persist and continue a full session.
- Use **Export HTML** to export a standalone session report including chat/output history, current code, and embedded generated graph images.

## Notes

- Session code runs with restricted builtins and preloaded `pd`, but this is not a hardened sandbox.
