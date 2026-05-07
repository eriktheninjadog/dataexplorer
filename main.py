from __future__ import annotations

import argparse

from dataexplorer.app import DataExplorerApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive pandas + local LLM data explorer")
    parser.add_argument("--csv", default="prices.csv", help="Default CSV path to load in the script")
    parser.add_argument("--model", default="llama3.1", help="Local LLM model name")
    parser.add_argument("--llm-command", default="ollama", help="Command used to invoke the local LLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = DataExplorerApp(csv_path=args.csv, model=args.model, llm_command=args.llm_command)
    app.run()


if __name__ == "__main__":
    main()

