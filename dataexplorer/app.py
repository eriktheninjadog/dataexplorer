from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, RichLog, Static, TextArea

from .core import default_script, request_llm_update, run_user_code


class DataExplorerApp(App[None]):
    TITLE = "Data Explorer"
    CSS = """
    Screen { layout: vertical; }
    #controls { height: 5; }
    #csv_path { width: 1fr; }
    #prompt { width: 2fr; }
    #content { height: 1fr; }
    #code { width: 2fr; }
    #output { width: 1fr; }
    """

    def __init__(self, csv_path: str = "prices.csv", model: str = "llama3.1", llm_command: str = "ollama"):
        super().__init__()
        self.csv_path = csv_path
        self.model = model
        self.llm_command = llm_command

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="controls"):
            yield Static("CSV path and prompt:")
            with Horizontal():
                yield Input(value=self.csv_path, placeholder="Path to price CSV", id="csv_path")
                yield Input(placeholder="Ask the LLM to improve the script...", id="prompt")
                yield Button("Ask LLM", id="ask_llm")
                yield Button("Run Code", id="run_code")
        with Horizontal(id="content"):
            yield TextArea.code_editor("", language="python", id="code")
            yield RichLog(id="output", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        code = self.query_one("#code", TextArea)
        code.text = default_script(self.csv_path)
        output = self.query_one("#output", RichLog)
        output.write("Ready. Edit code, ask the LLM for updates, then run the script.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask_llm":
            self._handle_llm_update()
        elif event.button.id == "run_code":
            self._handle_run_code()

    def _handle_llm_update(self) -> None:
        code = self.query_one("#code", TextArea)
        output = self.query_one("#output", RichLog)
        prompt = self.query_one("#prompt", Input).value.strip()
        csv_path = self.query_one("#csv_path", Input).value.strip() or self.csv_path
        if not prompt:
            output.write("Enter a prompt first.")
            return
        output.write(f"Requesting update from local LLM model '{self.model}'...")
        try:
            updated = request_llm_update(
                user_request=prompt,
                current_code=code.text,
                data_path=csv_path,
                model=self.model,
                command=self.llm_command,
            )
        except Exception as error:
            output.write(f"LLM error: {error}")
            return
        code.text = updated
        output.write("Code updated from LLM response.")

    def _handle_run_code(self) -> None:
        code = self.query_one("#code", TextArea)
        output = self.query_one("#output", RichLog)
        csv_path = self.query_one("#csv_path", Input).value.strip() or self.csv_path
        output.write(f"Running script with {csv_path}...")
        result = run_user_code(code.text, csv_path)
        output.write(result)

