from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, RichLog, Static, TextArea

from .core import (
    default_script,
    get_data_summary,
    open_figure,
    request_llm_chat,
    request_llm_update,
    run_user_code_with_plots,
)


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
    #chat_mode_btn { background: $accent; }
    """

    def __init__(self, csv_path: str = "prices.csv", model: str = "llama3.1", llm_command: str = "ollama"):
        super().__init__()
        self.csv_path = csv_path
        self.model = model
        self.llm_command = llm_command
        self._chat_mode: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="controls"):
            yield Static("CSV path and prompt:")
            with Horizontal():
                yield Input(value=self.csv_path, placeholder="Path to price CSV", id="csv_path")
                yield Input(placeholder="Ask the LLM to improve the script...", id="prompt")
                yield Button("Ask LLM", id="ask_llm")
                yield Button("Run Code", id="run_code")
                yield Button("💬 Chat Mode", id="chat_mode_btn")
        with Horizontal(id="content"):
            yield TextArea.code_editor("", language="python", id="code")
            yield RichLog(id="output", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        code = self.query_one("#code", TextArea)
        code.text = default_script(self.csv_path)
        output = self.query_one("#output", RichLog)
        output.write("Ready. Edit code, ask the LLM for updates, then run the script.")
        output.write("Press '💬 Chat Mode' to ask questions about the data without generating code.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ask_llm":
            if self._chat_mode:
                self._handle_chat()
            else:
                self._handle_llm_update()
        elif event.button.id == "run_code":
            self._handle_run_code()
        elif event.button.id == "chat_mode_btn":
            self._toggle_chat_mode()

    def _toggle_chat_mode(self) -> None:
        self._chat_mode = not self._chat_mode
        code_widget = self.query_one("#code", TextArea)
        run_btn = self.query_one("#run_code", Button)
        ask_btn = self.query_one("#ask_llm", Button)
        toggle_btn = self.query_one("#chat_mode_btn", Button)
        prompt_input = self.query_one("#prompt", Input)
        output = self.query_one("#output", RichLog)

        if self._chat_mode:
            code_widget.display = False
            run_btn.display = False
            ask_btn.label = "Send"
            toggle_btn.label = "💻 Code Mode"
            prompt_input.placeholder = "Ask a question about your data..."
            output.write("--- Switched to Chat Mode. Ask questions about your data. ---")
        else:
            code_widget.display = True
            run_btn.display = True
            ask_btn.label = "Ask LLM"
            toggle_btn.label = "💬 Chat Mode"
            prompt_input.placeholder = "Ask the LLM to improve the script..."
            output.write("--- Switched to Code Mode. ---")

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

    def _handle_chat(self) -> None:
        output = self.query_one("#output", RichLog)
        prompt = self.query_one("#prompt", Input).value.strip()
        csv_path = self.query_one("#csv_path", Input).value.strip() or self.csv_path
        if not prompt:
            output.write("Enter a question first.")
            return
        output.write(f"You: {prompt}")
        output.write(f"Asking '{self.model}'...")
        try:
            data_summary = get_data_summary(csv_path)
            response = request_llm_chat(
                user_message=prompt,
                data_path=csv_path,
                data_summary=data_summary,
                model=self.model,
                command=self.llm_command,
            )
        except Exception as error:
            output.write(f"LLM error: {error}")
            return
        output.write(f"Assistant: {response}")
        self.query_one("#prompt", Input).value = ""

    def _handle_run_code(self) -> None:
        code = self.query_one("#code", TextArea)
        output = self.query_one("#output", RichLog)
        csv_path = self.query_one("#csv_path", Input).value.strip() or self.csv_path
        output.write(f"Running script with {csv_path}...")
        result, figure_paths = run_user_code_with_plots(code.text, csv_path)
        output.write(result)
        for path in figure_paths:
            output.write(f"Plot saved: {path}")
            open_figure(path)

