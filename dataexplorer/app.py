from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, RichLog, Static, TextArea

from .core import (
    default_script,
    download_eodhd_csv,
    export_session_html,
    get_data_summary,
    list_available_csv_files,
    list_ollama_models,
    load_session_file,
    open_figure,
    request_llm_chat,
    request_llm_update,
    run_signal_trading_simulation,
    run_user_code_with_plots,
    sanitize_script_text,
    save_session_file,
)


class DataExplorerApp(App[None]):
    """Textual UI for interactive data exploration and strategy iteration."""

    TITLE = "Data Explorer"
    CSS = """
    Screen { layout: vertical; }
    #controls { height: 10; }
    #csv_path { width: 1fr; }
    #model { width: 1fr; }
    #llm_command { width: 1fr; }
    #prompt { width: 2fr; }
    #session_path { width: 2fr; }
    #export_path { width: 2fr; }
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
        # False => code-generation/editing mode, True => analyst-style chat mode.
        self._chat_mode: bool = False
        self._session_events: list[dict[str, str]] = []
        self._generated_plots: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="controls"):
            yield Static("CSV path, LLM settings, and prompt:")
            with Horizontal():
                yield Input(value=self.csv_path, placeholder="Path to price CSV", id="csv_path")
                yield Input(value=self.model, placeholder="Model (e.g. llama3.1)", id="model")
                yield Input(value=self.llm_command, placeholder="LLM command (e.g. ollama)", id="llm_command")
            with Horizontal():
                yield Input(placeholder="Ask the LLM to improve the script...", id="prompt")
                yield Button("Ask LLM", id="ask_llm")
                yield Button("Run Code", id="run_code")
                yield Button("💬 Chat Mode", id="chat_mode_btn")
                yield Button("Refresh Models", id="refresh_models")
            with Horizontal():
                yield Input(value="dataexplorer_session.json", placeholder="Session file path", id="session_path")
                yield Button("Save Session", id="save_session")
                yield Button("Load Session", id="load_session")
                yield Input(value="dataexplorer_session.html", placeholder="HTML export path", id="export_path")
                yield Button("Export HTML", id="export_html")
        with Horizontal(id="content"):
            yield TextArea.code_editor("", language="python", id="code")
            yield RichLog(id="output", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        code = self.query_one("#code", TextArea)
        code.text = default_script(self.csv_path)
        self._write_output("Ready. Edit code, ask the LLM for updates, then run the script.")
        self._write_output("Press '💬 Chat Mode' to ask questions about the data without generating code.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route all button events to the appropriate mode-specific handler."""
        if event.button.id == "ask_llm":
            if self._chat_mode:
                self._handle_chat()
            else:
                self._handle_llm_update()
        elif event.button.id == "run_code":
            self._handle_run_code()
        elif event.button.id == "chat_mode_btn":
            self._toggle_chat_mode()
        elif event.button.id == "refresh_models":
            self._handle_refresh_models()
        elif event.button.id == "save_session":
            self._handle_save_session()
        elif event.button.id == "load_session":
            self._handle_load_session()
        elif event.button.id == "export_html":
            self._handle_export_html()

    def _current_csv_path(self) -> str:
        """Return the current CSV path from UI input, falling back to app default."""
        return self.query_one("#csv_path", Input).value.strip() or self.csv_path

    def _current_model(self) -> str:
        return self.query_one("#model", Input).value.strip() or self.model

    def _current_llm_command(self) -> str:
        return self.query_one("#llm_command", Input).value.strip() or self.llm_command

    def _current_session_path(self) -> str:
        return self.query_one("#session_path", Input).value.strip() or "dataexplorer_session.json"

    def _current_export_path(self) -> str:
        return self.query_one("#export_path", Input).value.strip() or "dataexplorer_session.html"

    def _write_output(self, message: str, *, kind: str = "system", record: bool = True) -> None:
        output = self.query_one("#output", RichLog)
        output.write(message)
        if record:
            self._session_events.append({"kind": kind, "text": message})

    def _apply_chat_mode_ui(self, *, announce: bool) -> None:
        code_widget = self.query_one("#code", TextArea)
        run_btn = self.query_one("#run_code", Button)
        ask_btn = self.query_one("#ask_llm", Button)
        toggle_btn = self.query_one("#chat_mode_btn", Button)
        prompt_input = self.query_one("#prompt", Input)

        if self._chat_mode:
            code_widget.display = False
            run_btn.display = False
            ask_btn.label = "Send"
            toggle_btn.label = "💻 Code Mode"
            prompt_input.placeholder = "Ask a question about your data..."
            if announce:
                self._write_output("--- Switched to Chat Mode. Ask questions about your data. ---")
        else:
            code_widget.display = True
            run_btn.display = True
            ask_btn.label = "Ask LLM"
            toggle_btn.label = "💬 Chat Mode"
            prompt_input.placeholder = "Ask the LLM to improve the script..."
            if announce:
                self._write_output("--- Switched to Code Mode. ---")

    def _toggle_chat_mode(self) -> None:
        """Switch UI between code-editing mode and conversational analysis mode."""
        self._chat_mode = not self._chat_mode
        self._apply_chat_mode_ui(announce=True)

    def _handle_llm_update(self) -> None:
        code = self.query_one("#code", TextArea)
        prompt = self.query_one("#prompt", Input).value.strip()
        csv_path = self._current_csv_path()
        model = self._current_model()
        command = self._current_llm_command()
        if not prompt:
            self._write_output("Enter a prompt first.")
            return
        if self._handle_prompt_command(prompt):
            return
        self._write_output(f"Requesting update from local LLM model '{model}'...")
        try:
            updated = request_llm_update(
                user_request=prompt,
                current_code=code.text,
                data_path=csv_path,
                model=model,
                command=command,
            )
        except Exception as error:
            self._write_output(f"LLM error: {error}")
            return
        sanitized = sanitize_script_text(updated)
        if sanitized != updated:
            self._write_output("Removed non-ASCII characters from updated script.")
        code.text = sanitized
        self._write_output("Code updated from LLM response.")

    def _handle_chat(self) -> None:
        prompt = self.query_one("#prompt", Input).value.strip()
        csv_path = self._current_csv_path()
        model = self._current_model()
        command = self._current_llm_command()
        if not prompt:
            self._write_output("Enter a question first.")
            return
        if self._handle_prompt_command(prompt):
            return
        self._write_output(f"You: {prompt}", kind="user")
        self._write_output(f"Asking '{model}'...")
        try:
            data_summary = get_data_summary(csv_path)
            response = request_llm_chat(
                user_message=prompt,
                data_path=csv_path,
                data_summary=data_summary,
                model=model,
                command=command,
            )
        except Exception as error:
            self._write_output(f"LLM error: {error}")
            return
        self._write_output(f"Assistant: {response}", kind="assistant")
        self.query_one("#prompt", Input).value = ""

    def _handle_prompt_command(self, prompt: str) -> bool:
        """Handle slash-commands entered in the prompt input."""
        if not prompt.startswith("/"):
            return False
        command_name = prompt.split()[0].lower()
        if command_name == "/":
            self._write_output("Command cannot be empty. Try /ts.")
            self.query_one("#prompt", Input).value = ""
            return True
        if command_name == "/ts":
            self._handle_ts_command()
        elif command_name == "/csvs":
            self._handle_csvs_command(prompt)
        elif command_name == "/eodhd":
            self._handle_eodhd_command(prompt)
        else:
            self._write_output(f"Unknown command: {command_name}")
        self.query_one("#prompt", Input).value = ""
        return True

    def _handle_ts_command(self) -> None:
        """Run active script and simulate trades from ``signal`` on next ``close`` row."""
        code = self.query_one("#code", TextArea).text
        csv_path = self._current_csv_path()
        self._write_output(f"Running /ts with {csv_path} (columns: close, signal)...")
        result = run_signal_trading_simulation(
            code=code,
            data_path=csv_path,
            price_column="close",
            signal_column="signal",
        )
        self._write_output(result)

    def _handle_csvs_command(self, prompt: str) -> None:
        """List available CSV files from a target directory (defaults to current directory)."""
        parts = prompt.split()
        search_root = parts[1] if len(parts) > 1 else "."
        try:
            csv_files = list_available_csv_files(search_root)
        except Exception as error:
            self._write_output(f"/csvs error: {error}")
            return
        if not csv_files:
            self._write_output(f"No CSV files found under: {search_root}")
            return
        self._write_output(f"CSV files under {search_root}:")
        for path in csv_files:
            self._write_output(f"- {path}")

    def _handle_eodhd_command(self, prompt: str) -> None:
        """Download EODHD data to CSV and set the CSV path input to the downloaded file."""
        parts = prompt.split()
        required_arg_count = 5  # /eodhd + symbol + timeframe + start-date + end-date
        optional_arg_count = 6  # plus optional output path
        if len(parts) not in {required_arg_count, optional_arg_count}:
            self._write_output(
                "Usage: /eodhd <symbol> <timeframe> <start-date> <end-date> [output.csv]"
            )
            return
        symbol, timeframe, start_date, end_date = parts[1], parts[2], parts[3], parts[4]
        output_path = parts[5] if len(parts) == optional_arg_count else None
        self._write_output(
            f"Downloading EODHD data: {symbol} {timeframe} {start_date} -> {end_date}..."
        )
        try:
            written_path = download_eodhd_csv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
            )
        except Exception as error:
            self._write_output(f"/eodhd error: {error}")
            return
        self.query_one("#csv_path", Input).value = written_path
        self._write_output(f"EODHD CSV saved: {written_path}")
        self._write_output("CSV path updated to downloaded file.")

    def _handle_run_code(self) -> None:
        """Execute the current script and report text output and generated plots."""
        code = self.query_one("#code", TextArea)
        csv_path = self._current_csv_path()
        self._write_output(f"Running script with {csv_path}...")
        result, figure_paths = run_user_code_with_plots(code.text, csv_path)
        self._write_output(result)
        for path in figure_paths:
            self._write_output(f"Plot saved: {path}", kind="plot")
            if path not in self._generated_plots:
                self._generated_plots.append(path)
            if not open_figure(path):
                self._write_output(f"  (Could not open viewer for {path})")

    def _handle_refresh_models(self) -> None:
        command = self._current_llm_command()
        model_input = self.query_one("#model", Input)
        current_model = self._current_model()
        self._write_output(f"Requesting models from '{command} list'...")
        try:
            models = list_ollama_models(command=command)
        except Exception as error:
            self._write_output(f"Model list error: {error}")
            return
        if not models:
            self._write_output("No models returned by local LLM command.")
            return
        self._write_output(f"Available models: {', '.join(models)}")
        if current_model not in models:
            model_input.value = models[0]
            self._write_output(f"Model updated to '{models[0]}'.")

    def _build_session_payload(self) -> dict[str, object]:
        code = self.query_one("#code", TextArea)
        return {
            "title": "Data Explorer Session",
            "csv_path": self._current_csv_path(),
            "model": self._current_model(),
            "llm_command": self._current_llm_command(),
            "chat_mode": self._chat_mode,
            "code": code.text,
            "output_events": self._session_events,
            "generated_plots": self._generated_plots,
        }

    def _handle_save_session(self) -> None:
        session_path = self._current_session_path()
        payload = self._build_session_payload()
        try:
            saved_path = save_session_file(session_path, payload)
        except Exception as error:
            self._write_output(f"Could not save session: {error}")
            return
        self._write_output(f"Session saved: {saved_path}")

    def _handle_load_session(self) -> None:
        session_path = self._current_session_path()
        try:
            payload = load_session_file(session_path)
        except Exception as error:
            self._write_output(f"Could not load session: {error}")
            return

        csv_value = payload.get("csv_path")
        if isinstance(csv_value, str):
            self.query_one("#csv_path", Input).value = csv_value

        model_value = payload.get("model")
        if isinstance(model_value, str):
            self.query_one("#model", Input).value = model_value

        command_value = payload.get("llm_command")
        if isinstance(command_value, str):
            self.query_one("#llm_command", Input).value = command_value

        code_value = payload.get("code")
        if isinstance(code_value, str):
            self.query_one("#code", TextArea).text = code_value

        chat_mode_value = payload.get("chat_mode")
        if isinstance(chat_mode_value, bool):
            self._chat_mode = chat_mode_value
            self._apply_chat_mode_ui(announce=False)

        events: list[dict[str, str]] = []
        payload_events = payload.get("output_events")
        if isinstance(payload_events, list):
            for event in payload_events:
                if not isinstance(event, dict):
                    continue
                text = event.get("text")
                kind = event.get("kind", "system")
                if isinstance(text, str) and isinstance(kind, str):
                    events.append({"kind": kind, "text": text})

        output = self.query_one("#output", RichLog)
        output.clear()
        self._session_events = events.copy()
        for event in events:
            self._write_output(event["text"], kind=event["kind"], record=False)

        plots: list[str] = []
        payload_plots = payload.get("generated_plots")
        if isinstance(payload_plots, list):
            for path in payload_plots:
                if isinstance(path, str):
                    plots.append(path)
        self._generated_plots = plots
        self._write_output(f"Session loaded: {session_path}")

    def _handle_export_html(self) -> None:
        export_path = self._current_export_path()
        payload = self._build_session_payload()
        try:
            written_path = export_session_html(export_path, payload)
        except Exception as error:
            self._write_output(f"Could not export HTML: {error}")
            return
        self._write_output(f"Session HTML exported: {written_path}")
