import os
import unittest
import builtins
from unittest.mock import patch

from dataexplorer.core import (
    build_chat_prompt,
    build_prompt,
    default_script,
    extract_python_code,
    request_llm_chat,
    request_llm_update,
    run_user_code,
    run_user_code_with_plots,
)


class CoreTests(unittest.TestCase):
    def test_default_script_uses_data_path(self) -> None:
        script = default_script("prices.csv")
        self.assertIn("prices.csv", script)
        self.assertIn("pd.read_csv", script)

    def test_extract_python_code_from_fence(self) -> None:
        raw = "```python\nprint('hello')\n```"
        self.assertEqual(extract_python_code(raw), "print('hello')")

    def test_extract_python_code_without_fence(self) -> None:
        raw = "print('hello')"
        self.assertEqual(extract_python_code(raw), "print('hello')")

    def test_build_prompt_contains_context(self) -> None:
        prompt = build_prompt("show monthly volatility", "print('x')", "prices.csv")
        self.assertIn("show monthly volatility", prompt)
        self.assertIn("print('x')", prompt)
        self.assertIn("prices.csv", prompt)

    def test_build_prompt_includes_trading_simulation_guidance(self) -> None:
        prompt = build_prompt("run a trading simulation", "print('x')", "prices.csv")
        self.assertIn("trading simulation/backtest", prompt)
        self.assertIn("indicator columns", prompt)
        self.assertIn("signal column values", prompt)
        self.assertIn("default to 'close'", prompt)
        self.assertIn("pnl, Sharpe, and Sortino", prompt)
        self.assertIn("max drawdown", prompt)

    def test_run_user_code_returns_stdout(self) -> None:
        output = run_user_code("print('ok')", "prices.csv")
        self.assertEqual(output, "ok")

    def test_run_user_code_returns_error_message(self) -> None:
        output = run_user_code("raise ValueError('bad data')", "prices.csv")
        self.assertEqual(output, "Error: bad data")

    def test_run_user_code_preserves_stdout_before_error(self) -> None:
        output = run_user_code("print('before')\nraise ValueError('bad data')", "prices.csv")
        self.assertIn("before", output)
        self.assertIn("Error: bad data", output)

    def test_run_user_code_blocks_import_statements(self) -> None:
        output = run_user_code("import math\nprint(math.sqrt(4))", "prices.csv")
        self.assertIn("import statements are disabled", output)

    def test_run_user_code_works_without_pandas_when_not_referenced(self) -> None:
        real_import = builtins.__import__

        def import_with_missing_pandas(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_with_missing_pandas):
            output = run_user_code("print('no pandas needed')", "prices.csv")
        self.assertEqual(output, "no pandas needed")

    @patch("dataexplorer.core.subprocess.run")
    def test_request_llm_update_extracts_code(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "```python\nprint('ok')\n```"
        mock_run.return_value.stderr = ""
        output = request_llm_update("do it", "print('x')", "prices.csv")
        self.assertEqual(output, "print('ok')")

    @patch("dataexplorer.core.subprocess.run")
    def test_request_llm_update_raises_on_failure(self, mock_run) -> None:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "no model"
        with self.assertRaises(RuntimeError):
            request_llm_update("do it", "print('x')", "prices.csv")

    def test_build_chat_prompt_contains_context(self) -> None:
        prompt = build_chat_prompt("what is the max price?", "prices.csv", "Shape: 10 rows x 3 cols")
        self.assertIn("what is the max price?", prompt)
        self.assertIn("prices.csv", prompt)
        self.assertIn("Shape: 10 rows x 3 cols", prompt)
        self.assertNotIn("Return only valid Python code", prompt)

    def test_build_chat_prompt_without_summary(self) -> None:
        prompt = build_chat_prompt("describe the data", "prices.csv")
        self.assertIn("describe the data", prompt)
        self.assertIn("prices.csv", prompt)

    @patch("dataexplorer.core.subprocess.run")
    def test_request_llm_chat_returns_plain_text(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "The max price is 100."
        mock_run.return_value.stderr = ""
        output = request_llm_chat("what is the max price?", "prices.csv")
        self.assertEqual(output, "The max price is 100.")

    @patch("dataexplorer.core.subprocess.run")
    def test_request_llm_chat_raises_on_failure(self, mock_run) -> None:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "no model"
        with self.assertRaises(RuntimeError):
            request_llm_chat("any question", "prices.csv")

    def test_run_user_code_with_plots_returns_stdout(self) -> None:
        output, paths = run_user_code_with_plots("print('hello plots')", "prices.csv")
        self.assertEqual(output, "hello plots")
        self.assertEqual(paths, [])

    def test_run_user_code_with_plots_returns_error(self) -> None:
        output, paths = run_user_code_with_plots("raise ValueError('plot error')", "prices.csv")
        self.assertIn("plot error", output)
        self.assertEqual(paths, [])

    def test_run_user_code_with_plots_captures_plt_show(self) -> None:
        code = "plt.figure()\nplt.plot([1, 2, 3])\nplt.show()"
        output, paths = run_user_code_with_plots(code, "prices.csv")
        self.assertEqual(len(paths), 1)
        self.assertTrue(os.path.exists(paths[0]))
        self.assertTrue(paths[0].endswith(".png"))


if __name__ == "__main__":
    unittest.main()
