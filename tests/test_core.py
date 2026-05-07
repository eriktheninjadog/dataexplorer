import unittest

from dataexplorer.core import build_prompt, default_script, extract_python_code, run_user_code


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

    def test_run_user_code_returns_stdout(self) -> None:
        output = run_user_code("print('ok')", "prices.csv")
        self.assertEqual(output, "ok")

    def test_run_user_code_allows_imports(self) -> None:
        output = run_user_code("import math\nprint(math.sqrt(4))", "prices.csv")
        self.assertEqual(output, "2.0")


if __name__ == "__main__":
    unittest.main()
