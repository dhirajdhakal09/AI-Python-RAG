import io
import unittest
from contextlib import redirect_stdout

from ai_python_app.main import main


class MainTests(unittest.TestCase):
    def test_main_prints_hello_and_langsmith_value(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main()

        output = buffer.getvalue()
        self.assertIn("Hello from AI-Python app!", output)
        self.assertIn("LANGSMITH_API_KEY:", output)


if __name__ == "__main__":
    unittest.main()
