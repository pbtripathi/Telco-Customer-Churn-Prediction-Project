"""Export the saved notebook outputs to a PDF."""

import asyncio
import sys
from pathlib import Path

from nbconvert.nbconvertapp import NbConvertApp


def main() -> None:
    notebook = Path(__file__).with_name("Telco_Churn_Prediction.ipynb")

    if sys.platform == "win32":
        original_run = asyncio.run

        def run_with_proactor(coro, *, debug=None, loop_factory=None):
            return original_run(
                coro,
                debug=debug,
                loop_factory=asyncio.ProactorEventLoop,
            )

        # WebPDF needs subprocess support, which Windows selector loops lack.
        asyncio.run = run_with_proactor

    NbConvertApp.launch_instance(
        [
            "--to=webpdf",
            "--allow-chromium-download",
            f"--output={notebook.stem}",
            f"--output-dir={notebook.parent}",
            str(notebook),
        ]
    )


if __name__ == "__main__":
    main()
