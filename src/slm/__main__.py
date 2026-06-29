"""Enable ``python -m slm`` to invoke the Typer CLI."""

from __future__ import annotations

from slm.cli import app

if __name__ == "__main__":
    app()
