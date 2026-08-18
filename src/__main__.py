"""Allow `python -m src` to use the same CLI as `main.py`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
