#!/usr/bin/env python3
"""Launcher for the MCP server.

Equivalent to `python -m mcp_memory.server`, but as a plain script path.
MCP client configs pass server arguments through option parsers that treat
a leading `-m` as their own flag, so pointing at this file avoids the
problem entirely -- and it makes the package importable without depending
on the client to set the working directory or PYTHONPATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_memory.server import main  # noqa: E402

if __name__ == "__main__":
    main()
