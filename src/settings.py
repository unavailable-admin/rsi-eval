"""External prepared workspace location; code is always loaded from this repository."""
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
value = os.environ.get("RSI_EVAL_WORKSPACE")
if not value:
    raise SystemExit("Set RSI_EVAL_WORKSPACE to the prepared workspace (see docs/RSI_AGENT_INTEGRATION.md).")
WORKSPACE = Path(value).expanduser().resolve()
if not WORKSPACE.is_dir():
    raise SystemExit(f"Workspace does not exist: {WORKSPACE}")
