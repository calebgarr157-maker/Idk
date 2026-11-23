import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
STATE = ROOT / "state"
REPORTS = ROOT / "reports" / "weekly"


def load_config() -> dict:
    return json.loads((ROOT / "bot" / "config.json").read_text())


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append dict rows to a CSV, writing the header when the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def upsert_row(path: Path, row: dict, key: str = "date") -> None:
    """Replace the row with the same key (re-runs of the same day) or append it."""
    rows = [r for r in read_rows(path) if r.get(key) != str(row[key])]
    rows.append({k: str(v) for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)
