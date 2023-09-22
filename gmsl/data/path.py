from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | bytes | Path

PROCESSED_DIR = Path('processed-data')
parsed_dir = PROCESSED_DIR / 'parsed'
graph_save_dir = PROCESSED_DIR / 'graph'
