\
from pathlib import Path
from navigation_system.main import run

def test_full_run(tmp_path):
    cfg = Path(__file__).resolve().parents[1]/"config.yaml"
    run(str(cfg))
