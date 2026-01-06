"""Запуск всех сценариев (3 сценария из диплома).

Команда:
    python run_all.py
"""
from pathlib import Path
import subprocess
import sys

def main():
    base = Path(__file__).resolve().parent / "navigation_system"
    cfgs = [
        base / "config_uav_route.yaml",
        base / "config_uav_orbit.yaml",
        base / "config_uav_route_with_outage.yaml",
    ]
    for cfg in cfgs:
        print(f"==> running: {cfg.name}")
        subprocess.check_call([sys.executable, "-m", "navigation_system.main", str(cfg)], cwd=base.parent)

if __name__ == "__main__":
    main()
