"""
Full Pipeline Orchestrator
Runs stages 01 through 09 in order.
Each stage must exit 0 before the next begins.
Overall stdout/stderr are also captured per stage in outputs/logs/.
"""
import os
import sys
import subprocess
from datetime import datetime

# Ensure config paths are created
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    "01_prepare_dataset.py",
    "00_dataset_validation.py",
    "02_extract_embeddings.py",
    "03_reduce_dimensions.py",
    "04_cluster_embeddings.py",
    "05_scene_retrieval.py",
    "06_generate_statistics.py",
    "07_create_visualizations.py",
    "08_embedding_explorer.py",
    "09_embedding_analysis.py",
]

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(BASE_DIR, "outputs", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def run_stage(script_name: str) -> bool:
    script_path = os.path.join(SRC_DIR, script_name)
    stage_num   = script_name.split("_")[0]
    log_file    = os.path.join(LOG_DIR, f"pipeline_{stage_num}.log")

    print(f"\n{'='*60}")
    print(f"  RUNNING: {script_name}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}")

    with open(log_file, "w", encoding="utf-8") as lf:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        lf.write(result.stdout)
        print(result.stdout)

    if result.returncode != 0:
        print(f"\n  ERROR: {script_name} exited with code {result.returncode}. Stopping pipeline.")
        return False

    print(f"  OK: {script_name} completed successfully.")
    return True


def main():
    print(f"\nGeoSceneFM Pipeline  —  {datetime.now().isoformat()}")
    print("=" * 60)

    for script in SCRIPTS:
        if not run_stage(script):
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — all stages finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
