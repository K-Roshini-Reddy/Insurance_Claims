from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ScaffoldSpec:
    # folders relative to repo root
    folders: List[str]
    # files relative to repo root (created empty if not present)
    files: List[str]


SPEC = ScaffoldSpec(
    folders=[
        # Serving
        "backend/app",
        "backend/tests",

        # ML system (first-class)
        "ml/data/raw",
        "ml/data/processed",
        "ml/data/snapshots",
        "ml/features",
        "ml/training",
        "ml/evaluation",
        "ml/thresholding",
        "ml/pipelines",
        "ml/tests",

        # Single source of truth for constants
        "config",

        # Artifacts (generated outputs; typically gitignored)
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/thresholds",

        # Platform/Ops
        "monitoring/prometheus",
        "monitoring/grafana",
        "k8s/backend",
        "k8s/ml-jobs",
        "k8s/monitoring",
        "aws/s3",
        "aws/ecr",
        "aws/eks",
        ".github/workflows",

        # UI + docs
        "frontend",
        "docs",
    ],
    files=[
        # Root files
        "README.md",
        ".gitignore",
        "Makefile",

        # Config placeholders (we will fill later)
        "config/settings.yaml",
        "config/logging.yaml",

        # Backend placeholders
        "backend/requirements.txt",
        "backend/Dockerfile",
        "backend/app/__init__.py",
        "backend/app/main.py",
        "backend/app/schemas.py",
        "backend/app/config.py",
        "backend/app/model_loader.py",
        "backend/tests/test_health.py",

        # ML placeholders
        "ml/requirements.txt",
        "ml/data_snapshot.py",
        "ml/features/build_features.py",
        "ml/training/train.py",
        "ml/evaluation/evaluate.py",
        "ml/thresholding/tune_threshold.py",
        "ml/pipelines/README.md",
        "ml/tests/test_smoke.py",

        # Monitoring placeholders
        "monitoring/prometheus/prometheus.yml",
        "monitoring/grafana/README.md",

        # K8s placeholders
        "k8s/backend/deployment.yaml",
        "k8s/ml-jobs/training-job.yaml",
        "k8s/monitoring/monitoring.yaml",

        # AWS placeholders
        "aws/s3/README.md",
        "aws/ecr/README.md",
        "aws/eks/README.md",

        # CI/CD placeholders
        ".github/workflows/ci.yml",
        ".github/workflows/ml.yml",
    ],
)


def create_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_empty_file(path: Path) -> None:
    """
    Create an empty file if it doesn't exist.
    Safe behavior:
      - If file exists, do nothing.
      - If parent folders don't exist, create them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def main() -> None:
    repo_root = Path.cwd()

    created_folders = 0
    created_files = 0

    for folder in SPEC.folders:
        p = repo_root / folder
        if not p.exists():
            created_folders += 1
        create_folder(p)

    for file in SPEC.files:
        p = repo_root / file
        if not p.exists():
            created_files += 1
        create_empty_file(p)

    print("✅ Scaffold complete (folders + empty files only)")
    print(f"   Repo root: {repo_root}")
    print(f"   Folders created: {created_folders}")
    print(f"   Files created:   {created_files}")


if __name__ == "__main__":
    main()
