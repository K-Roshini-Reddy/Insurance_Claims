from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogger:
    """
    Industry-style training logger:
      - writes to artifacts/logs/training_<batch_id>.log
      - all log formatting lives here (NOT in train.py)
    """

    def __init__(self, batch_id: str, log_dir: Path | None = None, also_console: bool = True):
        self.batch_id = batch_id
        self.log_dir = log_dir or (Path("artifacts") / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"training_{batch_id}.log"

        self._logger = logging.getLogger(f"training.{batch_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # avoid duplicate handlers on reruns in same process
        if not self._logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

            if also_console:
                sh = logging.StreamHandler()
                sh.setFormatter(fmt)
                self._logger.addHandler(sh)

        self._logger.info("logger_initialized batch_id=%s log_file=%s", batch_id, str(self.log_path))

    def start(self, model_keys: List[str], cfg: Dict[str, Any]) -> None:
        self._logger.info("step_4_5_start batch_id=%s models=%s cfg=%s",
                          self.batch_id, ",".join(model_keys), json.dumps(cfg, default=str))

    def model_started(self, model_key: str, run_id: Optional[str]) -> None:
        self._logger.info("model_start model=%s run_id=%s", model_key, run_id or "none")

    def model_metrics(self, model_key: str, metrics: Dict[str, float], model_path: str, run_id: Optional[str]) -> None:
        self._logger.info(
            "model_metrics model=%s run_id=%s auc=%.6f precision=%.6f recall=%.6f f1=%.6f saved=%s",
            model_key,
            run_id or "none",
            metrics["auc"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            model_path,
        )

    def champion_selected(self, champion: Dict[str, Any], selection_json_path: str, champion_model_path: str) -> None:
        self._logger.info(
            "champion_selected model=%s auc=%.6f precision=%.6f recall=%.6f f1=%.6f selection_json=%s champion_artifact=%s",
            champion["model_key"],
            float(champion["auc"]),
            float(champion["precision"]),
            float(champion["recall"]),
            float(champion["f1"]),
            selection_json_path,
            champion_model_path,
        )

    def done(self) -> None:
        self._logger.info("step_4_5_done batch_id=%s", self.batch_id)
