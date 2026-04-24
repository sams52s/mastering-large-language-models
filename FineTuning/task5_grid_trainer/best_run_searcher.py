# best_run_searcher.py
import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from peft import PeftModel, PeftConfig
from transformers import PreTrainedModel


class BestRunSearcher:
    """
    Finds and loads the best-performing model from grid search results.

    Attributes:
        search_dir (Path): Directory containing grid search results.
    """

    def __init__(self, search_dir: str) -> None:
        """
        Initializes the BestRunSearcher.

        Args:
            search_dir (str): Directory containing grid search results.
        """
        self.search_dir = Path(search_dir)

    def find_best_run(self) -> Tuple[Path, pd.DataFrame]:
        """
        Finds the best-performing run from grid search results.

        Returns:
            Tuple[Path, pd.DataFrame]: 
                - Path to the best checkpoint directory
                - DataFrame of all results with hyperparameters and eval losses
        """
        results = []

        for run_dir in self._iter_run_dirs():
            try:
                checkpoint_dir, eval_loss = self._find_best_checkpoint(run_dir)
                params = self._load_hyperparams(run_dir)

                results.append({
                    "run_path": str(run_dir),
                    "checkpoint_path": str(checkpoint_dir),
                    "eval_loss": eval_loss,
                    **params
                })

            except Exception as e:
                print(f"Skipping run {run_dir.name}: {str(e)}")
                continue

        if not results:
            raise ValueError("No valid runs found in grid search results")

        results = pd.DataFrame(results)
        results.sort_values("eval_loss", inplace=True, ascending=True)
        best_checkpoint = Path(results.iloc[0]["checkpoint_path"])
        return best_checkpoint, results

    def _iter_run_dirs(self):
        """Yields valid run directories within the search directory."""
        return (d for d in self.search_dir.iterdir() if d.is_dir())

    def _find_best_checkpoint(self, run_dir: Path) -> Tuple[Path, float]:
        """
        Finds the best checkpoint in a run directory.

        Args:
            run_dir (Path): Path to the run directory.

        Returns:
            Tuple[Path, float]: Path to the best checkpoint and its evaluation loss.
        """
        checkpoints = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {run_dir}")

        best_checkpoint, best_loss = None, float('inf')

        for checkpoint in checkpoints:
            try:
                with open(checkpoint / "trainer_state.json") as f:
                    loss = json.load(f).get("best_metric", float('inf'))

                if loss is not None and loss < best_loss:
                    best_checkpoint, best_loss = checkpoint, loss

            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                print(f"Skipping checkpoint {checkpoint.name}: {str(e)}")
                continue

        return best_checkpoint, best_loss

    def _load_hyperparams(self, run_dir: Path) -> Dict:
        """
        Loads hyperparameters from a run directory.

        Args:
            run_dir (Path): Path to the run directory.

        Returns:
            Dict: Dictionary of hyperparameters.
        """
        with open(run_dir / "hyperparams.json") as f:
            return json.load(f)

    @staticmethod
    def load_peft_model(base_model: PreTrainedModel, checkpoint_dir: Path) -> PeftModel:
        """
        Loads a PEFT model from a checkpoint directory.

        Args:
            base_model (PreTrainedModel): Original base model.
            checkpoint_dir (Path): Path to the checkpoint directory.

        Returns:
            PeftModel: Loaded PEFT model.
        """
        return PeftModel.from_pretrained(
            base_model,
            str(checkpoint_dir),
            is_trainable=False,
            config=PeftConfig.from_pretrained(str(checkpoint_dir))
        )