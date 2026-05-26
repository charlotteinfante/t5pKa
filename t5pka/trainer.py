import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from transformers import Trainer


class EarlyStopTrainer(Trainer):
    """Trainer that keeps a copy of the checkpoint with the lowest eval loss."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_eval_loss: float = float("inf")

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        loss_key = f"{metric_key_prefix}_loss"
        cur_loss = metrics.get(loss_key)
        if cur_loss is not None and self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for checkpoint in Path(self.args.output_dir).glob("best_cp-*"):
                shutil.rmtree(checkpoint)
            output_dir = os.path.join(self.args.output_dir, f"best_cp-{self.state.global_step}")
            self.save_model(output_dir)
        return metrics
