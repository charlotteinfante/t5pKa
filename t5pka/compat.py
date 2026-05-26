import inspect
import math
import os
import subprocess
import sys
import types
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import torch
from sklearn.metrics import mean_squared_error
from transformers import TrainingArguments


class SimpleVocab:
    """Small serializable token vocabulary with the API this project needs."""

    def __init__(
        self,
        counter: Optional[Counter] = None,
        specials: Optional[Iterable[str]] = None,
        max_size: Optional[int] = None,
        tokens: Optional[Iterable[str]] = None,
    ) -> None:
        if tokens is None:
            ordered_tokens: List[str] = []
            for token in specials or []:
                if token not in ordered_tokens:
                    ordered_tokens.append(token)
            for token, _ in (counter or Counter()).most_common():
                if max_size is not None and len(ordered_tokens) >= max_size:
                    break
                if token not in ordered_tokens:
                    ordered_tokens.append(token)
        else:
            ordered_tokens = list(tokens)

        self.itos: List[str] = ordered_tokens
        self.stoi: Dict[str, int] = {token: idx for idx, token in enumerate(self.itos)}
        self.freqs: Counter = counter or Counter()

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi


def line_count(path: str) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def load_torch(path: str) -> Any:
    def _load() -> Any:
        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            return torch.load(path)

    def _load_with_legacy_torchtext_shim() -> Any:
        class LegacyTorchtextVocab:
            pass

        LegacyTorchtextVocab.__module__ = "torchtext.vocab"
        torchtext_module = types.ModuleType("torchtext")
        vocab_module = types.ModuleType("torchtext.vocab")
        vocab_module.Vocab = LegacyTorchtextVocab
        torchtext_module.vocab = vocab_module
        previous_torchtext = sys.modules.get("torchtext")
        previous_vocab = sys.modules.get("torchtext.vocab")
        sys.modules["torchtext"] = torchtext_module
        sys.modules["torchtext.vocab"] = vocab_module
        try:
            return _load()
        finally:
            if previous_torchtext is None:
                sys.modules.pop("torchtext", None)
            else:
                sys.modules["torchtext"] = previous_torchtext
            if previous_vocab is None:
                sys.modules.pop("torchtext.vocab", None)
            else:
                sys.modules["torchtext.vocab"] = previous_vocab

    try:
        return _load_with_legacy_torchtext_shim()
    except (AttributeError, ModuleNotFoundError, OSError):
        return _load()


def coerce_vocab(vocab: Any) -> SimpleVocab:
    if isinstance(vocab, SimpleVocab):
        return vocab
    if isinstance(vocab, dict) and "itos" in vocab:
        return SimpleVocab(
            tokens=vocab["itos"],
            counter=Counter(vocab.get("freqs", {})),
        )
    if hasattr(vocab, "get_itos"):
        tokens = list(vocab.get_itos())
        freqs = getattr(vocab, "freqs", Counter())
        return SimpleVocab(tokens=tokens, counter=Counter(freqs))
    if hasattr(vocab, "itos"):
        freqs = getattr(vocab, "freqs", Counter())
        return SimpleVocab(tokens=list(vocab.itos), counter=Counter(freqs))
    raise TypeError(f"Unsupported vocabulary object: {type(vocab)!r}")


def save_vocab(vocab: SimpleVocab, path: str) -> None:
    torch.save({"itos": vocab.itos, "freqs": dict(vocab.freqs)}, path)


def make_t5_config_kwargs(**kwargs: Any) -> Dict[str, Any]:
    config_kwargs = dict(kwargs)
    config_kwargs["use_cache"] = config_kwargs.pop("output_past", config_kwargs.get("use_cache", True))
    return config_kwargs


def make_training_arguments(**kwargs: Any) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and "eval_strategy" not in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if kwargs.get("eval_strategy") == "no" or kwargs.get("evaluation_strategy") == "no":
        kwargs["load_best_model_at_end"] = False
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def preprocess_logits_for_metrics(logits: Any, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim > 2:
        return torch.argmax(logits, dim=-1)
    if logits.ndim > 1 and logits.shape[-1] > 2:
        return torch.argmax(logits, dim=-1)
    return logits


def rmse(y_true: Any, y_pred: Any) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))
