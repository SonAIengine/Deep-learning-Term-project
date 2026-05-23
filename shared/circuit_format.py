"""Common schema for circuit-analysis results.

All three tracks dump findings as `CircuitResult` JSON in `results/`.
P3's comparison code joins them by task name to compute the universality score.
"""
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import json


@dataclass
class CircuitHead:
    layer: int
    head: int
    role: Optional[str] = None              # e.g., "name_mover", "binding_propagator"
    ablation_drop: Optional[float] = None   # accuracy drop when this head is ablated
    patching_recovery: Optional[float] = None  # logit recovery from activation patching


@dataclass
class CircuitResult:
    task: str           # "modular" | "code_binding" | "ioi"
    model: str          # e.g., "gpt2"
    n_samples: int
    circuit: List[CircuitHead] = field(default_factory=list)
    notes: str = ""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "task": self.task,
                "model": self.model,
                "n_samples": self.n_samples,
                "circuit": [asdict(h) for h in self.circuit],
                "notes": self.notes,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CircuitResult":
        with open(path) as f:
            d = json.load(f)
        d["circuit"] = [CircuitHead(**h) for h in d["circuit"]]
        return cls(**d)

    def head_set(self) -> set:
        """Return set of (layer, head) tuples — used for overlap scoring."""
        return {(h.layer, h.head) for h in self.circuit}


def overlap_score(a: CircuitResult, b: CircuitResult) -> float:
    """Jaccard overlap of two circuits' head sets — basic universality metric."""
    sa, sb = a.head_set(), b.head_set()
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
