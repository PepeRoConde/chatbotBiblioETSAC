# cost_tracker.py
from typing import Dict, Any
from collections import defaultdict

class CostTracker:
    def __init__(self, pricing: Dict[str, Dict[str, float]]):
        self.pricing = pricing
        self.reset()

    def reset(self):
        self.calls = []
        self.totals = defaultdict(float)

    def register_call(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        prices = self.pricing.get(model, {"input": 0.0, "output": 0.0})

        cost = (
            (input_tokens / 1_000_000) * prices["input"]
            + (output_tokens / 1_000_000) * prices["output"]
        )

        record = {
            "stage": stage,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }

        self.calls.append(record)
        self.totals["total_cost"] += cost
        self.totals[f"{stage}_cost"] += cost
        self.totals[f"{model}_cost"] += cost

    def summary(self) -> Dict[str, Any]:
        by_stage = {
            k.replace("_cost", ""): v
            for k, v in self.totals.items()
            if k.endswith("_cost") and k != "total_cost"
        }
        # Eliminar modelos del desglose por etapa para evitar redundancia
        for model_name in self.pricing.keys():
            by_stage.pop(model_name, None)

        return {
            "total_cost": self.totals["total_cost"],
            "by_stage": by_stage,
            "calls": self.calls,
        }
