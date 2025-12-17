# claude_cost_callback.py
from langchain_core.callbacks import BaseCallbackHandler

class ClaudeCostCallback(BaseCallbackHandler):
    def __init__(self, *, stage: str, model: str, cost_tracker):
        self.stage = stage
        self.model = model
        self.cost_tracker = cost_tracker

    def on_llm_end(self, response, **kwargs):
        # Intenta obtener 'usage' de la nueva estructura de Claude
        try:
            if response.llm_output and "usage" in response.llm_output:
                usage = response.llm_output["usage"]
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            else:
                # Fallback para estructuras anteriores
                usage = response.generations[0][0].message.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
        except (AttributeError, KeyError, IndexError):
            # Fallback si ninguna de las estructuras funciona
            return

        self.cost_tracker.register_call(
            stage=self.stage,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
