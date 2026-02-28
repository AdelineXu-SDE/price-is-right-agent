import os

from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    Fine-tuned pricing model hosted remotely (e.g., Modal).
    If remote service is unavailable, this agent gracefully disables itself.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.available = False
        self.pricer = None

        try:
            import modal

            self.log("Specialist Agent connecting to remote modal service")
            Pricer = modal.Cls.from_name("pricer-service", "Pricer")
            self.pricer = Pricer()
            self.available = True
            self.log("Specialist Agent connected successfully")

        except Exception as e:
            self.log(
                "Specialist Agent unavailable (modal not configured). "
                "Falling back to other ensemble models."
            )

    def price(self, description: str) -> float:
        if not self.available:
            raise RuntimeError("Specialist model not available")

        self.log("Specialist Agent calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return float(result)
