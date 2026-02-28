from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.preprocessor import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection):
        """
        Ensemble pricing via three predictors:
        - Frontier (RAG + frontier LLM): robust but slower / more expensive
        - Specialist (fine-tuned): accurate in-domain
        - Neural net (local): fast and cheap

        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Estimate a product's true value using a weighted fusion of three models.
        """
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        neural_network = self.neural_network.price(rewrite)
        # If NN weights are missing, NeuralNetworkAgent returns 0.0.
        # In that case, re-normalize weights to avoid dragging the estimate down.
        if neural_network == 0.0:
            combined = frontier * 0.9 + specialist * 0.1
        else:
            combined = frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined
