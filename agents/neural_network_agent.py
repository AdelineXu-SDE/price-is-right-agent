import os
from agents.agent import Agent
from agents.deep_neural_network import DeepNeuralNetworkInference


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, weights_path: str = "deep_neural_network.pth"):
        """
        NN model is optional. If weights are missing, the agent will gracefully skip.
        """
        self.weights_path = weights_path
        self.neural_network = None

        self.log("Neural Network Agent is initializing")
        if not os.path.exists(self.weights_path):
            self.log(
                f"Neural Network weights not found at {self.weights_path}; skipping NN model")
            return

        self.neural_network = DeepNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load(self.weights_path)
        self.log(
            f"Neural Network Agent is ready (weights loaded from {self.weights_path})")

    def price(self, description: str) -> float:
        """
        Predict price with NN if available; otherwise return 0.0 (ignored by ensemble weights).
        """
        if not self.neural_network:
            return 0.0
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
