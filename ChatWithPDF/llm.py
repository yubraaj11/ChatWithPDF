import random

from llama_index.llms.ollama import Ollama

class OllamaModel:
    """
    Initializes the OllamaModel class, which can be used as llm for question answering tasks

    Args:
        model_name (str): The name of the model user wants to use
        temperature (int): The degree of randomness for the llm
    """
    def __init__(self,
                 model_name = "llama3.2:3b-instruct-q4_K_M",
                 temperature = random.choice([0.5, 0.7, 0.9])
                 ):
        self.model_name = model_name
        self.temperature = temperature

    def load_model(self) -> Ollama:
        """
        Loads llm based on the model_name and temperature provided
        Returns:
            Ollama: Initialized ollama model

        """
        llm = Ollama(model=self.model_name,
                     temperature=self.temperature,
                     )

        return llm

