import logging
import os
import datetime
import json
import numpy as np

class InferenceModel:
    def __init__(self):
        pass
        
    # ...existing code...
    def preprocess_observation(self, observation):
        """
        Preprocess the observation as needed for the model.
        This method can be overridden by subclasses if specific preprocessing is required.
        """
        return observation

    def postprocess_action(self, model_output):
        """
        Convert model output into a valid environment action.
        Override to handle (action, state) tuples, batched outputs, recurrent states,
        logits -> discrete action mapping, and action masking.
        """
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        return model_output

    def postprocess_reward(self, info):
        """
        Called after env.step(...). Can modify / normalize observations, apply reward shaping,
        update RNN masks on reset, mirror opponent views, or sanitize `info` for logging.
        Return (observations, rewards, terminations, truncations, info).
        """
        raise NotImplementedError("Subclasses must implement the postprocess_reward method.")
    

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        This method should be implemented by subclasses.
        """
        pass
    
    def predict(self, observation):
        """
        Perform inference using the loaded model.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")
    
    def reset(self):
        """
        Reset any internal state (e.g., RNN states) at the start of a new episode.
        """
        pass
    