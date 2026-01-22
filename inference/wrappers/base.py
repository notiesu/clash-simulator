import logging
import os
import datetime
import json
import numpy as np

class InferenceModel:
    def __init__(self, env, printLogs=False):
        self.env = env
        self.model = None
        self.printLogs = printLogs
        self.lastObs = None

    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        raise NotImplementedError("Subclasses must implement the load_model method.")
        
    # ...existing code...
    def preprocess_observation(self, observation):
        """
        Preprocess the observation as needed for the model.
        This method can be overridden by subclasses if specific preprocessing is required.
        """
        return observation

    def postprocess_action(self, model_output, agent_id: str = None):
        """
        Convert model output into a valid environment action.
        Override to handle (action, state) tuples, batched outputs, recurrent states,
        logits -> discrete action mapping, and action masking.
        """
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        return model_output

    def postprocess_rollout(self, observations, rewards, terminations, truncations, info):
        """
        Called after env.step(...). Can modify / normalize observations, apply reward shaping,
        update RNN masks on reset, mirror opponent views, or sanitize `info` for logging.
        Return (observations, rewards, terminations, truncations, info).
        """
        return observations, rewards, terminations, truncations, info
    

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        This method should be implemented by subclasses.
        """

        raise NotImplementedError("Subclasses must implement the load_model method.")
    
    def predict(self, observation):
        """
        Perform inference using the loaded model.
        This method should be implemented by subclasses.
        """
        if self.model_p0 is None or self.model_p1 is None:
            raise ValueError("Models are not loaded. Please load the models before prediction.")
        raise NotImplementedError("Subclasses must implement the predict method.")
    