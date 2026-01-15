

class InferenceModel:
    def __init__(self, env):
        self.env = env
        self.model = None

    def load_model(self, model_path):
        """
        Load the model from the specified path.
        This method should be implemented by subclasses.
        """

        raise NotImplementedError("Subclasses must implement the load_model method.")

    def reshape_observation(self, observation):
        """
        Reshape the observation as needed for the model.
        This method can be overridden by subclasses if specific reshaping is required.
        """
        return observation
    
    def predict(self, observation):
        """
        Perform inference using the loaded model.
        This method should be implemented by subclasses.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        raise NotImplementedError("Subclasses must implement the predict method.")