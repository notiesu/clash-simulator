from src.clasher.model import InferenceModel

class VecInferenceModel(InferenceModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        #look for state arrays, make sure each environment has its own copy to avoid cross-talk
    # ...existing code...
    def preprocess_observation(self, observations):
        """
        Preprocess the observation as needed for the model.
        This method can be overridden by subclasses if specific preprocessing is required.
        """
        for i in range(len(observations)):
            observations[i] = self.model.preprocess_observation(observations[i])
        return observations

    def postprocess_action(self, actions):
        """
        Convert model output into a valid environment action.
        Override to handle (action, state) tuples, batched outputs, recurrent states,
        logits -> discrete action mapping, and action masking.
        """
        for i in range(len(actions)):
            actions[i] = self.model.postprocess_action(actions[i])
        return actions

    def postprocess_reward(self, info):
        """
        Called after env.step(...). Can modify / normalize observations, apply reward shaping,
        update RNN masks on reset, mirror opponent views, or sanitize `info` for logging.
        Return (observations, rewards, terminations, truncations, info).
        """
        return 0.0
    
    def predict(self, observations, valid_action_masks=None, states=None):
        """
        Perform inference using the loaded model.
        This method should be implemented by subclasses.
        """
        actions = []
        next_states = []
        for i in range(len(observations)):
            action, state = self.model.predict(observations[i], valid_action_mask=valid_action_masks[i] if valid_action_masks is not None else None, state=states[i] if states is not None else None)
            actions.append(action)
            next_states.append(state)
        return actions, next_states
    
    