import argparse
import time
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from wrappers.recurrentppo import RecurrentPPOInferenceModel
from src.clasher.gym_env import ClashRoyaleGymEnv


"""Export a sb3_contrib.RecurrentPPO policy to ONNX.

The exporter will attempt to build a small wrapper around the learned
policy that accepts:
 - `obs` : input observation tensor (batch, C, H, W) float32
 - `lstm_h`, `lstm_c` : LSTM hidden/cell states (num_layers, batch, hidden_size)
 - `episode_starts` : bool tensor (batch,)

The wrapper runs feature extraction -> mlp_extractor -> LSTM (if present)
-> action head and returns action logits and the new LSTM states.
"""


def build_policy_wrapper(policy):
	class PolicyWrapper(nn.Module):
		def __init__(self, policy):
			super().__init__()
			self.policy = policy

		def forward(self, obs, lstm_h, lstm_c, episode_starts):
			# obs: Tensor [batch, C, H, W] or [batch, obs_dim]
			x = obs
			# feature extractor
			if hasattr(self.policy, "extract_features"):
				features = self.policy.extract_features(x)
			elif hasattr(self.policy, "features_extractor"):
				features = self.policy.features_extractor(x)
			else:
				# some policies expose _get_latent or similar
				try:
					features = self.policy._get_features(x)
				except Exception:
					features = x

			# mlp extractor
			if hasattr(self.policy, "mlp_extractor"):
				try:
					latent_pi, latent_vf = self.policy.mlp_extractor(features)
				except Exception:
					# fallback: single latent
					latent_pi = features
			else:
				latent_pi = features

			# handle recurrent LSTM if present
			if hasattr(self.policy, "lstm") and self.policy.lstm is not None:
				lstm = self.policy.lstm
				# prepare input shape: LSTM expects (seq_len, batch, features)
				inp = latent_pi.unsqueeze(0) if latent_pi.dim() == 2 else latent_pi
				# try calling lstm with or without episode_starts
				try:
					out, (h_n, c_n) = lstm(inp, (lstm_h, lstm_c))
				except TypeError:
					out, (h_n, c_n) = lstm(inp, (lstm_h, lstm_c), episode_starts)
				latent_pi = out.squeeze(0)
			else:
				h_n = lstm_h
				c_n = lstm_c

			# action head
			if hasattr(self.policy, "action_net"):
				action_logits = self.policy.action_net(latent_pi)
			else:
				# try common alternatives
				if hasattr(self.policy, "actor"):
					action_logits = self.policy.actor(latent_pi)
				else:
					# last-resort: return latent as logits
					action_logits = latent_pi

			return action_logits, h_n, c_n

	return PolicyWrapper(policy)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("model", help="Path to the RecurrentPPO .zip model file")
	p.add_argument("out", help="Output ONNX filename")
	p.add_argument("--device", choices=["cpu","cuda"], default=None)
	p.add_argument("--opset", type=int, default=14)
	p.add_argument("--fp16", action="store_true", help="Export in fp16 (half) precision when possible")
	args = p.parse_args()

	model_path = args.model
	out_path = args.out

	# load model
	print(f"Loading model {model_path}")
	model = RecurrentPPO.load(model_path)
	policy = model.policy

	# build env to obtain observation shape
	env = PPOObsWrapper(ClashRoyaleGymEnv())
	sample_obs = env.reset()[0]
	# policy expects float tensors in NCHW
	obs_np = sample_obs
	obs_shape = obs_np.shape

	# default device choice
	if args.device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)

	wrapper = build_policy_wrapper(policy)
	wrapper.eval()
	wrapper.to(device)

	# lstm state sizes
	if hasattr(policy, "lstm") and policy.lstm is not None:
		lstm = policy.lstm
		# try to infer num_layers and hidden_size
		try:
			num_layers = lstm.num_layers
			hidden_size = lstm.hidden_size
		except Exception:
			# fallback to common attr
			num_layers = 1
			hidden_size = getattr(lstm, "hidden_size", 512)
	else:
		num_layers = 0
		hidden_size = 0

	# build example inputs
	batch = 1
	obs_tensor = torch.zeros((batch, *obs_shape), dtype=torch.float32, device=device)
	if args.fp16:
		obs_tensor = obs_tensor.half()

	if num_layers > 0:
		lstm_h = torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32, device=device)
		lstm_c = torch.zeros((num_layers, batch, hidden_size), dtype=torch.float32, device=device)
		if args.fp16:
			lstm_h = lstm_h.half(); lstm_c = lstm_c.half()
	else:
		# placeholders
		lstm_h = torch.zeros((1, batch, 1), dtype=torch.float32, device=device)
		lstm_c = torch.zeros((1, batch, 1), dtype=torch.float32, device=device)

	episode_starts = torch.zeros((batch,), dtype=torch.bool, device=device)

	# try an export
	print(f"Exporting to {out_path} on device {device} opset={args.opset} fp16={args.fp16}")
	dynamic_axes = {
		'obs': {0: 'batch'},
		'lstm_h': {1: 'batch'},
		'lstm_c': {1: 'batch'},
		'episode_starts': {0: 'batch'},
		'action_logits': {0: 'batch'}
	}

	# wrap inputs for export names
	input_names = ['obs', 'lstm_h', 'lstm_c', 'episode_starts']
	output_names = ['action_logits', 'lstm_h_out', 'lstm_c_out']

	# convert wrapper to fp16 if requested
	export_wrapper = wrapper.half() if args.fp16 else wrapper

	try:
		torch.onnx.export(
			export_wrapper,
			(obs_tensor, lstm_h, lstm_c, episode_starts),
			out_path,
			input_names=input_names,
			output_names=output_names,
			dynamic_axes=dynamic_axes,
			opset_version=args.opset,
			do_constant_folding=True,
		)
		print("ONNX export successful")
	except Exception as e:
		print("ONNX export failed:", e)


if __name__ == "__main__":
	main()


