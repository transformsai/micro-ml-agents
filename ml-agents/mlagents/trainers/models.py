import logging

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger("mlagents.trainers")


class VisualEncoder(nn.Module):
    def __init__(self, camera_params, h_size, num_layers):
        super().__init__()
        o_size_h = camera_params["height"]
        o_size_w = camera_params["width"]
        bw = camera_params["blackAndWhite"]
        c_channels = 1 if bw else 3

        self.model = nn.Sequential(
            nn.Conv2d(c_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.ELU()
        )
        self.vector_encoder = VectorEncoder(32*o_size_h*o_size_w, h_size, num_layers)

    def forward(self, image_input):
        hidden = self.model(image_input)
        hidden = hidden.flatten(start_dim=1)
        return self.vector_encoder(hidden)

class VectorEncoder(nn.Module):
    def __init__(self, input_size, h_size, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, h_size))
            layers.append(self.swish())
            input_size = h_size
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        self.model.apply(self._init_linear)

    def _init_linear(self, x):
        if type(x) == nn.Linear:
            # slightly different from tf implementation
            nn.init.kaiming_normal_(x.weight, a=0.2)

    def forward(self, vector_input):
        return self.model(vector_input)

    class swish(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x * torch.sigmoid(x)


class ObservationEncoder(nn.Module):
    def __init__(self, h_size, num_layers, brain):
        super().__init__()
        self.num_vis_obs = brain.number_visual_observations
        self.vector_obs_size = brain.vector_observation_space_size * \
            brain.num_stacked_vector_observations

        self.visual_encoders = nn.ModuleList()

        for i in range(self.num_vis_obs):
            self.visual_encoders.append(VisualEncoder(
                brain.camera_resolutions[i], h_size, num_layers))
        self.vector_encoder = VectorEncoder(self.vector_obs_size, h_size, num_layers)

        self.obs_size = self.num_vis_obs * h_size
        if self.vector_obs_size > 0:
            self.obs_size += h_size

    def forward(self, visual_in, vector_in):
        hidden_state, hidden_visual = None, None
        if self.num_vis_obs > 0:
            encoded_visuals = []
            for i in range(self.num_vis_obs):
                encoded_visual = self.visual_encoder[i](visual_in[i])
                encoded_visuals.append(encoded_visual)
            hidden_visual = torch.cat(encoded_visuals, 1)
        if self.vector_obs_size > 0:
            hidden_state = self.vector_encoder(vector_in)
        if hidden_state is not None and hidden_visual is not None:
            final_hidden = torch.cat([hidden_visual, hidden_state], 1)
        elif hidden_state is None and hidden_visual is not None:
            final_hidden = hidden_visual
        elif hidden_state is not None and hidden_visual is None:
            final_hidden = hidden_state
        else:
            raise Exception(
                "No valid network configuration possible. "
                "There are no states or observations in this brain"
            )
        return final_hidden


class CCActorCritic(nn.Module):
    def __init__(self, h_size, num_layers, stream_names, brain):
        super().__init__()
        self.act_size = brain.vector_action_space_size
        self.policy_observation_encoder = ObservationEncoder(h_size, num_layers, brain)
        self.value_observation_encoder = ObservationEncoder(h_size, num_layers, brain)

        self.policy_layer = torch.nn.Linear(self.policy_observation_encoder.obs_size, self.act_size[0])
        self.stream_names = stream_names
        self.value_layers = nn.ModuleDict()
        for name in self.stream_names:
            self.value_layers[name] = torch.nn.Linear(self.value_observation_encoder.obs_size, 1)

        self.log_sigma_sq = nn.Parameter(torch.zeros(self.act_size[0]))
        self.const1 = np.log(2.0 * np.pi)
        self.const2 = np.log(2 * np.pi * np.e)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.policy_layer.weight, a=10)

    def get_value_estimate(self, visual_in, vector_in):
        value_heads = {}
        hidden_value = self.value_observation_encoder(visual_in, vector_in)
        for name in self.stream_names:
            value = self.value_layers[name](hidden_value)
            value_heads[name] = value.data.numpy()
        return value_heads

    def forward(self, input_dict):
        visual_in = input_dict.get("visual_obs", None)
        vector_in = input_dict.get("vector_obs", None)
        output_pre = input_dict.get("actions_pre", None)
        epsilon = input_dict["random_normal_epsilon"]

        hidden_policy = self.policy_observation_encoder(visual_in, vector_in)
        hidden_value = self.value_observation_encoder(visual_in, vector_in)

        mu = self.policy_layer(hidden_policy)
        sigma_sq = torch.exp(self.log_sigma_sq)

        # Clip and scale output to ensure actions are always within [-1, 1] range.
        if output_pre is None:
            output_pre = (mu + torch.sqrt(sigma_sq) * epsilon).detach()
        output = torch.clamp(output_pre, -3, 3) / 3

        # Compute probability of model output.
        all_log_probs = (
            - 0.5 * torch.pow(output_pre - mu, 2) / sigma_sq
            - 0.5 * self.const1
            - 0.5 * self.log_sigma_sq
        )

        entropy = 0.5 * (self.const2 + self.log_sigma_sq).mean()

        value_heads = {}
        for name in self.stream_names:
            value = self.value_layers[name](hidden_value)
            value_heads[name] = value
        value = torch.mean(torch.stack(list(value_heads.values())))
        entropy = torch.ones(value.flatten().size()) * entropy

        return output, output_pre, all_log_probs, value_heads, value, entropy

class DCActorCritic(nn.Module):
    def __init__(self, h_size, num_layers, brain):
        super().__init__()
        raise Exception("DCActorCritic to be implemented")
