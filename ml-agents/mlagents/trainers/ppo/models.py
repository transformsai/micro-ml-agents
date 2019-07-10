import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mlagents.trainers.models import DCActorCritic, CCActorCritic

logger = logging.getLogger("mlagents.trainers")


class PPOModel(nn.Module):
    def __init__(
        self,brain,
        lr=1e-4,
        h_size=128,
        epsilon=0.2,
        beta=1e-3,
        max_step=5e6,
        normalize=False,
        use_recurrent=False,
        num_layers=2,
        m_size=None,
        seed=0,
        stream_names=None,
    ):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: BrainInfo used to generate specific network graph.
        :param lr: Learning rate.
        :param h_size: Size of hidden layers
        :param epsilon: Value for policy-divergence threshold.
        :param beta: Strength of entropy regularization.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers Number of hidden layers between encoded input and policy & value layers
        :param m_size: Size of brain memory.
        :param seed: Seed to use for initialization of model.
        :param stream_names: List of names of value streams. Usually, a list of the Reward Signals being used.
        :return: a sub-class of PPOAgent tailored to the environment.
        """
        super().__init__()
        self.brain = brain
        self.vector_in = None
        self.global_step = 0
        self.visual_in = []
        self.use_recurrent = use_recurrent
        self.m_size = m_size if self.use_recurrent else 0
        self.normalize = normalize
        if num_layers < 1:
            num_layers = 1

        self.vis_obs_size = brain.number_visual_observations
        self.vec_obs_size = brain.vector_observation_space_size * \
            brain.num_stacked_vector_observations
        self.act_size = brain.vector_action_space_size

        self.epsilon = epsilon
        self.beta = beta
        self.lr = lr
        self.max_step = max_step

        self.stream_names = stream_names or []
        if brain.vector_action_space_type == "continuous":
            self.ac = CCActorCritic(h_size, num_layers, stream_names, brain)
        else:
            self.ac = DCActorCritic(h_size, num_layers, stream_names, brain)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def increment_step(self):
        self.global_step += 1

    def get_value_estimate(self, visual_in, vector_in):
        return self.ac.get_value_estimate(visual_in, vector_in)

    def inference(self, input_dict):
        epsilon_np = np.random.normal(size=(len(input_dict["vector_obs"]), self.act_size[0]))
        epsilon = Variable(torch.Tensor(epsilon_np))
        input_dict["random_normal_epsilon"] = epsilon
        with torch.no_grad():
            output, output_pre, all_log_probs, value_heads, value, entropy = self.ac(input_dict)

        run_out = {}
        run_out["random_normal_epsilon"] = epsilon_np
        run_out["action"] = output.data.numpy()
        run_out["pre_action"] = output_pre.data.numpy()
        run_out["log_probs"] = all_log_probs.data.numpy()
        run_out["value"] = {key:value_heads[key].data.numpy() for key in value_heads}
        run_out["entropy"] = entropy.data.numpy()
        run_out["learning_rate"] = self.lr
        return run_out

    def update(self, input_dict):
        self.optimizer.zero_grad()
        returns = {}
        old_values = {}
        for name in self.stream_names:
            returns[name] = input_dict["{}_returns".format(name)]
            old_values[name] = input_dict["{}_value_estimates".format(name)]

        masks, advantages, old_log_probs = \
            input_dict["masks"], input_dict["advantages"], input_dict["action_probs"]
        # self.poly_lr_scheduler()

        decay_epsilon = self.polynomial_decay(self.epsilon, 0.1)
        decay_beta = self.polynomial_decay(self.beta, 1e-5)

        _, _, log_probs, value_heads, _, entropy = self.ac(input_dict)
        value_losses = []
        for name in self.stream_names:
            old_value_estimates = old_values[name]
            discounted_rewards = returns[name]
            clipped_value_estimate = old_value_estimates + torch.clamp(
                torch.sum(value_heads[name], 1) - old_value_estimates, 
                -decay_epsilon, decay_epsilon)
            v_opt_a = (discounted_rewards - torch.sum(value_heads[name], 1)) ** 2
            v_opt_b = (discounted_rewards - clipped_value_estimate) ** 2
            value_loss = torch.masked_select(torch.max(v_opt_a, v_opt_b), masks).mean()
            value_losses.append(value_loss)
        value_loss = torch.stack(value_losses).mean()

        # Here we calculate PPO policy loss. In continuous control this is done independently for each action gaussian
        # and then averaged together. This provides significantly better performance than treating the probability
        # as an average of probabilities, or as a joint probability.
        log_probs = torch.sum(log_probs, 1, keepdim=True)
        old_log_probs = torch.sum(old_log_probs, 1, keepdim=True)

        r_theta = torch.exp(log_probs - old_log_probs)
        p_opt_a = r_theta * advantages
        p_opt_b = torch.clamp(r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * advantages
        policy_loss = -torch.masked_select(torch.min(p_opt_a, p_opt_b), masks).mean()

        loss = (
            policy_loss
            + 0.5 * value_loss
            - decay_beta * torch.masked_select(entropy, masks).mean()
        )

        loss.backward()
        self.optimizer.step()

        run_out = {}
        run_out["value_loss"] = value_loss.data.numpy()
        run_out["policy_loss"] = policy_loss.data.numpy()

        return run_out

    def polynomial_decay(self, value, end_value):
        global_step = min(self.global_step, self.max_step)
        decayed_value = (value - end_value) * (1 - global_step / self.max_step) + end_value
        return decayed_value

    # def poly_lr_scheduler(self):
    #     global_step = min(self.global_step, self.max_step)
    #     lr = (self.lr - 1e-10) * (1 - global_step / self.max_step) + 1e-10

    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = lr

    #     return lr
