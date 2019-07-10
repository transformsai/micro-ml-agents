import logging
import numpy as np
import torch
from torch.autograd import Variable

from mlagents.trainers import BrainInfo, ActionInfo
from mlagents.trainers.ppo.models import PPOModel
from mlagents.trainers.policy import Policy
from mlagents.trainers.components.reward_signals.reward_signal_factory import (
    create_reward_signal,
)

logger = logging.getLogger("mlagents.trainers")


class PPOPolicy(Policy):
    def __init__(self, seed, brain, trainer_params, is_training, load):
        """
        Policy for Proximal Policy Optimization Networks.
        :param seed: Random seed.
        :param brain: Assigned Brain object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        """
        super().__init__(seed, brain, trainer_params)

        reward_signal_configs = trainer_params["reward_signals"]

        self.reward_signals = {}
        self.model = PPOModel(
            brain,
            lr=float(trainer_params["learning_rate"]),
            h_size=int(trainer_params["hidden_units"]),
            epsilon=float(trainer_params["epsilon"]),
            beta=float(trainer_params["beta"]),
            max_step=float(trainer_params["max_steps"]),
            normalize=trainer_params["normalize"],
            use_recurrent=trainer_params["use_recurrent"],
            num_layers=int(trainer_params["num_layers"]),
            m_size=self.m_size,
            seed=seed,
            stream_names=list(trainer_params["reward_signals"].keys()),
        )

        # Create reward signals
        for reward_signal, config in reward_signal_configs.items():
            self.reward_signals[reward_signal] = create_reward_signal(
                self, reward_signal, config
            )

        if load:
            self._load_graph()

    def evaluate(self, brain_info):
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo object containing inputs.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {}
        feed_dict["action_masks"] = Variable(torch.ByteTensor(brain_info.action_masks))
        feed_dict["visual_obs"] = Variable(torch.Tensor(brain_info.visual_observations))
        feed_dict["vector_obs"] = Variable(torch.Tensor(brain_info.vector_observations))
        # if self.use_recurrent:
        #     to be implement

        self.model.eval()
        run_out = self.model.inference(feed_dict)
        return run_out

    def update(self, mini_batch, num_sequences):
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param mini_batch: Experience batch.
        :return: Output from update process.
        """
        mini_batch["advantages"] = mini_batch["advantages"].reshape(-1, 1)
        for key in mini_batch:
            if key == "masks":
                mini_batch[key] = Variable(torch.ByteTensor(mini_batch[key]))
            else:
                mini_batch[key] = Variable(torch.Tensor(mini_batch[key]))
        # if self.use_recurrent:
        #     to be implement

        self.model.train()
        run_out = self.model.update(mini_batch)
        return run_out

    def get_value_estimates(self, brain_info, idx):
        """
        Generates value estimates for bootstrapping.
        :param brain_info: BrainInfo to be used for bootstrapping.
        :param idx: Index in BrainInfo of agent.
        :return: The value estimate dictionary with key being the name of the reward signal and the value the
        corresponding value estimate.
        """
        visual_in = [brain_info.visual_observations[i][idx] \
            for i in range(len(brain_info.visual_observations))]
        visual_in = Variable(torch.Tensor(visual_in))
        vector_in = [brain_info.vector_observations[idx]] if self.use_vec_obs else None
        vector_in = Variable(torch.Tensor(vector_in))
        # if self.use_recurrent:
        #     to be implement

        self.model.eval()
        value_estimate = self.model.get_value_estimate(visual_in, vector_in)
        return value_estimate

    def get_action(self, brain_info: BrainInfo) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param brain_info: A dictionary of brain names and BrainInfo from environment.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(brain_info.agents) == 0:
            return ActionInfo([], [], [], None, None)

        run_out = self.evaluate(brain_info)
        mean_values = np.mean(
            np.array(list(run_out.get("value").values())), axis=0
        ).flatten()

        return ActionInfo(
            action=run_out.get("action"),
            memory=run_out.get("memory_out"),
            text=None,
            value=mean_values,
            outputs=run_out,
        )
