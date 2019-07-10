import logging
from typing import Any, Dict

import torch

from mlagents.trainers import ActionInfo, UnityException
from mlagents.envs import BrainInfo


logger = logging.getLogger("mlagents.trainers")


class Policy(object):
    """
    Contains a learning model, and the necessary
    functions to interact with it to perform evaluate and updating.
    """
    def __init__(self, seed, brain, trainer_parameters):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param brain: The corresponding Brain for this policy.
        :param trainer_parameters: The trainer parameters.
        """
        self.m_size = None
        self.model = None
        self.inference_dict = {}
        self.update_dict = {}
        self.sequence_length = 1
        self.seed = seed
        self.brain = brain
        self.use_recurrent = trainer_parameters["use_recurrent"]
        self.use_continuous_act = brain.vector_action_space_type == "continuous"
        self.model_path = trainer_parameters["model_path"]
        self.keep_checkpoints = trainer_parameters.get("keep_checkpoints", 5)
        if self.use_recurrent:
            self.m_size = trainer_parameters["memory_size"]
            self.sequence_length = trainer_parameters["sequence_length"]
            if self.m_size == 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is 0 even "
                    "though the trainer uses recurrent.".format(brain.brain_name)
                )
            elif self.m_size % 4 != 0:
                raise UnityPolicyException(
                    "The memory size for brain {0} is {1} "
                    "but it must be divisible by 4.".format(
                        brain.brain_name, self.m_size
                    )
                )

    def _load_graph(self):
        logger.info("Loading Model for brain {}".format(self.brain.brain_name))
        if not os.path.exists(self.model_path):
            logger.info(
                "The model {0} could not be found. Make "
                "sure you specified the right "
                "--run-id".format(self.model_path)
            )
        self.model = torch.load(self.model_path)

    def evaluate(self, brain_info: BrainInfo) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param brain_info: BrainInfo input to network.
        :return: Output from policy based on self.inference_dict.
        """
        raise UnityPolicyException("The evaluate function was not implemented.")

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
        return ActionInfo(
            action=run_out.get("action"),
            memory=run_out.get("memory_out"),
            text=None,
            value=run_out.get("value"),
            outputs=run_out,
        )

    def update(self, mini_batch, num_sequences):
        """
        Performs update of the policy.
        :param num_sequences: Number of experience trajectories in batch.
        :param mini_batch: Batch of experiences.
        :return: Results of update.
        """
        raise UnityPolicyException("The update function was not implemented.")

    def make_empty_memory(self, num_agents):
        """
        Creates empty memory for use with RNNs
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.m_size))

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        return self.model.global_step

    def increment_step(self):
        """
        Increments model step.
        """
        self.model.increment_step()

    def save_model(self, steps):
        """
        Saves the model
        :param steps: The number of steps the model was trained for
        :return:
        """
        torch.save(self.model, self.model_path)

    def export_model(self):
        """
        Exports latest saved model to .nn format for Unity embedding.
        """
        logger.info("export_model to be implemented")

    @property
    def vis_obs_size(self):
        return self.model.vis_obs_size

    @property
    def vec_obs_size(self):
        return self.model.vec_obs_size

    @property
    def use_vis_obs(self):
        return self.model.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.model.vec_obs_size > 0
