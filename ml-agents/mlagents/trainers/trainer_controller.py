# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import logging
import shutil
import sys
from typing import *

import numpy as np
import tensorflow as tf
from time import time

from mlagents.envs import AllBrainInfo, BrainParameters
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs.exception import UnityEnvironmentException
from mlagents.envs.subprocess_environment import StepInfo
from mlagents.trainers import Trainer
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.sac.trainer import SACTrainer
from mlagents.trainers.bc.offline_trainer import OfflineBCTrainer
from mlagents.trainers.bc.online_trainer import OnlineBCTrainer
from mlagents.trainers.meta_curriculum import MetaCurriculum

import multiprocessing

from queue import Queue


class TrainerController(object):
    def __init__(
        self,
        model_path: str,
        summaries_dir: str,
        run_id: str,
        save_freq: int,
        meta_curriculum: Optional[MetaCurriculum],
        load: bool,
        train: bool,
        keep_checkpoints: int,
        lesson: Optional[int],
        external_brains: Dict[str, BrainParameters],
        training_seed: int,
        fast_simulation: bool,
    ):
        """
        :param model_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        :param load: Whether to load the model or randomly initialize.
        :param train: Whether to train model, or only run inference.
        :param keep_checkpoints: How many model checkpoints to keep.
        :param lesson: Start learning from this lesson.
        :param external_brains: dictionary of external brain names to BrainInfo objects.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        """

        self.model_path = model_path
        self.summaries_dir = summaries_dir
        self.external_brains = external_brains
        self.external_brain_names = external_brains.keys()
        self.logger = logging.getLogger("mlagents.envs")
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.load_model = load
        self.train_model = train
        self.keep_checkpoints = keep_checkpoints
        self.trainers: Dict[str, Trainer] = {}
        self.trainer_metrics: Dict[str, TrainerMetrics] = {}
        self.global_step = 0
        self.meta_curriculum = meta_curriculum
        self.seed = training_seed
        self.training_start_time = time()
        self.fast_simulation = fast_simulation
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.step_queue = multiprocessing.Queue()
        self.weights_queue = multiprocessing.Queue()

    def _get_measure_vals(self):
        if self.meta_curriculum:
            brain_names_to_measure_vals = {}
            for (
                brain_name,
                curriculum,
            ) in self.meta_curriculum.brains_to_curriculums.items():
                if curriculum.measure == "progress":
                    measure_val = (
                        self.trainers[brain_name].get_step
                        / self.trainers[brain_name].get_max_steps
                    )
                    brain_names_to_measure_vals[brain_name] = measure_val
                elif curriculum.measure == "reward":
                    measure_val = np.mean(self.trainers[brain_name].reward_buffer)
                    brain_names_to_measure_vals[brain_name] = measure_val
            return brain_names_to_measure_vals
        else:
            return None

    def _save_model(self, steps=0):
        """
        Saves current model to checkpoint folder.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info("Saved Model")

    def _save_model_when_interrupted(self, steps=0):
        self.logger.info(
            "Learning was interrupted. Please wait " "while the graph is generated."
        )
        self._save_model(steps)

    def _write_training_metrics(self):
        """
        Write all CSV metrics
        :return:
        """
        for brain_name in self.trainers.keys():
            if brain_name in self.trainer_metrics:
                self.trainers[brain_name].write_training_metrics()

    def _export_graph(self):
        """
        Exports latest saved models to .nn format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].export_model()

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException(
                "The folder {} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly.".format(model_path)
            )

    def _reset_env(self, env: BaseUnityEnvironment):
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        if self.meta_curriculum is not None:
            return env.reset(
                train_mode=self.fast_simulation,
                config=self.meta_curriculum.get_config(),
            )
        else:
            return env.reset(train_mode=self.fast_simulation)

    def create_training_process(self, trainer_config):
        # import threading
        # thread = threading.Thread(target=self.continually_train, args=())
        # thread.start()
        self.trainer_process = multiprocessing.Process(
            target=train_worker, args=(
                self.step_queue,
                self.weights_queue,
                trainer_config,
                self.external_brains,
                self.summaries_dir,
                self.model_path,
                self.run_id,
                self.keep_checkpoints,
                self.train_model,
                self.load_model,
                self.seed,
                self.meta_curriculum,
                self.save_freq
            )
        )
        self.trainer_process.start()

    def start_learning(self, env: BaseUnityEnvironment, trainer_config):
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        if self.meta_curriculum is not None:
            self.meta_curriculum.set_all_curriculums_to_lesson_num(self.lesson)
        self._create_model_path(self.model_path)

        self.create_training_process(trainer_config)

        tf.reset_default_graph()

        # Prevent a single session from taking all GPU memory.
        self.trainers, self.trainer_metrics = initialize_trainers(
            trainer_config,
            self.external_brains,
            self.summaries_dir,
            self.model_path,
            self.run_id,
            self.keep_checkpoints,
            self.train_model,
            self.load_model,
            self.seed,
            self.meta_curriculum
        )
        for _, t in self.trainers.items():
            self.logger.info(t)
        env.set_policies(
            {brain_name: t.policy for brain_name, t in self.trainers.items()}
        )

        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text("Hyperparameters", trainer.parameters)
        try:
            self._reset_env(env)
            # print("WAITING FOR INITIAL WEIGHTS")
            # initial_weights = self.weights_queue.get(block=True)
            # for brain_name, trainer in self.trainers.items():
            #     trainer.policy.set_weights(initial_weights[brain_name])
            # print("INITIAL WEIGHTS APPLIED")

            while True:
                self.advance(env)
                # for i in range(n_steps):
                #     for brain_name, trainer in self.trainers.items():
                #         # Write training statistics to Tensorboard.
                #         delta_train_start = time() - self.training_start_time
                #         if self.meta_curriculum is not None:
                #             trainer.write_summary(
                #                 self.global_step,
                #                 delta_train_start,
                #                 lesson_num=self.meta_curriculum.brains_to_curriculums[
                #                     brain_name
                #                 ].lesson_num,
                #             )
                #         else:
                #             trainer.write_summary(self.global_step, delta_train_start)
                #         if (
                #             self.train_model
                #             and trainer.get_step <= trainer.get_max_steps
                #         ):
                #             trainer.increment_step_and_update_last_reward()
                #     self.global_step += 1
                #     if (
                #         self.global_step % self.save_freq == 0
                #         and self.global_step != 0
                #         and self.train_model
                #     ):
                #         # Save Tensorflow model
                #         self._save_model(steps=self.global_step)
            # # Final save Tensorflow model
            # if self.global_step != 0 and self.train_model:
            #     self._save_model(steps=self.global_step)
        except KeyboardInterrupt:
            pass
        env.close()

    def advance(self, env: BaseUnityEnvironment):
        # if self.meta_curriculum:
        #     # Get the sizes of the reward buffers.
        #     reward_buff_sizes = {
        #         k: len(t.reward_buffer) for (k, t) in self.trainers.items()
        #     }
        #     # Attempt to increment the lessons of the brains who
        #     # were ready.
        #     lessons_incremented = self.meta_curriculum.increment_lessons(
        #         self._get_measure_vals(), reward_buff_sizes=reward_buff_sizes
        #     )
        # else:
        #     lessons_incremented = {}
        #
        # # If any lessons were incremented or the environment is
        # # ready to be reset
        # if self.meta_curriculum and any(lessons_incremented.values()):
        #     self._reset_env(env)
        #     for brain_name, trainer in self.trainers.items():
        #         trainer.end_episode()
        #     for brain_name, changed in lessons_incremented.items():
        #         if changed:
        #             self.trainers[brain_name].reward_buffer.clear()

        # time_start_step = time()
        steps: List[StepInfo] = env.step()

        for step in steps:
            self.step_queue.put(step, block=True)

        new_weights = None
        while not self.weights_queue.empty():
            new_weights = self.weights_queue.get_nowait()
        if new_weights is not None:
            for brain_name, trainer in self.trainers.items():
                trainer.policy.set_weights(new_weights[brain_name])
        return len(steps)

    # def continually_train(self):
    #     while True:
    #         while not self.step_queue.empty():
    #             step = self.step_queue.get()
    #             for brain_name, trainer in self.trainers.items():
    #                 trainer.add_experiences(
    #                     step.last_all_brain_info,
    #                     step.current_all_brain_info,
    #                     step.all_action_info[brain_name].outputs,
    #                 )
    #                 trainer.process_experiences(
    #                     step.last_all_brain_info, step.current_all_brain_info
    #                 )
    #         for brain_name, trainer in self.trainers.items():
    #             if (
    #                 trainer.is_ready_update()
    #                 and self.train_model
    #                 and trainer.get_step <= trainer.get_max_steps
    #             ):
    #                 # Perform gradient descent with experience buffer
    #                 trainer.update_policy()


def train_worker(
    step_queue: Queue,
    weights_queue: Queue,
    trainer_config: Dict[str, Dict[str, str]],
    external_brains,
    summaries_dir,
    model_path,
    run_id,
    keep_checkpoints,
    train_model,
    load_model,
    seed,
    meta_curriculum,
    save_freq
):
    training_iter_since_sync = 0
    training_start_time = time()
    print("STARTING TRAINING WORKER")
    trainers, trainer_metrics = initialize_trainers(
        trainer_config,
        external_brains,
        summaries_dir,
        model_path,
        run_id,
        keep_checkpoints,
        train_model,
        load_model,
        seed,
        meta_curriculum
    )
    # print("OUTPUTTING INITIAL WEIGHTS")
    # initial_weights = {}
    # for brain_name, trainer in trainers.items():
    #     initial_weights[brain_name] = trainer.policy.get_weights()
    # print(initial_weights)
    # weights_queue.put(initial_weights)


    global_step = 0
    try:
        while (
                any([t.get_step <= t.get_max_steps for k, t in trainers.items()])
                or not train_model
        ):
            steps = 0
            while not step_queue.empty():
                step = step_queue.get()
                steps = steps + 1
                for brain_name, trainer in trainers.items():
                    trainer.add_experiences(
                        step.last_all_brain_info,
                        step.current_all_brain_info,
                        step.all_action_info[brain_name].outputs,
                    )
                    trainer.process_experiences(
                        step.last_all_brain_info, step.current_all_brain_info
                    )
            updated = False
            for brain_name, trainer in trainers.items():
                if (
                        trainer.is_ready_update()
                        and train_model
                        and trainer.get_step <= trainer.get_max_steps
                ):
                    # Perform gradient descent with experience buffer
                    trainer.update_policy()
                    updated = True

            if updated:
                training_iter_since_sync += 1

            if training_iter_since_sync >= 10:
                training_iter_since_sync = 0
                weights = {}
                for brain_name, trainer in trainers.items():
                    weights[brain_name] = trainer.policy.get_weights()
                weights_queue.put(weights)
            for i in range(steps):
                delta_train_start = time() - training_start_time
                for brain_name, trainer in trainers.items():
                    # Write training statistics to Tensorboard.
                    # if meta_curriculum is not None:
                    #     trainer.write_summary(
                    #         global_step,
                    #         delta_train_start,
                    #         lesson_num=meta_curriculum.brains_to_curriculums[
                    #             brain_name
                    #         ].lesson_num,
                    #     )
                    # else:
                    trainer.write_summary(global_step, delta_train_start)

                    if (
                            train_model
                            and trainer.get_step <= trainer.get_max_steps
                    ):
                        trainer.increment_step_and_update_last_reward()
                global_step += 1
                if (
                        global_step % save_freq == 0
                        and global_step != 0
                        and train_model
                ):
                    # Save Tensorflow model
                    save_models(trainers)
        # Final save Tensorflow model
        if global_step != 0 and train_model:
            save_models(trainers)
    except KeyboardInterrupt:
        if train_model:
            save_model_when_interrupted(trainers)
        pass
    if train_model:
        # self._write_training_metrics()
        for brain_name in trainers.keys():
            trainers[brain_name].export_model()


def initialize_trainers(
    trainer_config: Dict[str, Dict[str, str]],
    external_brains,
    summaries_dir,
    model_path,
    run_id,
    keep_checkpoints,
    train_model,
    load_model,
    seed,
    meta_curriculum
):
    """
    Initialization of the trainers
    :param trainer_config: The configurations of the trainers
    """
    trainers = {}
    trainer_metrics = {}
    trainer_parameters_dict = {}
    for brain_name in external_brains:
        trainer_parameters = trainer_config["default"].copy()
        trainer_parameters["summary_path"] = "{basedir}/{name}".format(
            basedir=summaries_dir, name=str(run_id) + "_" + brain_name
        )
        trainer_parameters["model_path"] = "{basedir}/{name}".format(
            basedir=model_path, name=brain_name
        )
        trainer_parameters["keep_checkpoints"] = keep_checkpoints
        if brain_name in trainer_config:
            _brain_key = brain_name
            while not isinstance(trainer_config[_brain_key], dict):
                _brain_key = trainer_config[_brain_key]
            for k in trainer_config[_brain_key]:
                trainer_parameters[k] = trainer_config[_brain_key][k]
        trainer_parameters_dict[brain_name] = trainer_parameters.copy()
    for brain_name in external_brains:
        if trainer_parameters_dict[brain_name]["trainer"] == "offline_bc":
            trainers[brain_name] = OfflineBCTrainer(
                external_brains[brain_name],
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
            )
        elif trainer_parameters_dict[brain_name]["trainer"] == "online_bc":
            trainers[brain_name] = OnlineBCTrainer(
                external_brains[brain_name],
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
            )
        elif trainer_parameters_dict[brain_name]["trainer"] == "ppo":
            trainers[brain_name] = PPOTrainer(
                external_brains[brain_name],
                meta_curriculum.brains_to_curriculums[
                    brain_name
                ].min_lesson_length
                if meta_curriculum
                else 0,
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
            )
            trainer_metrics[brain_name] = trainers[
                brain_name
            ].trainer_metrics
        elif trainer_parameters_dict[brain_name]["trainer"] == "sac":
            trainers[brain_name] = SACTrainer(
                external_brains[brain_name],
                meta_curriculum.brains_to_curriculums[
                    brain_name
                ].min_lesson_length
                if meta_curriculum
                else 0,
                trainer_parameters_dict[brain_name],
                train_model,
                load_model,
                seed,
                run_id,
            )
            trainer_metrics[brain_name] = trainers[
                brain_name
            ].trainer_metrics
        else:
            raise UnityEnvironmentException(
                "The trainer config contains "
                "an unknown trainer type for "
                "brain {}".format(brain_name)
            )
        return trainers, trainer_metrics


def save_models(trainers):
    """
    Saves current model to checkpoint folder.
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    for brain_name in trainers.keys():
        trainers[brain_name].save_model()
    logging.getLogger("mlagents.envs").info("Saved Model")


def save_model_when_interrupted(trainers):
    logging.getLogger("mlagents.envs").info(
        "Learning was interrupted. Please wait " "while the graph is generated."
    )
    save_models(trainers)