from typing import *
import copy
import numpy as np
import cloudpickle

from mlagents.envs import UnityEnvironment
from multiprocessing import Process, Pipe, Queue
from multiprocessing.queues import Empty as EmptyQueue
from multiprocessing.connection import Connection
from mlagents.envs.base_unity_environment import BaseUnityEnvironment
from mlagents.envs import AllBrainInfo, UnityEnvironmentException
from mlagents.trainers import Policy, ActionInfo


class EnvironmentCommand(NamedTuple):
    name: str
    payload: Any = None


class EnvironmentResponse(NamedTuple):
    name: str
    worker_id: int
    payload: Any


class StepInfo(NamedTuple):
    last_all_brain_info: AllBrainInfo
    current_all_brain_info: AllBrainInfo
    all_action_info: Dict[str, ActionInfo]


class UnityEnvWorker(NamedTuple):
    process: Process
    worker_id: int
    conn: Connection

    def send(self, name: str, payload=None):
        try:
            cmd = EnvironmentCommand(name, payload)
            self.conn.send(cmd)
        except (BrokenPipeError, EOFError):
            raise KeyboardInterrupt

    def recv(self) -> EnvironmentResponse:
        try:
            response: EnvironmentResponse = self.conn.recv()
            return response
        except (BrokenPipeError, EOFError):
            raise KeyboardInterrupt

    def close(self):
        try:
            self.conn.send(EnvironmentCommand("close"))
        except (BrokenPipeError, EOFError):
            pass
        self.process.join()


def worker(
    parent_conn: Connection, step_queue: Queue, pickled_env_factory: str, worker_id: int
):
    env_factory: Callable[[int], UnityEnvironment] = cloudpickle.loads(
        pickled_env_factory
    )
    env = env_factory(worker_id)

    def _send_response(cmd_name, payload):
        parent_conn.send(EnvironmentResponse(cmd_name, worker_id, payload))

    try:
        while True:
            cmd: EnvironmentCommand = parent_conn.recv()
            if cmd.name == "step":
                all_action_info = cmd.payload
                actions = {}
                memories = {}
                texts = {}
                values = {}
                outputs = {}
                for brain_name, action_info in all_action_info.items():
                    actions[brain_name] = action_info.action
                    memories[brain_name] = action_info.memory
                    texts[brain_name] = action_info.text
                    values[brain_name] = action_info.value
                    outputs[brain_name] = action_info.outputs
                all_brain_info = env.step(actions, memories, texts, values)
                step_queue.put(
                    EnvironmentResponse(
                        "step", worker_id, (all_brain_info, all_action_info)
                    )
                )
            elif cmd.name == "external_brains":
                _send_response("external_brains", env.external_brains)
            elif cmd.name == "reset_parameters":
                _send_response("reset_parameters", env.reset_parameters)
            elif cmd.name == "reset":
                all_brain_info = env.reset(cmd.payload[0], cmd.payload[1])
                _send_response("reset", (all_brain_info, None))
            elif cmd.name == "global_done":
                _send_response("global_done", env.global_done)
            elif cmd.name == "close":
                break
    except KeyboardInterrupt:
        print("UnityEnvironment worker: keyboard interrupt")
    finally:
        step_queue.close()
        env.close()


class SubprocessEnvironmentManager:
    def __init__(
        self, env_factory: Callable[[int], BaseUnityEnvironment], n_env: int = 1
    ):
        self.envs = []
        self.env_last_steps: List[EnvironmentResponse] = []
        self.last_action_infos: List[ActionInfo] = []
        self.env_agent_counts = {}
        self.waiting: List[bool] = [False] * n_env
        self.policies: Dict[str, Policy] = {}
        self.step_queue = Queue()
        self._cached_external_brains = None
        for worker_id in range(n_env):
            self.envs.append(
                self.create_worker(worker_id, self.step_queue, env_factory)
            )
            self.env_last_steps.append(None)
            self.last_action_infos.append(None)

    def set_policies(self, policies: Dict[str, Policy]):
        self.policies = policies

    @staticmethod
    def create_worker(
        worker_id: int,
        step_queue: Queue,
        env_factory: Callable[[int], BaseUnityEnvironment],
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()

        # Need to use cloudpickle for the env factory function since function objects aren't picklable
        # on Windows as of Python 3.6.
        pickled_env_factory = cloudpickle.dumps(env_factory)
        child_process = Process(
            target=worker, args=(child_conn, step_queue, pickled_env_factory, worker_id)
        )
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def _get_action_for_env(self, env_id):
        env_action_info = {}
        last_step = self.env_last_steps[env_id]
        last_all_brain_info = last_step.payload[0]
        for brain_name in last_all_brain_info:
            action_info = self.policies[brain_name].get_action(
                last_all_brain_info[brain_name]
            )
            env_action_info[brain_name] = action_info
        return env_action_info

    def _queue_steps(self):
        for worker_id, is_waiting in enumerate(self.waiting):
            if not is_waiting:
                env_action_info = self._get_action_for_env(worker_id)
                self.envs[worker_id].send("step", env_action_info)
                self.waiting[worker_id] = True

    def step(self) -> List[StepInfo]:
        self._queue_steps()

        worker_steps = []
        step_workers = set()
        while len(worker_steps) < 1:
            steps_to_requeue = []
            try:
                while True:
                    step = self.step_queue.get_nowait()
                    self.waiting[step.worker_id] = False
                    if step.worker_id not in step_workers:
                        worker_steps.append(step)
                        step_workers.add(step.worker_id)
                    else:
                        steps_to_requeue.append(step)
            except EmptyQueue:
                pass
            finally:
                for step in steps_to_requeue:
                    self.step_queue.put(step)

        step_infos = self._postprocess_steps(worker_steps)
        return step_infos

    def reset(self, config=None, train_mode=True) -> List[StepInfo]:
        for env in self.envs:
            env.send("reset", (config, train_mode))
        self.env_last_steps = [self.envs[i].recv() for i in range(len(self.envs))]
        return [StepInfo(step[0], None, None) for step in self.env_last_steps]

    @property
    def external_brains(self):
        self.envs[0].send("external_brains")
        return self.envs[0].recv().payload

    def close(self):
        self.step_queue.close()
        self.step_queue.join_thread()
        for env in self.envs:
            env.close()

    def _postprocess_steps(self, env_steps: List[EnvironmentResponse]) -> List[StepInfo]:
        step_infos = []
        for step in env_steps:
            all_brain_info: AllBrainInfo = step.payload[0]
            for brain_name, brain_info in all_brain_info.items():
                for i in range(len(brain_info.agents)):
                    if not isinstance(brain_info.agents[i], str):
                        brain_info.agents[i] = (
                            str(step.worker_id) + "-" + str(brain_info.agents[i])
                        )
            step_infos.append(
                StepInfo(
                    self.env_last_steps[step.worker_id].payload[0],
                    step.payload[0],
                    step.payload[1],
                )
            )
            self.env_last_steps[step.worker_id] = step
        return step_infos
