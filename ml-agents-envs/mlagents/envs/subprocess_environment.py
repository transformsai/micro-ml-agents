from typing import *
import copy
import numpy as np

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
        cmd = EnvironmentCommand(name, payload)
        self.conn.send(cmd)

    def recv(self) -> EnvironmentResponse:
        response: EnvironmentResponse = self.conn.recv()
        return response

    def close(self):
        self.process.join()


def worker(parent_conn: Connection, step_queue: Queue, env_factory: Callable[[int], UnityEnvironment], worker_id: int):
    env = env_factory(worker_id)

    def _send_response(cmd_name, payload):
        parent_conn.send(
            EnvironmentResponse(cmd_name, worker_id, payload)
        )
    try:
        while True:
            cmd: EnvironmentCommand = parent_conn.recv()
            if cmd.name == 'step':
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
                all_brain_info = env.step(
                    actions, memories, texts, values
                )
                step_queue.put(EnvironmentResponse('step', worker_id, (all_brain_info, all_action_info)))
            elif cmd.name == 'external_brains':
                _send_response('external_brains', env.external_brains)
            elif cmd.name == 'reset_parameters':
                _send_response('reset_parameters', env.reset_parameters)
            elif cmd.name == 'reset':
                all_brain_info = env.reset(cmd.payload[0], cmd.payload[1])
                _send_response('reset', (all_brain_info, None))
            elif cmd.name == 'global_done':
                _send_response('global_done', env.global_done)
            elif cmd.name == 'close':
                env.close()
                break
    except KeyboardInterrupt:
        print('UnityEnvironment worker: keyboard interrupt')
    finally:
        step_queue.close()
        env.close()


class SubprocessUnityEnvironment(BaseUnityEnvironment):
    def __init__(self,
                 env_factory: Callable[[int], BaseUnityEnvironment],
                 n_env: int = 1):
        self.envs = []
        self.env_last_steps: List[EnvironmentResponse] = []
        self.last_action_infos: List[ActionInfo] = []
        self.env_agent_counts = {}
        self.waiting: List[bool] = [False] * n_env
        self.policies: Dict[str, Policy] = {}
        self.step_queue = Queue()
        self._cached_external_brains = None
        for worker_id in range(n_env):
            self.envs.append(self.create_worker(worker_id, self.step_queue, env_factory))
            self.env_last_steps.append(None)
            self.last_action_infos.append(None)

    def set_policies(self, policies: Dict[str, Policy]):
        self.policies = policies

    @staticmethod
    def create_worker(
            worker_id: int,
            step_queue: Queue,
            env_factory: Callable[[int], BaseUnityEnvironment]
    ) -> UnityEnvWorker:
        parent_conn, child_conn = Pipe()
        child_process = Process(target=worker, args=(child_conn, step_queue, env_factory, worker_id))
        child_process.start()
        return UnityEnvWorker(child_process, worker_id, parent_conn)

    def _get_action_for_envs(self, env_ids):
        env_action_infos = {}
        for env_id in env_ids:
            env_action_infos[env_id] = {}
            last_step = self.env_last_steps[env_id]
            last_all_brain_info = last_step.payload[0]
            for brain_name in last_all_brain_info:
                action_info = self.policies[brain_name].get_action(last_all_brain_info[brain_name])
                env_action_infos[env_id][brain_name] = action_info
        return env_action_infos

    def _queue_steps(self):
        workers_to_step = []
        for worker_id, is_waiting in enumerate(self.waiting):
            if not is_waiting:
                workers_to_step.append(worker_id)
        if len(workers_to_step) == 0:
            return

        env_action_infos = self._get_action_for_envs(workers_to_step)
        for worker_id, action_info in env_action_infos.items():
            self.envs[worker_id].send('step', action_info)
            self.waiting[worker_id] = True

    def step(self, vector_action=None, memory=None, text_action=None, value=None) -> List[StepInfo]:
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

        step_infos = []
        for step in worker_steps:
            step_infos.append(StepInfo(
                self.env_last_steps[step.worker_id].payload[0],
                step.payload[0],
                step.payload[1]
            ))
            self.env_last_steps[step.worker_id] = step
        self._queue_steps()

        return step_infos

    def reset(self, config=None, train_mode=True) -> AllBrainInfo:
        self._broadcast_message('reset', (config, train_mode))
        self.env_last_steps = [self.envs[i].recv() for i in range(len(self.envs))]

        return self._merge_step_info(self.env_last_steps)

    @property
    def global_done(self):
        self._broadcast_message('global_done')
        dones: List[EnvironmentResponse] = [
            self.envs[i].recv().payload for i in range(len(self.envs))
        ]
        return all(dones)

    @property
    def external_brains(self):
        if self._cached_external_brains is None:
            self.envs[0].send('external_brains')
            self._cached_external_brains = self.envs[0].recv().payload
        return self._cached_external_brains

    @property
    def reset_parameters(self):
        self.envs[0].send('reset_parameters')
        return self.envs[0].recv().payload

    def close(self):
        self.step_queue.close()
        self.step_queue.join_thread()
        for env in self.envs:
            env.close()

    def _get_agent_counts(self, step_list: Iterable[AllBrainInfo]):
        for i, step in enumerate(step_list):
            for brain_name, brain_info in step.items():
                if brain_name not in self.env_agent_counts.keys():
                    self.env_agent_counts[brain_name] = [0] * len(self.envs)
                self.env_agent_counts[brain_name][i] = len(brain_info.agents)

    @staticmethod
    def _merge_step_info(env_steps: List[EnvironmentResponse]) -> AllBrainInfo:
        accumulated_brain_info: AllBrainInfo = None
        for env_step in env_steps:
            all_brain_info: AllBrainInfo = env_step.payload[0]
            for brain_name, brain_info in all_brain_info.items():
                for i in range(len(brain_info.agents)):
                    if not isinstance(brain_info.agents[i], str):
                        brain_info.agents[i] = str(env_step.worker_id) + '-' + str(brain_info.agents[i])
                if accumulated_brain_info:
                    accumulated_brain_info[brain_name].merge(brain_info)
            if not accumulated_brain_info:
                accumulated_brain_info = copy.deepcopy(all_brain_info)
        return accumulated_brain_info

    def _broadcast_message(self, name: str, payload = None):
        for env in self.envs:
            env.send(name, payload)