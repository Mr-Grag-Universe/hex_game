""" RL env runner """
from collections import defaultdict
import numpy as np

class GameRunner:
    """ Reinforcement learning runner in an environment with given policy """
    def __init__(self, game, policy, nsteps, transforms=None, step_var=None):
        self.game = game
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or []
        self.step_var = step_var if step_var is not None else 0
        self.latest_observation = self.game.reset()

    def reset(self):
        """ Resets env and runner states. """
        self.latest_observation = self.game.reset()
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        dones = []
        
        for i in range(self.nsteps):
            observations.append(self.latest_observation)
            # print("acting...")
            act = self.policy.act(self.latest_observation)
            
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            assert type(act["actions"]) is np.ndarray, "Error: actions for environment must be numpy"
            
            for key, val in act.items():
                trajectory[key].append(val)

            r, obs, done = self.game.step(act['dist'].decode_action(trajectory["actions"][-1].argmax()))
            self.latest_observation = obs
            rewards.append(r)
            dones.append(done)
            self.step_var += 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if np.all(done):
                self.latest_observation = self.game.reset()

        trajectory.update(
            observations=observations,
            rewards=rewards,
            dones=dones)

        for transform in self.transforms:
            transform(trajectory, self.latest_observation)
        
        return trajectory