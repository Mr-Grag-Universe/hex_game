""" RL env runner """
from collections import defaultdict
import numpy as np

class GameRunner:
    """ Reinforcement learning runner in an environment with given policy """
    def __init__(self, game, policy, nsteps, transforms=None, step_var=None):
        self.game = game
        self.policy = policy
        self.nsteps = nsteps
        self.transforms = transforms or [[], []]
        self.step_var = step_var if step_var is not None else 0
        self.device = next(self.policy[0].model.parameters()).device

        self.t = self.game.team_counter.get()
        self.latest_observation = self.game.reset()

    def reset(self):
        """ Resets env and runner states. """
        self.latest_observation = self.game.reset()
        self.policy[self.t].reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = [defaultdict(list, {"actions": []}) for _ in range(2)]
        observations = [[], []]
        rewards = [[], []]
        dones = [[], []]
        
        self.t = self.game.team_counter.get()
        for i in range(self.nsteps):
            observations[self.t].append(self.latest_observation)
            # print("acting...")
            act = self.policy[self.t].act(self.latest_observation)
            
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            assert type(act["actions"]) is np.ndarray, "Error: actions for environment must be numpy"
            
            for key, val in act.items():
                trajectory[self.t][key].append(val)

            r, obs, done = self.game.step(act['dist'].decode_action(trajectory[self.t]["actions"][-1].argmax()))
            self.latest_observation = obs
            rewards[self.t].append(r)
            dones[self.t].append(done)
            self.step_var += 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if done:
                # print("reset")
                self.latest_observation = self.game.reset()
            self.t = self.game.team_counter.get()

        for i in range(2):
            trajectory[i].update(
                observations=observations[i],
                rewards=rewards[i],
                dones=dones[i])

            for transform in self.transforms[i]:
                transform(trajectory[i], self.latest_observation)
        
        return trajectory[0]