""" RL env runner """
from collections import defaultdict
import numpy as np
import torch

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
        self.memory = (torch.zeros(128, dtype=torch.float32, device=self.device), torch.zeros(128, dtype=torch.float32, device=self.device))

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
        mem_1 = [[], []]
        mem_2 = [[], []]
        
        self.t = self.game.team_counter.get()
        for i in range(self.nsteps):
            observations[self.t].append(self.latest_observation)
            # print(len(self.memory))
            act = self.policy[self.t].act(self.latest_observation, mem_1=self.memory[0], mem_2=self.memory[1])
            
            if "actions" not in act:
                raise ValueError("result of policy.act must contain 'actions' "
                                 f"but has keys {list(act.keys())}")
            assert type(act["actions"]) is np.ndarray, "Error: actions for environment must be numpy"
            
            for key, val in act.items():
                if key != 'memory':
                    trajectory[self.t][key].append(val)
            self.memory = act['memory']

            r, obs, done = self.game.step(act['dist'].decode_action(trajectory[self.t]["actions"][-1].argmax()))
            self.latest_observation = obs
            rewards[self.t].append(r)
            dones[self.t].append(done)
            mem_1[self.t].append(self.memory[0].cpu().detach().numpy())
            mem_2[self.t].append(self.memory[1].cpu().detach().numpy())
            self.step_var += 1

            # Only reset if the env is not batched. Batched envs should
            # auto-reset.
            if done:
                # print("reset")
                self.latest_observation = self.game.reset()
                self.memory = (torch.zeros(128, dtype=torch.float32, device=self.device), torch.zeros(128, dtype=torch.float32, device=self.device))
            self.t = self.game.team_counter.get()

        for i in range(2):
            trajectory[i].update(
                observations=observations[i],
                rewards=rewards[i],
                dones=dones[i],
                mem_1=mem_1[i],
                mem_2=mem_2[i])

            for transform in self.transforms[i]:
                transform(trajectory[i], self.latest_observation, self.memory)
        
        return trajectory[0]