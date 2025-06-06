import torch

class GAE:
    """Generalized Advantage Estimator."""

    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(self, trajectory, last_observation):
        """
        This method should modify trajectory inplace by adding
        items with keys 'advantages' and 'value_targets' to it

        input:
            trajectory - dict from runner
            latest_observation - last state, numpy, (features)
        """
        value_target = self.policy.act(last_observation)["values"]

        env_steps = len(trajectory["rewards"])

        trajectory["rewards"] = torch.tensor(trajectory["rewards"], dtype=torch.float32)
        trajectory["dones"] = torch.tensor(trajectory["dones"], dtype=torch.float32)
        trajectory["values"] = torch.tensor(trajectory["values"], dtype=torch.float32)

        is_not_done = 1 - trajectory["dones"]

        advantages = []
        value_targets = []

        gae = 0.0
        for step in reversed(range(env_steps)):
            if step == env_steps - 1:
                next_value = torch.tensor(value_target)
            else:
                next_value = trajectory["values"][step + 1]

            delta = (
                trajectory["rewards"][step]
                + self.gamma * next_value * is_not_done[step]
                - trajectory["values"][step]
            )

            gae = delta + self.gamma * self.lambda_ * is_not_done[step] * gae

            advantages.append(gae)
            value_targets.append(gae + trajectory["values"][step])

        trajectory["advantages"] = torch.tensor(
            list(reversed(advantages)),
            dtype=torch.float32,
        )
        trajectory["value_targets"] = torch.tensor(
            list(reversed(value_targets)),
            dtype=torch.float32,
        )