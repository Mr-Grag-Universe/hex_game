import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_

from .GAE import GAE
from .GameRunner import GameRunner

class AsArray:
    """
    Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory, last_observation, *args, **kwargs):
        # Modifies trajectory inplace.
        # Just switches python lists to numpy arrays
        for k, v in trajectory.items():
            trajectory[k] = np.asarray(v)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, inputs, training=False):
        """
        input:
            inputs - numpy array if training is False, otherwise tensor, (batch_size x features)
            training - flag, bool
        output:
            if training is True, dict containing keys ['actions', 'log_probs', 'values']:
                `distribution` - MultivariateNormal, (batch_size x actions_dim)
                'values' - critic estimations, tensor, (batch_size)
            if training is False, dict containing keys ['actions', 'log_probs', 'values']:
                'actions' - selected actions, numpy, (batch_size)
                'log_probs' - log probs of selected actions, numpy, (batch_size)
                'values' - critic estimations, numpy, (batch_size)
        """
        # if training is false, input is numpy
        if not training:
            inputs = torch.FloatTensor(inputs).to(DEVICE)

        dist, values = self.model(inputs)
        actions = dist.sample()

        if training:
            return {
                "distribution": dist,
                "values": values,
            }

        else:
            log_probs = dist.log_prob(actions)
            return {
                "actions": actions.cpu().detach().numpy(),
                "log_probs": log_probs.cpu().detach().numpy(),
                "values": values.cpu().detach().numpy(),
            }

class Sampler:
    """Samples minibatches from trajectory for a number of epochs."""

    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []

    def get_next(self):
        """
        Yields next minibatch (dict) for training with at least following keys:
                'observations' - numpy, (batch_size x features)
                'actions' - numpy, (batch_size x actions_dim)
                'advantages' - numpy, (batch_size)
                'log_probs' - numpy, (batch_size)
        """
        trajectory = self.runner.get_next()

        for epoch in range(self.num_epochs):
            # shuffle dataset and separate it into minibatches
            # you can use any standard utils to do that
            num_samples = len(trajectory["actions"])

            indices = np.random.permutation(num_samples)
            batch_indices = np.array_split(indices, self.num_minibatches)

            for mb_indices in batch_indices:
                minibatch = {
                    key: value[mb_indices] for key, value in trajectory.items()
                }

                # applying additional transforms
                for transform in self.transforms:
                    transform(minibatch)

                yield minibatch

class NormalizeAdvantages:
    """Normalizes advantages to have zero mean and variance 1."""
    def __call__(self, batch):
        adv = batch["advantages"]
        # adv = (adv - adv.mean()) / (adv.std() + 1e-7)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            adv = adv - adv.mean()
        batch["advantages"] = adv

class PyTorchify:
    """Moves everything to PyTorch"""
    def __call__(self, batch):
        for k, v in batch.items():
            try:
                batch[k] = torch.FloatTensor(v).to(DEVICE)
            except:
                # print(v, batch[k])
                pass

def make_ppo_sampler(
    env,
    policy_1, policy_2,
    num_runner_steps=2048,
    gamma=0.99,
    lambda_=0.95,
    num_epochs=10,
    num_minibatches=32,
):
    """Creates runner for PPO algorithm."""
    runner_transforms = [[AsArray(), GAE(policy_1, gamma=gamma, lambda_=lambda_)], [AsArray(), GAE(policy_2, gamma=gamma, lambda_=lambda_)]]
    runner = GameRunner(env, [policy_1, policy_2], num_runner_steps, transforms=runner_transforms)

    sampler_transforms = [NormalizeAdvantages(), PyTorchify()]
    sampler = Sampler(
        runner,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        transforms=sampler_transforms,
    )

    return sampler

class PPO:
    def __init__(
        self,
        policy,
        optimizer,
        sampler,
        cliprange=0.2,
        value_loss_coef=0.25,
        max_grad_norm=0.5,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.sampler = sampler
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.iteration = 0

    def policy_loss(self, batch, act):
        """
        Computes and returns policy loss on a given minibatch.
        input:
            batch - dict from sampler, containing:
                'advantages' - advantage estimation, tensor, (batch_size)
                'actions' - actions selected in real trajectory, (batch_size)
                'log_probs' - probabilities of actions from policy used to collect this trajectory, (batch_size)
            act - dict from your current policy, containing:
                'distribution' - MultivariateNormal, (batch_size x actions_dim)
        output:
            policy loss - torch scalar
        """
        log_probs_all = act["dist"].log_prob(batch["actions"])
        log_old_probs_all = batch["log_probs"]
        ratio = torch.exp(log_probs_all - log_old_probs_all.detach())

        advantages = batch["advantages"].detach()
        J_pi = ratio * advantages
        J_pi_clipped = (
            torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange) * advantages
        )

        policy_loss = -torch.mean(torch.min(J_pi, J_pi_clipped))

        entropy = act["dist"].entropy().mean().item()
        clip_frac = ((ratio - 1).abs() > self.cliprange).float().mean().item()
        max_ratio = ratio.max().item()

        return policy_loss

    def value_loss(self, batch, act):
        """
        Computes and returns policy loss on a given minibatch.
        input:
            batch - dict from sampler, containing:
                'value_targets' - computed targets for critic, (batch_size)
                'values' - critic estimation from network that generated trajectory, (batch_size)
            act - dict from your current policy, containing:
                'values' - current critic estimation, tensor, (batch_size)
        output:
            critic loss - torch scalar
        """
        # print(batch["value_targets"].shape, act["values"].shape)
        assert batch["value_targets"].shape == act["values"].shape, (
            "Danger: your values and value targets have different shape. Watch your broadcasting!"
        )

        values_pred = act["values"]
        values_target = batch["value_targets"].detach()
        values_old = batch["values"]

        L_simple = (values_pred - values_target) ** 2

        delta = values_pred - values_old
        delta_clipped = torch.clamp(delta, -self.cliprange, self.cliprange)
        values_clipped = values_old + delta_clipped
        L_clipped = (values_clipped - values_target) ** 2

        value_loss = torch.mean(torch.max(L_simple, L_clipped))

        avg_value = values_pred.mean().item()
        clipped_frac = (delta.abs() > self.cliprange).float().mean().item()

        return value_loss

    def loss(self, batch):
        """Computes loss for current batch"""

        # let's run our current policy on this batch
        act = self.policy.act(batch["observations"], mem_1=batch["mem_1"], mem_2=batch["mem_2"], training=True)

        # compute losses
        # note that we don't need entropy regularization for this env.
        policy_loss = self.policy_loss(batch, act)
        value_loss = self.value_loss(batch, act)
        total_loss = policy_loss + self.value_loss_coef * value_loss

        # Return scalar loss
        return total_loss

    def step(self, batch):
        """Computes the loss function and performs a single gradient step for this batch."""
        self.optimizer.zero_grad()

        loss = self.loss(batch)
        loss.backward()

        grad_norm = clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # do not forget to clip gradients using self.max_grad_norm
        # and log gradient norm

        # this is for logging
        self.iteration += 1
        return loss