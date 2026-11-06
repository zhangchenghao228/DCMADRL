"""Synthetic regression tests; no EnergyPlus or training-performance claims.

Run from the repository root: python -m unittest discover -s DCMADRL/tests -v
Set DCMADRL_TEST_DEVICE=cuda to also validate GPU updates.
"""
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlkit import pytorch_util as ptu
from maddpg import actor_critic, maddpg


class DistributionalCostTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get('DCMADRL_TEST_DEVICE', 'cpu'))
        ptu.device = self.device
        actor_critic.device = maddpg.device = self.device
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(
            save_dir=self.temp.name, scenario_name='distributional_cost_test',
            lr_actor=1e-4, lr_critic=1e-3, lr_alpha=1e-3, lagrangian_lr=1e-3,
            gamma=0.95, tau=0.01, save_rate=5000, policy_noise=0.1,
            noise_clip=0.1, max_action=1, policy_update_freq=2)
        self.net = maddpg.COST_CRITIC_NET(self.args, 0)
        self.o = [torch.randn(4, 11, device=self.device) for _ in range(5)]
        self.u = [torch.randn(4, 2, device=self.device) for _ in range(5)]

    def test_joint_inputs_and_quantile_dependence(self):
        o = [x.clone().requires_grad_() for x in self.o]
        u = [x.clone().requires_grad_() for x in self.u]
        tau = torch.linspace(0.01, 0.99, 32, device=self.device).repeat(4, 1)
        z = self.net.critic_network1(o, u, tau)
        self.assertEqual(z.shape, (4, 32))
        self.assertGreater(z.std(dim=1).min().item(), 0)
        z.sum().backward()
        for x in o + u:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertFalse(hasattr(self.net, 'critic_network2'))

    def test_quantile_loss_against_hand_calculation(self):
        # residuals [[1, 4], [-1, 2]], Huber [[.5, 3.5], [.5, 1.5]]
        prediction = torch.tensor([[0., 2.]], device=self.device, requires_grad=True)
        target = torch.tensor([[1., 4.]], device=self.device, requires_grad=True)
        tau = torch.tensor([[.25, .75]], device=self.device)
        weights = torch.tensor([[.2, .8]], device=self.device)
        loss = self.net.quantile_regression_loss(prediction, target, tau, weights)
        expected = (.25 * .5 * .2 + .25 * 3.5 * .8
                    + .25 * .5 * .2 + .75 * 1.5 * .8) / 2
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_distributional_target_update_and_soft_update(self):
        old = [p.detach().clone() for p in self.net.critic_network1.parameters()]
        cost = torch.arange(4., device=self.device)
        next_z = torch.linspace(-1, 2, 32, device=self.device).repeat(4, 1)
        with patch.object(self.net.critic_target_network1, 'forward', return_value=next_z), \
                patch.object(self.net, 'quantile_regression_loss',
                             wraps=self.net.quantile_regression_loss) as loss:
            self.net.upda(self.o, self.u, self.o, self.u, cost, None)
        target = loss.call_args.args[1]
        torch.testing.assert_close(target, cost[:, None] + .95 * next_z)
        self.assertFalse(target.requires_grad)
        online = list(self.net.critic_network1.parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, online)))
        for before, after, target_param in zip(old, online, self.net.critic_target_network1.parameters()):
            torch.testing.assert_close(target_param, .99 * before + .01 * after)
            self.assertIsNone(target_param.grad)
        self.assertTrue(math.isfinite(self.net.critic_loss1))

    def test_weighted_expectation_and_constraint_gradient(self):
        tau, weights = self.net.get_tau(self.o[0])
        torch.testing.assert_close(weights.sum(1), torch.ones(4, device=self.device))
        self.assertTrue(((tau > 0) & (tau < 1)).all())
        with patch.object(self.net, 'get_tau', return_value=(tau, weights)):
            z = self.net.critic_network1(self.o, self.u, tau)
            torch.testing.assert_close(self.net.expected_cost(self.o, self.u),
                                       (weights * z).sum(1, keepdim=True))
            u = [x.clone().requires_grad_() for x in self.u]
            # Activate the existing penalty so its action gradient is observable.
            loss = self.net.actor_loss(self.o, u, cost_limit=-100.)
            loss.backward()
            self.assertGreater(u[0].grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(u[0].grad).all())

    def test_checkpoint_round_trip_and_reject_scalar_weights(self):
        self.net.save_model(self.args.save_rate)
        filename = Path(self.net.model_path) / '1cost_critic1_params.pkl'
        clone = actor_critic.Cost_critic().to(self.device)
        clone.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))
        tau, _ = self.net.get_tau(self.o[0])
        torch.testing.assert_close(clone(self.o, self.u, tau),
                                   self.net.critic_network1(self.o, self.u, tau))
        legacy = actor_critic.Critic(n_agents=5).state_dict()
        with self.assertRaises(RuntimeError):
            clone.load_state_dict(legacy)

    def test_all_five_agents_complete_actual_training_update(self):
        from agent import Agent
        agents = [Agent(i, self.args, -2) for i in range(5)]
        reward = [maddpg.Quantile_net(self.args, i) for i in range(5)]
        costs = [maddpg.COST_CRITIC_NET(self.args, i) for i in range(5)]
        transitions = {}
        for i in range(5):
            transitions.update({
                f'o_{i}': self.o[i], f'u_{i}': self.u[i],
                f'o_next_{i}': torch.randn_like(self.o[i]),
                f'r_{i}': -torch.rand(4, device=self.device),
                f'c_{i}': torch.rand(4, device=self.device)})
        for i, agent in enumerate(agents):
            before = [p.detach().clone() for p in agent.policy.actor_network.parameters()]
            agent.learn(dict(transitions), agents[:i] + agents[i + 1:], reward[i], costs[i], None)
            self.assertEqual(costs[i].train_step, 1)
            self.assertEqual(reward[i].train_step, 1)
            self.assertEqual(agent.policy.train_step, 1)
            self.assertTrue(math.isfinite(costs[i].critic_loss1))
            self.assertTrue(math.isfinite(reward[i].critic_loss1))
            self.assertTrue(math.isfinite(reward[i].critic_loss2))
            self.assertTrue(torch.isfinite(agent.policy.actor_loss))
            self.assertTrue(any(not torch.equal(a, b) for a, b in
                                zip(before, agent.policy.actor_network.parameters())))


if __name__ == '__main__':
    unittest.main()
