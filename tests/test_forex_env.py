import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.forex_env import ForexEnv

class TestForexEnv(unittest.TestCase):

    def setUp(self):
        self.env = ForexEnv(
            instrument='EUR_USD',
            start_date='2020-01-01',
            end_date='2020-01-10',
            granularity='D',
            initial_balance=1000,
            leverage=50,
            window_size=5
        )
        # Mock data for testing
        data = {
            'time': pd.date_range(start='2020-01-01', periods=10, freq='D'),
            'open': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            'high': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
            'low': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'close': [1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05]
        }
        self.env.data = pd.DataFrame(data).set_index('time')

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(self.env.balance, 1000)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.profit, 0)
        self.assertEqual(self.env.done, False)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(obs.shape, (5, 4))

    def test_step(self):
        self.env.reset()
        obs, reward, done, _ = self.env.step(1)
        self.assertEqual(self.env.position, 1)
        self.assertEqual(obs.shape, (5, 4))
        self.assertFalse(done)

    def test_get_reward(self):
        self.env.reset()
        self.env._take_action(1)
        self.env.current_step = 5
        reward = self.env._get_reward()
        self.assertEqual(reward, (1.65 - 1.15) * 50)

    def test_render(self):
        self.env.reset()
        self.env.render()

if __name__ == '__main__':
    unittest.main()
