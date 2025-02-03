import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestForexEnv(unittest.TestCase):
    def setUp(self):
        from forex_env import ForexEnv
        # Initialize environment with sample parameters
        self.env = ForexEnv(
            instrument='EUR_USD',
            start_date='2020-01-01',
            end_date='2021-01-01',
            granularity='D',
            initial_balance=1000,
            leverage=50,
            window_size=14
        )

    def test_reset(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.shape[0], self.env.window_size)
        self.assertFalse(self.env.done)

    def test_step(self):
        self.env.reset()
        obs, reward, done, trunc, info = self.env.step(1)
        self.assertEqual(obs.shape[0], self.env.window_size)
        self.assertIn('close', self.env.data_handler.feature_names)

    def test_step_until_done(self):
        self.env.reset()
        while not self.env.done:
            obs, reward, done, trunc, info = self.env.step(1)
        self.assertTrue(self.env.done)


if __name__ == '__main__':
    unittest.main()
