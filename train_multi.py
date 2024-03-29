# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Import a function which checks if my custom env is correct ot not
from stable_baselines3.common.env_checker import check_env

import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

from gym import Wrapper

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.currentInfo = {
            "life": 2,
            "time": 400,
            "world": 1,
            "stage": 1,
            "x_pos": 40,
            "y_pos": 40,
            "status": "small",
            "x_pos_screen": 0,
        }
        self.status = {
            "small": 1,
            "tall": 2,
            "fireball": 3,
        }

    def step(self, action):
        # Get the next state, reward, done, and info from the environment
        data = self.env.step(action)
        obs, reward, done, info = data
        custom_reward = 0
        if(done and info["life"] == 0):
            custom_reward = -100
        elif(self.currentInfo["life"] != info["life"]):
            if(self.currentInfo["life"]>info["life"]):
                custom_reward = -50
            else:
                custom_reward = 50
        elif(self.currentInfo["world"] < info["world"]):
            custom_reward = (info["world"] - self.currentInfo["world"]) * 50
        elif(self.currentInfo["stage"] < info["stage"]):
            custom_reward = (info["stage"] - self.currentInfo["stage"]) * 10
        else: 
            custom_reward = (info["x_pos_screen"] - self.currentInfo["x_pos_screen"]) * 2 - (info["time"] - self.currentInfo["time"]) * 1
        
        self.currentInfo = info
        return obs, custom_reward, done, info

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    
def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Setup game
        env = gym_super_mario_bros.make('SuperMarioBros-v0')

        env.reset(seed=seed + rank)

        env = JoypadSpace(env, [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
            ['right', 'B'],
            ['right', 'A', 'B'],
            ['A'],
            ['down'],
            ['left'],
        ])

        env = CustomRewardWrapper(env)
        

        # # 3. Grayscale
        # env = GrayScaleObservation(env, keep_dim=True)

        # env = DummyVecEnv([lambda: env])

        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    num_of_cpus = os.cpu_count()

    if(num_of_cpus == None):
        num_of_cpus = 4
    elif num_of_cpus > 4:
        num_of_cpus = 9
    
    env = SubprocVecEnv([make_env(i) for i in range(num_of_cpus)])

    env = VecMonitor(env)

    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/' 

    # Setup model saving callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=100000, log_dir=CHECKPOINT_DIR)

    # This is the AI model started
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00003, n_steps=10000)

    # Train the AI model, this is where the AI model starts to learn
    model.learn(total_timesteps=1000000, callback=callback)

    model.save('thisisatestmodel')