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
            "x_pos_screen": 40,
        }
        self.status = {
            "small": 1,
            "tall": 2,
            "fireball": 3,
        }

    def step(self, action):
        # Get the next state, reward, done, and info from the environment
        data = self.env.step([action])
        obs, reward, done, info = data[0][0], data[1].item(0), data[2].item(0), data[3][0]
        custom_reward = 0

        if(info["time"] == 400):
            self.currentInfo["time"] = 400.0

        if(done and (info["life"] == 0 or info["time"] == 0)):
            custom_reward = -100.0
        elif(self.currentInfo["life"] != info["life"]):
            if(self.currentInfo["life"]>info["life"]):
                custom_reward = -50.0
            else:
                custom_reward = 50.0
        elif(self.currentInfo["world"] < info["world"]):
            custom_reward = (info["world"] - self.currentInfo["world"]) * 50.0
        elif(self.currentInfo["stage"] < info["stage"]):
            custom_reward = (info["stage"] - self.currentInfo["stage"]) * 10.0
        else: 
            custom_reward = (info["x_pos_screen"] - self.currentInfo["x_pos_screen"]) * 1.0 - (self.currentInfo["time"] - info["time"]) * 1.0
        
        self.currentInfo = info
        return obs, custom_reward, done, info

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')

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
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

env = CustomRewardWrapper(env)


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00003, n_steps=10000)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=2000000, callback=callback)

model.save('thisisatestmodel')