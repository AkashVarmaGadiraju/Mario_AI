# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

from gym import Wrapper

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        
    def step(self, action):
        # Get the next state, reward, done, and info from the environment
        obs, reward, done, info = self.env.step(action)
        
        # Calculate custom reward based on your logic
        # custom_reward = self.calculate_custom_reward(obs, reward, done, info)
        custom_reward = (400 + info["score"] - info["time"]) + (info["life"] * 500) + ((info["world"] - 1) *  1000  + (info["stage"] * 100)) + (info["x_pos"] / 100)
        return obs, custom_reward, done, info

# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')

env = JoypadSpace(env, [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['down'],
])
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

# Load model
model = PPO.load('./train/best_model_2000000')

# Create a flag - restart or not
done = True
# Start the game    
# Loop through the game
while True: 
    # Start the game to begin with 
    if done: 
        # Start the gamee
        state = env.reset()
    action, _ = model.predict(state.copy())
    state, reward, done, info = env.step(action)
    env.render()
