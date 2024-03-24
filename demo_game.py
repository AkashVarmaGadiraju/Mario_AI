# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, [
    ['NOOP'],
    ['up'],
    ['down'],
    ['left'],
    ['right'],
    ['A'],
    ['B'],
])

print(env.get_action_meanings(), env.action_space.sample())

# # Create a flag - restart or not
# done = True
# # Loop through each frame in the game
# for step in range(100000): 
#     # Start the game to begin with 
#     if done: 
#         # Start the gamee
#         env.reset()
#     # Do random actions
#     state, reward, done, info = env.step(env.action_space.sample())
#     # Show the game on the screen
#     env.render()
# # Close the game
# env.close()