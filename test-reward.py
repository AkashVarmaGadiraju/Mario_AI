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

from gym import Wrapper
from msvcrt import getch, kbhit

from gym.utils import play

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
        data = self.env.step(action)
        obs, reward, done, info = data[0][0], data[1], data[2], data[3]
        custom_reward = 0

        if(info["time"] == 400):
            self.currentInfo["time"] = 400.0

        if(done and (info["life"] == 0 or info["time"] == 0)):
            custom_reward = -100.0
        elif(self.currentInfo["life"]>info["life"]):
            custom_reward = -50.0
        elif(self.currentInfo["world"] < info["world"]):
            custom_reward = (info["world"] - self.currentInfo["world"]) * 100.0
        elif(self.currentInfo["stage"] < info["stage"]):
            custom_reward = 10.0
        elif(info["x_pos_screen"] != self.currentInfo["x_pos_screen"]):
            if info["x_pos_screen"] > self.currentInfo["x_pos_screen"]:
                custom_reward = 1.0
            else: 
                custom_reward = -1.0
        else: 
            custom_reward = -5.0
        
        self.currentInfo = info
        return obs, custom_reward, done, info

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

env = CustomRewardWrapper(env)
# print(env.get_action_meanings(), env.action_space.sample())

play.play(env, zoom=3)

# # Create a flag - restart or not
# done = True
# keypressed = ""
# action = None
# # Loop through each frame in the game
# while keypressed!="o": 
#     # Start the game to begin with 
#     if kbhit():
#         # If a key is pressed, get the key and do something with it
#         keypressed = ord(getch())
#         if keypressed == "a":
#             action = 3
#         elif keypressed == "w":
#             action = 1
#         elif keypressed == "s":
#             action = 2
#         elif keypressed == "d":
#             action = 4
#         elif keypressed == "n":
#             action = 5
#         elif keypressed == "m":
#             action = 6
#         elif keypressed == "0":
#             action = 0
#             break
#         else:
#             action = 0
#             keypressed = ""
#     else:
#         keypressed = ""
#         action = 0
    
#     if done: 
#         # Start the gamee
#         env.reset()
#     # Do random actions
#     state, reward, done, info = env.step(action)
#     # Show the game on the screen
#     env.render()
# # Close the game
# env.close()