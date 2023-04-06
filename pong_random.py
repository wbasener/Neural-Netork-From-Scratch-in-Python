# See: https://www.youtube.com/watch?v=XbWhJdQgi7E
# https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
# env.reset()

#trained_model = A2C.load("a2c_pong", verbose=1)
trained_model = A2C.load("a2c_pong_3", verbose=1)
env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
trained_model.set_env(env)

vec_env = trained_model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = trained_model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()