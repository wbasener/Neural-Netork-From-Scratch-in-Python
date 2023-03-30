# See: https://www.youtube.com/watch?v=XbWhJdQgi7E
# https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
# NOTE: I needed to do 'pip install pyglet==1.5.27'

import gym
from stable_baselines3 import A2C

env = gym.make('LunarLander-v2', render_mode="human")  # continuous: LunarLanderContinuous-v2
env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 10

for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		env.render()
		print(rewards)