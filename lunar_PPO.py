import gym
from stable_baselines3 import PPO

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

episodes = 1

for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		env.render()
		print(rewards)
		print(ep)

print('pause')