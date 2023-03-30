# See: https://www.youtube.com/watch?v=XbWhJdQgi7E
# https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
import gym

env = gym.make('LunarLander-v2', render_mode="human")  # continuous: LunarLanderContinuous-v2
env.reset()

for step in range(150):
	env.render()
	# take random action
	obs, reward, landed, done, info = env.step(env.action_space.sample())
	print('Observation: '+str(obs))
	print('Reward: '+str(reward)+'|'+str(landed))

env.close()