import gym
env = gym.make('CartPole-v0')
env.reset()

counter = 0
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        #print('Game failed')
        counter += 1
        print('Game failed', counter)
        env.reset()
