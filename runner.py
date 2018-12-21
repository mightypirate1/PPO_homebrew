import gym
from agent import ppo_discrete
import tensorflow as tf

restore = None
# restore = "my_first_agent-45"
test = restore is not None
env = gym.make("Breakout-v0")
s_prime, t0, R = env.reset(), 0, 0
with tf.Session() as session:
    agent = ppo_discrete( session, state_size=s_prime.shape, action_size=env.action_space.n )
    if test:
        agent.restore(restore)
    for t in range(10000000):
        s = s_prime
        a = agent.get_action(s)
        if test:
            env.render()
        s_prime, r, done, _ = env.step(a)
        if not test:
            agent.remember((s,a,r,s_prime,done))
        if done:
            alpha = 0.1
            R = (1-alpha)*R + alpha*(t-t0)
            print("Episode length: {}, score: {}".format(t-t0, R))
            s_prime, t0 = env.reset(), t
