import gym
import docopt
from agent import ppo_discrete
import tensorflow as tf

docoptstring = '''PPO_homebrew!
Usage:
  runner.py (--train | --test | --help) [options]

Options:
    --train        Train agent.
    --test         Test agent. (Enables rendering)
    --env ENV      Use Gym-environment ENV to run in. Continuous action envs not supported! [default: CartPole-v0]
    --steps S      Run S environment steps. [default: 1000000]
    --load AGENT   Load AGENT.
    --name N       Name the agent N. [default: ape]
    --verbose      More print statements!
    --help         Print this message.
'''
arguments = docopt.docopt(docoptstring)

env = gym.make(arguments["--env"])
with tf.Session() as session:
    #Init agent!
    agent = ppo_discrete( arguments["--name"], session, state_size=env.observation_space.shape, action_size=env.action_space.n )
    if arguments["--load"] is not None:
        agent.restore(arguments["--load"])
    #Init variables...
    s_prime, n_episodes, round_score,t0, R = env.reset(), 0, 0, -1, 0
    for t in range(int(arguments["--steps"])):
        s = s_prime
        a = agent.get_action(s)
        if arguments["--test"]:
            env.render()
        s_prime, r, done, _ = env.step(a)
        round_score += r
        if arguments["--train"]:
            agent.remember((s,a,r,s_prime,done))
        if done:
            n_episodes += 1
            alpha = 0.03
            R = (1-alpha)*R + alpha*round_score
            w = alpha*( 1-(1-alpha)**(n_episodes) )/(alpha)
            print("Episode length: {}, score: {} (ExpAvg: {})".format(t-t0, round_score, R/w))
            s_prime, round_score, t0 = env.reset(), 0, t
