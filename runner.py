import gym
import docopt
from agent import ppo_discrete
import aux
import wrappers
import tensorflow as tf
import threaded_runner
from time import time

docoptstring = '''PPO_homebrew!
Usage:
  runner.py (--train | --test | --help) [options]
  runner.py (--train | --test | --help) [options] --x (<opt> <setting>)...

Options:
    --train        Train agent.
    --test         Test agent. (Enables rendering)
    --env ENV      Use Gym-environment ENV to run in. Continuous action envs not supported! [default: CartPole-v0]
    --steps S      Run S environment steps. [default: 1000000]
    --load AGENT   Load AGENT.
    --name N       Name the agent N. [default: ape]
    --atari        Applies a set of wrappers for Atari-environments.
    --verbose      More print statements!
    --help         Print this message.

Using the --x option is developer-mode.
'''

settings = docopt.docopt(docoptstring)
agent_settings = aux.settings_dict(settings["<opt>"],settings["<setting>"])
env = gym.make(settings["--env"])
if settings["--atari"]:
    env = wrappers.wrap_atari(env)

''' THREADED VERSION '''
agents = []
n_actors = 4
with tf.Session() as session:
    envs = [gym.make(settings["--env"]) for _ in range(n_actors)]
    model = None
    for i in range(n_actors):
        name = settings["--name"] + str(i)
        with tf.device("/device:GPU:0"):
            agent = ppo_discrete(
                                    name,
                                    session,
                                    state_size=env.observation_space.shape,
                                    action_size=env.action_space.n,
                                    settings=agent_settings,
                                    model=model,
                                    threaded=True
                                )
            model = agent.model
            agents.append( agent )
    with tf.device("/device:GPU:0"):
        trainer = ppo_discrete(
                                settings["--name"]+"_trainer",
                                session,
                                state_size=env.observation_space.shape,
                                action_size=env.action_space.n,
                                settings=agent_settings,
                                threaded=True
                                )
    # thread_runner = threaded_runner.threaded_runner(
    #                                                 envs=envs,
    #                                                 runners=agents,
    #                                                 trainer=trainer,
    #                                                 train_epochs=3,
    #                                                 steps=int(128/n_actors),
    #                                                 )

    for t in range( int(4096/n_actors) ):
        print("-----iteration{}-----".format(t))
        thread_runner = threaded_runner.threaded_runner(
                                                        envs=envs,
                                                        runners=agents,
                                                        trainer=trainer,
                                                        train_epochs=3,
                                                        steps=int(4096/n_actors),
                                                        )

        thread_runner.run()
        thread_runner.join()

exit()

with tf.Session() as session:
    #Init agent!
    agent = ppo_discrete( settings["--name"], session, state_size=env.observation_space.shape, action_size=env.action_space.n, settings=agent_settings )
    if settings["--load"] is not None:
        agent.restore(settings["--load"])

    #Init variables...
    s_prime, n_episodes, round_score,t0, R = env.reset(), 0, 0, -1, 0
    for t in range(int(settings["--steps"])):
        s = s_prime
        a = agent.get_action(s)
        if settings["--test"]:
            env.render()
        s_prime, r, done, _ = env.step(a)
        round_score += r
        if settings["--train"]:
            agent.remember((s,a,r,s_prime,done))
        if done:
            n_episodes += 1
            alpha = 0.03
            R = (1-alpha)*R + alpha*round_score
            w = alpha*( 1-(1-alpha)**(n_episodes) )/(alpha)
            print("Episode length: {}, score: {} (ExpAvg: {})".format(t-t0, round_score, R/w))
            s_prime, round_score, t0 = env.reset(), 0, t
