import gym
import docopt
from agent import ppo_discrete
import wrappers
import tensorflow as tf
import numpy as np
from time import time
from aux.parameters import *
from aux import aux

docoptstring = \
'''PPO_homebrew!
Usage:
  runner.py (--train | --test | --help) [options]
  runner.py (--train | --test | --help) [options] --x (<opt> <setting>)...

Options:
    --train        Train agent.
    --test         Test agent. (Enables rendering)
    --env ENV      Use Gym-environment ENV to run in. Continuous action envs not supported! [default: CartPole-v0]
    --n_envs N     Training uses N parallel environments to collect trajectories [default: 64].
    --steps S      Run S environment steps. [default: 1000000]
    --load AGENT   Load AGENT.
    --name N       Name the agent N. [default: no-id]
    --atari        Applies a set of wrappers for Atari-environments.
    --verbose      More print statements!
    --help         Print this message.

Using the --x option is developer-mode.
'''
settings = docopt.docopt(docoptstring) #Handle args...

#We keep some separate hyper-parameters for the Atari-stuff, out of convenience :)
if settings["--atari"]:
    agent_settings = {
                        #Runner
                        "render_training"          : False,
                        #Agent
                        "evals_on_cpu"             : False,
                        "normalize_advantages"     : False,
                        "minibatch_size"           : 256, #128
                        "n_train_epochs"           : 3,
                        "steps_before_training"    : 4*8192,
                        "trajectory_length"        : 128,
                        "gamma"                    : 0.99,
                        "lambda"                   : 0.95,
                        #Model
                        "epsilon"                  : linear_parameter(0.1   , final_val=0, time_horizon=1e7), #0.1,
                        "lr"                       : linear_parameter(2.5e-4, final_val=0, time_horizon=1e7), #1e-4,
                        "weight_loss_policy"       : 1.0,
                        "weight_loss_entropy"      : 0.01,
                        "weight_loss_value"        : 1.00,
                     }
else: #Options not set in this dict will use default values specified in each class
    agent_settings = {
                        "render_training" : False,
                     }

#MAIN CODE:
n_envs = int(settings["--n_envs"]) if settings["--train"] else 1
agent_settings = aux.settings_dict(settings["<opt>"],settings["<setting>"], dict=agent_settings)
wrapper = wrappers.wrap_atari if settings["--atari"] else None #Wrap the Atari-envs like the big boys do!
env = wrappers.multi_env(settings["--env"], n=n_envs, wrapper=wrapper) #Class for keeping a bunch of environments. It implements vector-versions of the relevant functions...

with tf.Session() as session:
    #Init agent!
    agent = ppo_discrete(
                            settings["--name"],
                            session,
                            state_size=env.observation_space.shape,
                            action_size=env.action_space.n,
                            settings=agent_settings,
                            n_envs=n_envs
                        )
    #Optionally, we can restore an agent.
    if settings["--load"] is not None:
        agent.restore(settings["--load"])

    #Init variables...
    s_prime, n_episodes, round_score,t0, R = env.reset(), 0, np.zeros(n_envs), -np.ones(n_envs), 0

    #Train!
    for t in range(int(settings["--steps"])):
        s = s_prime
        a = agent.get_action(s)
        if settings["--test"] or agent_settings["render_training"]:
            env.render()
        s_prime, r, done, _ = env.step(a)
        round_score += np.array(r)
        if settings["--train"]:
            agent.remember((s,a,r,s_prime,done))
        for i,d in enumerate(done):
            if d: #If some env reached a terminal state:
                #We keep track of an exponentially running average of the episode-returns...
                n_episodes += 1
                alpha = 0.01
                R = (1-alpha)*R + alpha*round_score[i]
                if i == 0: #...and for one special env, we print  some stats, and the running average.
                    w = (1-(1-alpha)**n_episodes)
                    print("{} :: Episode length: {}, score: {} (ExpAvg: {})".format(t*n_envs,str(t-t0[i]).rjust(5), str(round_score[i]).rjust(7), str(R/w)))
                #Don't forget to also reset it!
                s_prime[i] = env.reset_by_idx(i)
                round_score[i], t0[i] = 0, t
