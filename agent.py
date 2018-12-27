import tensorflow as tf
import numpy as np
import model
from model import ppo_discrete_model
from trajectory import trajectory

default_settings = {
                    "minibatch_size" : 512, #128
                    "n_train_epochs" : 3,
                    "steps_before_training" : 4096, #8k
                    "trajectory_length" : 1024, #128
                    "gamma" : 0.99,
                    "lambda" : 0.95,
                    "save_period" : 10,
                    }

class ppo_discrete:
    def __init__(self, name, session, state_size, action_size, model=None, settings={}, threaded=False):
        self.name = name
        self.threaded = threaded
        self.settings = default_settings.copy()
        for x in settings: self.settings[x] = settings[x]
        self.state_size = state_size
        self.action_size = action_size
        self.pixels = len(self.state_size) == 3 #Assume pixels!
        if model is None:
            self.model = ppo_discrete_model(
                                            self.name,
                                            self.state_size,
                                            self.action_size,
                                            session,
                                            pixels=self.pixels,
                                            settings=self.settings
                                            )
        else:
            self.model = model
        self.current_trajectory = trajectory(self.state_size, self.action_size)
        self.trajectories = []
        self.internal_t = 0
        self.n_trainings = 0
        self.n_saves = 0

    def get_action(self, s):
        p, v = self.model.evaluate([s], pixels=self.pixels)
        a = np.random.choice(np.arange(self.action_size), p=p[0])
        assert not np.isinf(v[0]) and not np.isnan(v[0]), v[0]
        assert not np.any(np.isinf(p[0])) and not np.any(np.isnan(p[0])), p[0]
        return a

    def remember(self,e):
        s,a,r,s_p,d = e
        self.internal_t += 1
        self.current_trajectory.add((s,a,r,d))
        if self.current_trajectory.get_length() >= self.settings["trajectory_length"] or e[4] and not self.threaded: #If we run in threaded mode, the thread-handler will be responsible for initiating training!
            self.end_episode(s_p,d)
        if self.internal_t % self.settings["steps_before_training"] == 0 and not self.threaded: #If we run in threaded mode, the thread-handler will be responsible for initiating training!
            self.end_episode(s_p,d)
            self.do_training(self.get_train_data(), self.settings["n_train_epochs"])
        if self.threaded:
            return [s_p, d] #This is by convention of the threading-framework: any thing you want to recieve in the end_episode function is passed as a list by remember
    def end_episode(self, s_prime, done):
        if self.current_trajectory.get_length() > 0:
            self.current_trajectory.add((s_prime, None, None, done), end_of_trajectory=True) #This is the system used to get an s' in for the last state too!
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = trajectory(self.state_size, self.action_size)
    def get_train_data(self):
        ret = self.trajectories
        self.trajectories = []
        return ret
    def do_training(self, samples, epochs):
        states, actions, advantages, target_values, old_probabilities, n_samples \
                    = self.trainsamples_from_trajectories(samples)
        print("-------")
        print("training on {} samples".format(n_samples), end='',flush=True)
        for i in range(epochs):
            pi = np.random.permutation(np.arange(n_samples))
            for x in range(0,n_samples,self.settings["minibatch_size"]):
                self.model.train(
                                    states[pi[x:x+self.settings["minibatch_size"]]],
                                    actions[pi[x:x+self.settings["minibatch_size"]]],
                                    advantages[pi[x:x+self.settings["minibatch_size"]]],
                                    target_values[pi[x:x+self.settings["minibatch_size"]]],
                                    old_probabilities[pi[x:x+self.settings["minibatch_size"]]],
                                    pixels=self.pixels
                                )
            print(".",end='',flush=True)
        print("\n")
        if self.n_trainings % self.settings["save_period"] == 0:
            self.model.save(self.n_saves)
            self.n_saves += 1
        self.n_trainings += 1
        print("-------")

    def trainsamples_from_trajectories(self, trajectories):
        states, actions, advantages, target_values, old_probabilities, n_samples = [], [], [], [], [], 0
        for t in trajectories:
            adv, targ, old_prob = t.process_trajectory(
                                                        self.model,
                                                        gamma_discount=self.settings["gamma"],
                                                        lambda_discount=self.settings["lambda"],
                                                      )
            states += t.get_states()
            actions += t.a_1hot
            advantages += adv
            target_values += targ
            old_probabilities += old_prob
            n_samples += t.get_length()
        return np.array(states), np.array(actions), np.array(advantages), np.array(target_values), np.array(old_probabilities), n_samples
    def restore(self, savepoint):
        self.model.restore(savepoint)
