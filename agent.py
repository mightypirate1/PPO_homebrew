import tensorflow as tf
import numpy as np
import model
from model import ppo_discrete_model
from trajectory import trajectory

default_settings = {
                    "name" : "breakout",
                    "minibatch_size" : 64,
                    "n_train_epochs" : 10,
                    "learning_rate" : 10**-4,
                    "epsilon" : 0.2,
                    "weight_loss_policy" : 1.0,
                    "weight_loss_entropy" : 0.0001,
                    "weight_loss_value" : 1.05,
                    "steps_before_training" : 1024,
                    "trajectory_length" : 32,
                    "gamma" : 0.99,
                    "lambda" : 0.99,
                    "save_period" : 10,
                    }

class ppo_discrete:
    def __init__(self, session, state_size, action_size, trajectory_length=50, steps_before_update=1000, settings={}):
        self.settings = default_settings.copy()
        for x in settings: self.settings[x] = settings[x]
        self.state_size = state_size
        self.action_size = action_size
        pixels = len(self.state_size) == 3 #Assume pixels!
        self.model = ppo_discrete_model(
                                        self.settings["name"],
                                        self.state_size,
                                        self.action_size,
                                        session,
                                        lr=self.settings["learning_rate"],
                                        epsilon=self.settings["epsilon"],
                                        pixels=pixels,
                                        loss_weights=(
                                                        self.settings["weight_loss_policy"],
                                                        self.settings["weight_loss_entropy"],
                                                        self.settings["weight_loss_value"]
                                                     )
                                        )
        self.current_trajectory = trajectory(self.state_size, self.action_size)
        self.trajectories = []
        self.internal_t = 0
        self.n_saves = 0

    def get_action(self, s):
        p, v = self.model.evaluate([s])
        a = np.random.choice(np.arange(self.action_size), p=p[0])
        assert not np.isinf(v[0]) and not np.isnan(v[0]), v[0]
        assert not np.any(np.isinf(p[0])) and not np.any(np.isnan(p[0])), p[0]
        return a

    def remember(self,e):
        s,a,r,s_p,d = e
        if e[1] is not None:
            self.internal_t += 1
        self.current_trajectory.add((s,a,r,d))
        if self.current_trajectory.get_length() == self.settings["trajectory_length"] or e[4]:
            self.current_trajectory.add((s_p,None,None,d))
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = trajectory(self.state_size, self.action_size)
        if self.internal_t % self.settings["steps_before_training"] == 0:
            self.do_training()
            self.trajectories.clear()

    def do_training(self):
        states, actions, advantages, target_values, old_probabilities, n_samples \
                    = self.trainsamples_from_trajectories(self.trajectories)
        print("-------")
        print("training on {} samples".format(n_samples), end='',flush=True)
        for i in range(self.settings["n_train_epochs"]):
            pi = np.random.permutation(np.arange(n_samples))
            for x in range(0,n_samples,self.settings["minibatch_size"]):
                self.model.train(
                                    states[pi[x:x+self.settings["minibatch_size"]]],
                                    actions[pi[x:x+self.settings["minibatch_size"]]],
                                    advantages[pi[x:x+self.settings["minibatch_size"]]],
                                    target_values[pi[x:x+self.settings["minibatch_size"]]],
                                    old_probabilities[pi[x:x+self.settings["minibatch_size"]]]
                                )
            print(".",end='',flush=True)
        print("-------")
        if self.n_saves % self.settings["save_period"] == 0:
            self.model.save(self.n_saves)
            self.n_saves += 1

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
