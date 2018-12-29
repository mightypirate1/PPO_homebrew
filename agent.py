import tensorflow as tf
import numpy as np
import model
from model import ppo_discrete_model
from trajectory import trajectory

default_settings = {
                    "normalize_reward" : False,
                    "n_train_epochs" : 3,
                    "steps_before_training" : 8192,
                    "trajectory_length" : 1024, #128
                    "gamma" : 0.99,
                    "lambda" : 0.95,
                    "save_period" : 10,
                    }

class ppo_discrete:
    def __init__(self, name, session, state_size, action_size, model=None, settings={}, n_envs=1):
        self.name = name
        self.r_mu, self.r_sigma, self._r_mu, self._r_sigma = 0,1,0,0
        self.n_envs = n_envs
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
        self.current_trajectory = [trajectory(self.state_size, self.action_size) for _ in range(self.n_envs)]
        self.trajectories = []
        self.internal_t = 0
        self.n_trainings = 0
        self.n_saves = 0

    def get_action(self, s):
        state_list = s if isinstance(s,list) else [s]
        p, v = self.model.evaluate( state_list )
        assert not np.isinf(v).any() and not np.isnan(v).any(), v
        assert not np.isinf(p).any() and not np.isnan(p).any(), p
        if not isinstance(s, list):
            a = np.random.choice(np.arange(self.action_size), p=p[0])
            return a
        return [ np.random.choice(np.arange(self.action_size), p=p[i]) for i in range(len(s)) ]

    def remember(self,e):
        _s,_a,_r,_s_p,_d = e
        self.internal_t += self.n_envs
        for i in range(self.n_envs):
            s,a,r,s_p,d,t = _s[i],_a[i],_r[i],_s_p[i],_d[i], self.current_trajectory[i]
            t.add((s,a,r,d))
            if t.get_length() >= self.settings["trajectory_length"] or d:
                self.end_episode(_s_p, _d, indices=[i])
        if self.internal_t > self.settings["steps_before_training"]:
            self.end_episode(_s_p,_d)
            self.do_training(self.get_train_data(), self.settings["n_train_epochs"])
            self.internal_t -= self.settings["steps_before_training"]

    def end_episode(self, s_prime, done, indices=None):
        if indices is None: indices = [i for i in range(self.n_envs)]
        for i,idx in enumerate(indices):
            if self.current_trajectory[idx].get_length() > 0:
                self.current_trajectory[idx].end_episode(s_prime[i], done[i])
                self.trajectories.append(self.current_trajectory[idx])
                self.current_trajectory[idx] = trajectory(self.state_size, self.action_size)
    def get_train_data(self):
        ret = self.trajectories
        self.trajectories = []
        return ret
    def do_training(self, samples, epochs):
        states, actions, rewards, cumulative_rewards, advantages, target_values, old_probabilities, trajectory_lengths, n_samples \
                    = self.trainsamples_from_trajectories(samples)
        self.model.train(states, actions, cumulative_rewards, advantages, target_values, old_probabilities, trajectory_lengths, n_samples, epochs=epochs)
        if self.settings["normalize_reward"]:
            alpha = 0.01
            w = alpha*(alpha**(self.n_trainings+1)-1)/(alpha-1)
            self._r_mu    = (1-alpha) * self._r_mu    + alpha * rewards.mean()
            self._r_sigma = (1-alpha) * self._r_sigma + alpha * rewards.var()**0.5
            self.r_mu = self._r_mu / w
            self.r_sigma = self._r_sigma / w
            print("Reward mu:{}, sigma:{}".format(str(np.round(self.r_mu,decimals=4)).ljust(7), str(np.round(self.r_sigma,decimals=4)).ljust(7)))
        if self.n_trainings % self.settings["save_period"] == 0:
            self.model.save(self.n_saves)
            self.n_saves += 1
        self.n_trainings += 1

    def trainsamples_from_trajectories(self, trajectories):
        states, actions, rewards, cumulative_rewards, advantages, target_values, old_probabilities, trajectory_lengths, n_samples = [], [], [], [], [], [], [], [], 0
        for t in trajectories:
            adv, targ, old_prob = t.process_trajectory(
                                                        self.model,
                                                        gamma_discount=self.settings["gamma"],
                                                        lambda_discount=self.settings["lambda"],
                                                        r_mu=self.r_mu if self.settings["normalize_reward"] else 0,
                                                        r_sigma=self.r_sigma if self.settings["normalize_reward"] else 1,
                                                      )
            states += t.get_states()
            rewards += t.r
            cumulative_rewards += [t.get_cumulative_reward(gamma_discount=self.settings["gamma"])]
            actions += t.a_1hot
            advantages += adv
            target_values += targ
            old_probabilities += old_prob
            trajectory_lengths += [t.get_length()]
            n_samples += t.get_length()
        return np.array(states), np.array(actions), np.array(rewards), np.array(cumulative_rewards), np.array(advantages), np.array(target_values), np.array(old_probabilities), np.array(trajectory_lengths), n_samples
    def restore(self, savepoint):
        self.model.restore(savepoint)
