class trajectory:
    def __init__(self, state_size, action_size):
        self.length = 0
        self.state_size = state_size
        self.action_size = action_size
        self.s, self.a, self.a_1hot, self.r, self.d = [], [], [], [], []
    def get_states(self):
        return [self.s[i] for i,a in enumerate(self.a) if a is not None]
    def add(self,e, end_of_trajectory=False):
        s,a,r,d = e
        self.s.append(s)
        if not end_of_trajectory:
            self.a.append(a)
            self.a_1hot.append([int(x == a) for x in range(self.action_size)]) #1-hot encoding of the action
            self.r.append(r)
            self.d.append(d)
            self.length += 1
    def get_length(self):
        return self.length
    def get_cumulative_reward(self, gamma_discount=0.99):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, gamma_discount=0.99, lambda_discount=0.95):
        advantages     = [0 for x in range(self.length)]
        td_errors      = [0 for x in range(self.length)]
        target_values  = [0 for x in range(self.length)]
        p,v = model(self.s)
        for i in range(self.length):
            td_errors[i] = -v[i] + self.r[i] + gamma_discount*v[i+1]*int(not self.d[i])

        adv = 0
        for i, td in reversed(list(enumerate(td_errors))):
            adv += lambda_discount * td
                advantages[i] = adv
        #ADNANTAGE METHOD
        target_values = [x+y for x,y in zip(v.tolist(),advantages)]
        old_probabilities = [[p[i,a]] for i,a in enumerate(self.a)]
        return advantages, target_values, old_probabilities
    def end_episode(self,sp,d):
        self.add((sp, None, None, d), end_of_trajectory=True)
    def __len__(self):
        return self.get_length()
