class trajectory:
    def __init__(self, state_size, action_size):
        self.length = 0
        self.state_size = state_size
        self.action_size = action_size
        self.s, self.a, self.a_1hot, self.r, self.d = [], [], [], [], []
    def get_states(self):
        return [self.s[i] for i,a in enumerate(self.a) if a is not None]
    def add(self,e):
        s,a,r,d = e
        self.s.append(s)
        if a is not None:
            self.a.append(a)
            self.a_1hot.append([int(x == a) for x in range(self.action_size)]) #1-hot encoding of the action
            self.r.append(r)
            self.d.append(d)
            self.length += 1
    def get_length(self):
        return self.length
    def process_trajectory(self, model, gamma_discount=0.99, lambda_discount=0.95):
        advantages     = [0 for x in range(self.length)]
        td_errors      = [0 for x in range(self.length)]
        target_values  = [0 for x in range(self.length)]
        p,v = model(self.s)
        for i in range(self.length):
            td_errors[i] = -v[i] + self.r[i] + gamma_discount*v[i+1]*int(not self.d[i])
        for i in range(self.length):
            for j in range(i, self.length):
                advantages[i] += lambda_discount**(j-i) * td_errors[j]
        #ADNANTAGE METHOD
        target_values = [x+y for x,y in zip(v.tolist(),advantages)]
        #OTHER METHOD
        # for i in range(self.length):
        #     target_values[i] = self.r[i] + gamma_discount * v[i+1]
        old_probabilities = [[p[i,a]] for i,a in enumerate(self.a)]
        return advantages, target_values, old_probabilities
