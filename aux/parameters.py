import math

class parameter:
    def __init__(self, value, final_val=None, time_horizon=None):
        self.init_val = value
        self.final_val = final_val
        self.time_horizon = time_horizon
    def __eval__(self, t):
        return self.get_value(t)
    def get_value(self, t):
        return self.init_val
    def __str__(self):
        return "{}(init:{}, final:{}, time_horizon:{})".format(type(self).__name__,self.init_val,self.final_val,self.time_horizon)

class exp_parameter(parameter): #Very crude exp-decay :)
    def __init__(self, value, **kwargs):
        parameter.__init__(self, value, **kwargs)
        self.__init_vars__()
    def __init_vars__(self):
        assert self.final_val is not None and self.time_horizon is not None, "exp_parameter requires final_val=x and time_horizon=y with valid x,y to be passed to its constructor."
        self.offset = min(self.init_val, self.final_val, 0) - 1
        self.a = self.init_val - self.offset
        self.b = self.final_val - self.offset
        self.decay = math.log(self.b/self.a)/self.time_horizon
    def get_value(self, t):
        return min(max(self.a * math.exp(self.decay * t) + self.offset, self.final_val), self.init_val)

class linear_parameter(parameter):
    def __init__(self, value, **kwargs):
        parameter.__init__(self, value, **kwargs)
        self.__init_vars__()
    def __init_vars__(self):
        pass
    def get_value(self,t):
        x = max( min(t,self.time_horizon), 0 ) / self.time_horizon
        return x * self.final_val + (1-x) * self.init_val

constant_parameter = parameter
