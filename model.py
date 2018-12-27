import tensorflow as tf
import numpy as np

default_settings = {
                    "dense_n_hidden"      : 6,
                    "dense_hidden_size"   : 2048,

                    "conv_n_convs"        : 3,
                    "conv_n_channels"     : 32,
                    "conv_filter_size"    : (5,5),
                    "conv_n_dense"        : 3,
                    "conv_dense_size"     : 1024,

                    "epsilon"             : 0.2,
                    "lr"                  : 1e-4,
                    "weight_loss_policy"  : 1.0,
                    "weight_loss_entropy" : 0.01,
                    "weight_loss_value"   : 0.50,
                    }

class ppo_discrete_model:
    def __init__(self, name, state_size, action_size, session, pixels=False, settings={}):
        self.settings = default_settings.copy()
        for x in settings: self.settings[x] = settings[x]
        print("model created with settings:")
        for x in default_settings:
            print("\t{}\t{}".format(x.ljust(20), self.settings[x]))
        print("---")
        self.saved_weights = None
        self.session = session
        self.name = name
        with tf.variable_scope("ppo_discrete"+self.name) as scope:
            self.states_tf = tf.placeholder(dtype=tf.float32, shape=(None, *state_size), name='states')
            self.actions_tf = tf.placeholder(dtype=tf.float32, shape=(None, action_size), name='actions')
            self.advantages_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='advantages')
            self.target_values_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='target_values')
            self.old_probabilities_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='old_probs')
            # self.probabilities_tf = self.create_net(name='policy_net', input=self.states_tf, output_size=action_size, output_activation=tf.nn.softmax)
            # self.values_tf        = self.create_net(name='value_net',  input=self.states_tf, output_size=1,           output_activation=None)
            self.probabilities_tf, self.values_tf = self.create_net(
                                                                    name='policy_net',
                                                                    input=self.states_tf,
                                                                    output_activation=tf.nn.softmax,
                                                                    output_size=action_size,
                                                                    add_value_head=True,
                                                                    pixels=pixels
                                                                    )
            self.training_ops = self.create_training_ops(
                                                            self.actions_tf,
                                                            self.probabilities_tf,
                                                            self.old_probabilities_tf,
                                                            self.advantages_tf,

                                                            self.values_tf,
                                                            self.target_values_tf,
                                                            epsilon=self.settings["epsilon"],
                                                            lr=self.settings["lr"],
                                                        )
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.assign_ops, assign_values = self.create_weight_setting_ops()
            self.init_ops = tf.variables_initializer(self.all_variables)
            #self.saver = tf.train.Saver(self.all_variables, self.name)
        session.run(self.init_ops)

    def evaluate(self, states):
        run_list = [self.probabilities_tf,self.values_tf]
        feed_dict = {self.states_tf : states}
        probs, vals = self.session.run(run_list, feed_dict=feed_dict)
        return probs, vals

    def train(self, states, actions, advantages, target_values, old_probabilities):
        run_list = [self.training_ops]
        feed_dict = {
                        self.states_tf : states,
                        self.actions_tf : actions,
                        self.advantages_tf : advantages,
                        self.target_values_tf : target_values,
                        self.old_probabilities_tf : old_probabilities,
                    }
        self.session.run(run_list, feed_dict=feed_dict)

    def create_training_ops(self,actions_tf, probabilities_tf, old_probabilities_tf, advantages_tf, values_tf, target_values_tf, lr=None, epsilon=None):
        #Fudge it up so it doesnt inf/nan...
        e = 10**-7
        probs = tf.maximum(probabilities_tf, e)
        old_probs = tf.maximum(old_probabilities_tf, e)
        #Define some intermediate tensors...
        entropy_tf = tf.reduce_sum(-tf.multiply(probs, tf.log(probs)), axis=1)
        action_prob_tf = tf.reduce_sum(tf.multiply(actions_tf, probs), axis=1, keep_dims=True)
        ratio_tf = tf.div( action_prob_tf , old_probs )
        ratio_clipped_tf = tf.clip_by_value(ratio_tf, 1-epsilon, 1+epsilon)
        #Define the loss tensors!
        loss_clip_tf = tf.reduce_mean(tf.minimum( tf.multiply(ratio_tf,advantages_tf), tf.multiply(ratio_clipped_tf,advantages_tf) ) )
        loss_entropy_tf = tf.reduce_mean(entropy_tf)
        loss_value_tf = tf.losses.mean_squared_error(values_tf, target_values_tf)
        loss_tf = - self.settings["weight_loss_policy"]  * loss_clip_tf      \
                  - self.settings["weight_loss_entropy"] * loss_entropy_tf   \
                  + self.settings["weight_loss_value"]   * loss_value_tf
        #Minimize loss!
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_tf)

    def create_net(self, name=None, input=None, output_size=None, output_activation=tf.nn.elu, add_value_head=False, pixels=False):
        print("model: create net")
        with tf.variable_scope(name):
            if not pixels:
                hidden = self.create_dense(input)
            else:
                hidden = self.create_conv(input)
            ret = tf.layers.dense(
                                hidden,
                                output_size,
                                activation=output_activation,
                                )
            if add_value_head:
                val = tf.layers.dense(
                                    hidden,
                                    1,
                                    activation=None,
                                    )
                return ret, val
            return ret
    def create_dense(self, input_tensor):
        print("model: create dense")
        x = input_tensor
        for n in range(self.settings["dense_n_hidden"]):
            print("\t",self.settings["dense_hidden_size"]," unit layer")
            x = tf.layers.dense(
                                x,
                                self.settings["dense_hidden_size"],
                                activation=tf.nn.elu,
                                )
        return x
    def create_conv(self, input_tensor):
        x = input_tensor
        print("model: create conv")
        for n in range(self.settings["conv_n_convs"]):
            print("\t",self.settings["conv_n_channels"]," channel layer: ", self.settings["conv_filter_size"])
            x = tf.layers.conv2d(
                                x,
                                self.settings["conv_n_channels"],
                                self.settings["conv_filter_size"],
                                padding='same',
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                )
        # x = tf.layers.average_pooling2d(x, 4, 4, padding='same')
        x = tf.layers.flatten(x)
        for n in range(self.settings["conv_n_dense"]):
            print("\t",self.settings["conv_dense_size"]," unit layer")
            x = tf.layers.dense(
                                x,
                                self.settings["conv_dense_size"],
                                activation=tf.nn.elu
                                )
        return x

    def get_weights(self):
        return self.session.run(self.all_variables)
    def set_weights(self, weights):
        feed_dict = dict(zip(self.assign_ops, weights))
        self.session.run(self.assign_ops, feed_dict=feed_dict)
    def save_weights(self):
        self.saved_weights = self.get_weights()

    def create_weight_setting_ops(self):
        assign_ops = []
        assign_values = []
        for var in self.all_variables:
            shape, dtype = var.shape, var.dtype
            assign_val_placeholder_tf = tf.placeholder(shape=shape, dtype=dtype)
            assign_op_tf = var.assign(assign_val_placeholder_tf)
            assign_ops.append(assign_op_tf)
            assign_values.append(assign_val_placeholder_tf)
        return assign_ops, assign_values

    def save(self, step):
        return
        path = self.saver.save(self.session, "saved/"+self.name, global_step=step)
        print("Saved model at: "+path)
    def restore(self, savepoint):
        return
        # self.saver.restore(self.session, "saved/"+self.name+"/")
        self.saver.restore(self.session, "saved/"+savepoint)
    def __call__(self,x):
        return self.evaluate(x)
