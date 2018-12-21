import tensorflow as tf
import numpy as np

class ppo_discrete_model:
    def __init__(self, name, state_size, action_size, session, epsilon=0.8, lr=0.0001, loss_weights=(1.0,0.001,1.0), pixels=False):
        self.session = session
        self.name = name
        c1,c2,c3 = loss_weights
        self.weight_loss_policy = c1
        self.weight_loss_entropy = c2
        self.weight_loss_value = c3
        with tf.variable_scope("ppo_discrete") as scope:
            self.states_tf = tf.placeholder(dtype=tf.float32, shape=(None, *state_size), name='states')
            self.actions_tf = tf.placeholder(dtype=tf.float32, shape=(None, action_size), name='actions')
            self.advantages_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='advantages')
            self.target_values_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='target_values')
            self.old_probabilities_tf = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='old_probs')
            # self.probabilities_tf = self.create_net(name='policy_net', input=self.states_tf, output_size=action_size, output_activation=tf.nn.softmax)
            # self.values_tf        = self.create_net(name='value_net',  input=self.states_tf, output_size=1,           output_activation=None)
            self.probabilities_tf, self.values_tf = self.create_net(name='policy_net', input=self.states_tf, output_activation=tf.nn.softmax, output_size=action_size, add_value_head=True, pixels=pixels)
            self.training_ops = self.create_training_ops(
                                                            self.actions_tf,
                                                            self.probabilities_tf,
                                                            self.old_probabilities_tf,
                                                            self.advantages_tf,

                                                            self.values_tf,
                                                            self.target_values_tf,
                                                            epsilon=epsilon,
                                                            lr=lr,
                                                        )
            self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.init_ops = tf.variables_initializer(self.all_variables)
            self.saver = tf.train.Saver(self.all_variables, self.name)
        session.run(self.init_ops)
    def evaluate(self, states, actions=None):
        run_list = [self.probabilities_tf,self.values_tf]
        feed_dict = {self.states_tf : states}
        probs, vals = self.session.run(run_list, feed_dict=feed_dict)
        return probs, vals
        # if actions is None:
        #     return probs, vals
        # else: #This is if we request the probability of particular actions...
        #     return [[probs[i][a]] for i,a in enumerate(actions)], vals

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
        entropy_tf = tf.reduce_sum(-tf.multiply(probabilities_tf, tf.log(probabilities_tf)), axis=1)
        action_prob_tf = tf.reduce_sum(tf.multiply(actions_tf, probabilities_tf), axis=1, keep_dims=True)
        ratio_tf = tf.div(action_prob_tf, old_probabilities_tf)
        ratio_clipped_tf = tf.clip_by_value(ratio_tf, 1-epsilon, 1+epsilon)
        loss_clip_tf = tf.reduce_mean(tf.minimum( tf.multiply(ratio_tf,advantages_tf), tf.multiply(ratio_clipped_tf,advantages_tf) ) )
        loss_entropy_tf = tf.reduce_mean(entropy_tf)
        loss_value_tf = tf.losses.mean_squared_error(values_tf, target_values_tf)
        loss_tf = -self.weight_loss_policy * loss_clip_tf - self.weight_loss_entropy * loss_entropy_tf + self.weight_loss_value * loss_value_tf
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_tf)

    def create_net(self, name=None, input=None, output_size=None, output_activation=tf.nn.elu, add_value_head=False, pixels=False):
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
        x = input_tensor
        for n in range(6):
            x = tf.layers.dense(
                                x,
                                40,
                                activation=tf.nn.elu,
                                )
        return x
    def create_conv(self, input_tensor):
        print(input_tensor)
        x = tf.layers.average_pooling2d(input_tensor, 2, 2, padding='same')
        x = tf.reduce_mean(x, axis=3, keepdims=True)
        for n in range(3):
            x = tf.layers.conv2d(
                                x,
                                32//2**n,
                                (5, 5),
                                padding='same',
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                )
        x = tf.layers.average_pooling2d(x, 4, 4, padding='same')
        x = tf.layers.flatten(x)
        for n in range(2):
            x = tf.layers.dense(
                                x,
                                256,
                                activation=tf.nn.elu
                                )
        return x

    def save(self, step):
        path = self.saver.save(self.session, "saved/"+self.name, global_step=step)
        print("Saved model at: "+path)
    def restore(self, savepoint):
        # self.saver.restore(self.session, "saved/"+self.name+"/")
        self.saver.restore(self.session, "saved/"+savepoint)
    def __call__(self,x, actions=None):
        return self.evaluate(x, actions=actions)
