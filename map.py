import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape


class Mapper:

    _positive_distributions = [tfd.InverseGamma, tfd.Gamma]
    _simplex_distributions = [tfd.Dirichlet]

    def __init__(self, model, model_name, observed_variable_names, *args, **kwargs):
        self.model = model
        self.model_name = model_name
        self.observed_variable_names = observed_variable_names
        self.loss_fn = ed.make_log_joint_fn(model) 
        self._args = args
        self._kwargs = kwargs
        with tape() as self.tape:
            self.output = self.model(*args, **kwargs)
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
            self.variable_names = [key for key in self.tape.keys() if key not in self.observed_variable_names]
            self.variable_shapes = {key: self.tape[key].shape for key in self.variable_names}
            self.variable_dist = {key: self.tape[key].distribution for key in self.variable_names}
            self.unconstrained_variables = {key: tf.get_variable(key, shape=self.variable_shapes[key]) for key in self.variable_names}
            self.variables = {key: self.wrap(val, self.variable_dist[key]) for key, val in self.unconstrained_variables.items()}
        
    def wrap(self, variable, distribution):
        if distribution.__class__ in self._positive_distributions:
            return tfp.trainable_distributions.softplus_and_shift(variable)
        if distribution.__class__ in self._simplex_distributions:
            return tf.nn.softmax(variable)
        return variable

    def map_loss(self, **kwargs):
        return self.loss_fn(*self._args, **self._kwargs, **self.variables, **kwargs)

    def map_optimizer(self, **kwargs):
        loss = -self.map_loss(**kwargs)
        return loss, tf.contrib.opt.ScipyOptimizerInterface(loss, self.unconstrained_variables.values())