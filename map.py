import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape


class Mapper:

    _positive_distributions = [tfd.InverseGamma, tfd.Gamma]
    _simplex_distributions = [tfd.Dirichlet]

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.loss_fn = ed.make_log_joint_fn(model) 
        self._args = args
        self._kwargs = kwargs
        with tape() as self.tape:
            self.model(*args, **kwargs)
        self.variable_names = self.tape.keys()
        self.variable_shapes = {key: val.shape for key, val in self.tape.items()}
        self.variable_dist = {key: val.distribution for key, val in self.tape.items()}
        self.unconstrained_variables = {key: tf.get_variable(key, shape=self.variable_shapes[key]) for key in self.tape.keys()}
        self.variables = {key: self.wrap(val, self.variable_dist[key]) for key, val in self.unconstrained_variables.items()}
        
    def wrap(self, variable, distribution):
        if distribution.__class__ in self._positive_distributions:
            return tfp.trainable_distributions.softplus_and_shift(variable)
        if distribution.__class__ in self._simplex_distributions:
            return tf.nn.softmax(variable)
        return variable

    def map_loss(self):
        return self.loss_fn(*self._args, **self._kwargs, **self.variables)
