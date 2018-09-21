import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
ed = tfp.edward2
import numpy as np

from future_features import tape, SoftmaxCentered

from tfopt import PylbfgsInterface


class Mapper:
    _positive_distributions = [tfd.InverseGamma, tfd.Gamma]
    _simplex_distributions = [tfd.Dirichlet]
    _tril_distributions = [tfd.Wishart]

    def __init__(self, model, model_name, observed_variable_names, *args, **kwargs):
        self.model = model
        self.model_name = model_name
        self.observed_variable_names = observed_variable_names
        self.log_joint_fn = ed.make_log_joint_fn(model) 
        self._args = args
        self._kwargs = kwargs
        with tape() as self.tape:
            self.output = self.model(*args, **kwargs)
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
            self.variable_names = [key for key in self.tape.keys() if key not in self.observed_variable_names]
            self.variable_dist = {key: self.tape[key].distribution for key in self.variable_names}
            self.variable_shapes = {key: self.tape[key].shape for key in self.variable_names}
            self.variable_dtypes = {key: self.tape[key].dtype for key in self.variable_names}
            self.transforms = {key: self.get_bijector(self.tape[key]) for key in self.variable_names}
            self.unconstrained_variable_shapes = {key: self.transforms[key].inverse_event_shape(val) for key, val in self.variable_shapes.items()}
            self.unconstrained_variables = {key: tf.get_variable(key, shape=self.unconstrained_variable_shapes[key], dtype='float64') for key in self.variable_names}
            self.variables = {key: self.transforms[key].forward(val) for key, val in self.unconstrained_variables.items()}


    def get_bijector(self, random_variable):
        positivity_shift = tf.convert_to_tensor(1e-3,dtype=tf.float64)
        diag_shift = tf.convert_to_tensor(1.,dtype=tf.float64)
        
        distribution = random_variable.distribution
        if distribution.__class__ in self._positive_distributions:
            #return tfb.Softplus() #tfp.trainable_distributions.softplus_and_shift(variable)
            return tfb.Chain([tfb.AffineScalar(shift=positivity_shift), tfb.Exp()], name="scaled_sigmoid")
            #return tfb.Chain([tfb.Affine(shift=1e-4), tfb.Softplus()], name="softplus_and_shift")
        elif distribution.__class__ in self._simplex_distributions:
            return SoftmaxCentered()
        elif distribution.__class__ in self._tril_distributions:
            return tfb.ScaleTriL(diag_shift=diag_shift)
        else:
            return tfb.Identity()

    def replace_bijector(self, key, bijector):
        self.transforms.update({key: bijector})
        self.unconstrained_variable_shapes.update({key: self.transforms[key].inverse_event_shape(self.variable_shapes[key])})
        self.unconstrained_variables.update({key: tf.get_variable(key, shape=self.unconstrained_variable_shapes[key], dtype='float64')})
        self.variables.update({key: self.transforms[key].forward(self.unconstrained_variables[key])})

    def append_bijector(self, key, bijector, prepend=False):
        if prepend:
            self.replace_bijector(key, tfb.Chain([self.transforms[key], bijector], name='{}_bijector_chain'.format(key)))
        else:
            self.replace_bijector(key, tfb.Chain([bijector, self.transforms[key]], name='{}_bijector_chain'.format(key)))

#    def get_bijector64(self, random_variable):
#        shift = tf.convert_to_tensor(1e-3,dtype=tf.float64)
#        scale = tf.convert_to_tensor(1e3,dtype=tf.float64)
#        return tfb.Chain([tfb.AffineScalar(shift=shift,scale=scale), tfb.Sigmoid()], name="scaled_sigmoid")

    def map_neg_log_joint_fn(self, **kwargs):
        return -self.log_joint_fn(*self._args, **self._kwargs, **self.variables, **kwargs)

    def bfgs_optimizer(self, **kwargs):
        map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        return map_neg_log_joint, tf.contrib.opt.ScipyOptimizerInterface(map_neg_log_joint, self.unconstrained_variables.values(), method='BFGS')

    def l_bfgs_optimizer(self, **kwargs):
        map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        return map_neg_log_joint, tf.contrib.opt.ScipyOptimizerInterface(map_neg_log_joint, self.unconstrained_variables.values(), method='L-BFGS-B')

    def pybfgs_optimizer(self, **kwargs):
        map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        return map_neg_log_joint, PylbfgsInterface(map_neg_log_joint, self.unconstrained_variables.values())


    def cg_optimizer(self, **kwargs):
        map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        return map_neg_log_joint, tf.contrib.opt.ScipyOptimizerInterface(map_neg_log_joint, self.unconstrained_variables.values(), method='CG')

    def adam_optimizer(self, learning_rate = 0.005, **kwargs):
        map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        return map_neg_log_joint, tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(map_neg_log_joint)

    def assigner(self, **kwargs):
        assign_ops = []
        for key, val in kwargs.items():
            assign_ops.append(tf.assign(self.unconstrained_variables[key], self.transforms[key].inverse(val)))
        return tf.group(assign_ops)

#class IFA_EM(Mapper):
#    def __init__(self, model, model_name, observed_variable_names, *args, **kwargs):
#        super(IFA_EM).__init__(model, model_name, observed_variable_names, *args, **kwargs)

        
        #self.posterior_eval = 
        #self.po