import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
ed = tfp.edward2
import numpy as np
from itertools import product
from future_features import tape, SoftmaxCentered

from tfopt import PylbfgsInterface

import pdb

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
        positivity_shift = tf.convert_to_tensor(0.,dtype=tf.float64)
        diag_shift = tf.convert_to_tensor(0.,dtype=tf.float64)
        #positivity_shift = tf.convert_to_tensor(1e-3,dtype=tf.float64)
        #diag_shift = tf.convert_to_tensor(1.,dtype=tf.float64)
        
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
        with tf.variable_scope(self.model_name, reuse=tf.AUTO_REUSE):
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

    def test_model(self, inferred_latent_variables=None):
        if inferred_latent_variables is None:
            inferred_latent_variables = self.variables
        def replace_latents(**inferred_latent_variables):
            """When called inside a with ed.interception clause, this replaces sampling ops of variable "name", with the corresponding tensor in inferred_latent_variables"""
            def interceptor(model, *args, **kwargs):
                name = kwargs.pop("name")
                for key in inferred_latent_variables:
                    if name == key:
                        kwargs["value"] = inferred_latent_variables[key]
                return model(*args, **kwargs)
            return interceptor
        with ed.interception(replace_latents(**inferred_latent_variables)):
            return self.model(**self._kwargs)

    def neg_log_lik(self, data, inferred_latent_variables=None):
        model_MAP = self.test_model(inferred_latent_variables)
        return -tf.reduce_mean(model_MAP.distribution.log_prob(data))

class IFA_MAPEM(Mapper):
    def __init__(self, model, model_name, observed_variable_names, *args, **kwargs):
        super().__init__(model, model_name, observed_variable_names, *args, **kwargs)
        self.current = {key: tf.Variable(initial_value=tf.zeros(val, dtype='float64')) for key, val in self.variable_shapes.items()}

    def MAPEMsteps(self, data):
        mixture_weights = self.current['mixture_weights']
        mixture_component_var = self.current['mixture_component_var']
        factor_loadings = self.current['factor_loadings']
        data_var = self.current['data_var']
        
        jitter = 1e-6  #jitter is not automatically matched!
        data = tf.convert_to_tensor(data, dtype='float64')
        n_observations = data.shape[0].value
        n_sources, n_components_in_mixture = mixture_component_var.shape.as_list()
        n_features = factor_loadings.shape[1].value

        combinations = tf.one_hot(np.array(list(
            product(range(n_components_in_mixture), repeat=n_sources))).astype('float64'), 
            depth=n_components_in_mixture, dtype='float64') # combinations x sources x component indicator
        n_combinations = combinations.shape[0]
        all_mixture_weights = tf.reduce_prod(tf.reduce_sum(combinations * mixture_weights[None, :, :], axis=-1), axis=1)
        all_mixture_vars = tf.reduce_sum(combinations * mixture_component_var[None, :, :], axis=-1)

        all_mixture_cov_factor = tf.einsum('sf,cs->csf', factor_loadings, tf.sqrt(all_mixture_vars))
        all_mixture_cov = tf.einsum('csf,csg->cfg', all_mixture_cov_factor, all_mixture_cov_factor) + (jitter + data_var)*tf.eye(n_features, dtype='float64')[None,:,:]
        all_mixture_cov_chol = tf.cholesky(all_mixture_cov)

        marginal_eval = ed.MultivariateNormalTriL(loc=tf.zeros((n_combinations, n_features), dtype='float64'),
                                                scale_tril=all_mixture_cov_chol, sample_shape=(n_observations,)).distribution.log_prob(data[:,None,:])
        log_posterior_assignment = tf.log(all_mixture_weights)[None,:] + marginal_eval
        log_posterior_assignment -= tf.reduce_logsumexp(log_posterior_assignment, axis=-1, keepdims=True)
        posterior_assignment = tf.exp(log_posterior_assignment)

        posterior_component_inv_cov = tf.matmul(factor_loadings, factor_loadings * (1./data_var), transpose_b=True) + tf.map_fn(tf.diag, 1./all_mixture_vars)
        posterior_component_inv_cov_chol = tf.cholesky(posterior_component_inv_cov)
        posterior_component_cov = tf.stack([tf.transpose(tf.cholesky_solve(L, tf.eye(n_sources, dtype='float64'))) for L in tf.unstack(posterior_component_inv_cov_chol)], axis=0)
        
        posterior_component_mean = tf.stack([tf.transpose(tf.cholesky_solve(L, tf.matmul(factor_loadings, data / data_var, transpose_b=True))) for L in tf.unstack(posterior_component_inv_cov_chol)], axis=1)
        posterior_component_second_moment = posterior_component_cov[None,:,:,:] + tf.einsum('nqk,nql->nqkl', posterior_component_mean, posterior_component_mean) 
        posterior_component_second_moment_diag = tf.einsum('nqij,nqij->nqi', posterior_component_second_moment,
            tf.tile(tf.eye(n_sources, dtype='float64')[None, None, :, :], [n_observations, n_combinations, 1,1]))

        posterior_marginal_source_mean = tf.einsum('nq,nqi->ni', posterior_assignment, posterior_component_mean)
        posterior_marginal_source_second_moment = tf.einsum('nq,nqij->nij', posterior_assignment, posterior_component_second_moment)
        posterior_marginal_source_second_moment_sum = tf.reduce_sum(posterior_marginal_source_second_moment, axis=0)
        
        updates = {}
        #does not handle arbitrary mean
        concentration = self.tape['mixture_weights'].distribution.parameters['concentration']
        updates['mixture_weights'] = (concentration[None,:] - 1. + tf.einsum('qkc,nq->kc', combinations, posterior_assignment))/(n_observations + tf.reduce_sum(concentration - 1.))
        
        concentration = self.tape['mixture_component_var'].distribution.parameters['concentration']
        rate = self.tape['mixture_component_var'].distribution.parameters['rate']
        updates['mixture_component_var'] = (2.*rate + tf.reduce_sum(tf.einsum('qkc,nq,nqk->qkc', combinations, posterior_assignment, posterior_component_second_moment_diag), axis=0))/(2.*(1.+concentration) + tf.einsum('qkc,nq->kc', combinations, posterior_assignment))
        #updates['mixture_component_var'] = (2.*rate + tf.einsum('qkc,nq,nk->kc', combinations, posterior_assignment, posterior_marginal_source_cov_diag + tf.square(posterior_marginal_source_mean)))/(2.*(1.+concentration) + tf.einsum('qkc,nq->kc', combinations, posterior_assignment))
        
        scale = self.tape['factor_loadings'].distribution.parameters['scale']
        updates['factor_loadings'] = (tf.linalg.cholesky_solve(tf.linalg.cholesky(posterior_marginal_source_second_moment_sum + (data_var/tf.square(scale))*tf.eye(n_sources, dtype='float64')), tf.einsum('ns,nd->sd',posterior_marginal_source_mean,data)))
        #updates['factor_loadings'] = tf.transpose(tf.linalg.solve(posterior_marginal_source_second_moment_sum, tf.transpose(tf.matmul(data, posterior_marginal_source_mean, transpose_a=True))))
        
        concentration = self.tape['data_var'].distribution.parameters['concentration']
        rate = self.tape['data_var'].distribution.parameters['rate']
        updates['data_var'] = (tf.reduce_sum(tf.square(data)) + 
                               tf.reduce_sum(tf.matmul(factor_loadings, factor_loadings, transpose_b=True) * posterior_marginal_source_second_moment_sum) - 
                               2.*tf.reduce_sum(data * tf.matmul(posterior_marginal_source_mean, factor_loadings)) + 
                               2.*rate)/(n_observations*n_features+2.*(1.+concentration))    
        return updates

    def MAPEM_optimizer(self, **kwargs):
        set_current = tf.group([tf.assign(self.current[key], self.variables[key]) for key in self.current.keys()])
        steps = self.assigner(**self.MAPEMsteps(data=kwargs['data']))
        map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        
        return map_neg_log_joint, set_current, steps
    
    def MAPEM_enhanced_map_neg_log_joint_fn(self, **kwargs):
        set_current = tf.group([tf.assign(self.current[key], self.variables[key]) for key in self.current.keys()])
        with tf.control_dependencies([set_current]):
            emsteps = self.MAPEMsteps(data=kwargs['data'])
        regularizer = tf.reduce_sum([tf.linalg.norm(emsteps[key] - self.variables[key]) for key in emsteps.keys])
        return -self.log_joint_fn(*self._args, **self._kwargs, **emsteps, **kwargs) + regularizer

    def bfgs_optimizer(self, enhanced=False, **kwargs):
        if enhanced:
            map_neg_log_joint = self.MAPEM_enhanced_map_neg_log_joint_fn(**kwargs)
        else:
            map_neg_log_joint = self.map_neg_log_joint_fn(**kwargs)
        return map_neg_log_joint, tf.contrib.opt.ScipyOptimizerInterface(map_neg_log_joint, self.unconstrained_variables.values(), method='BFGS')
   
                