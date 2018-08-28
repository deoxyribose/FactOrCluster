import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape

def independentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 3, n_features = 4):
    mixture_component_means = ed.Normal(loc=0., scale=1., sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_means')
    mixture_component_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=np.ones(n_components_in_mixture, dtype='float32'), sample_shape=(n_sources,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=mixture_component_means, scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='source')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,n_features), name='data_std')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
    return data

def centeredIndependentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 3, n_features = 4):
    mixture_component_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=np.ones(n_components_in_mixture, dtype='float32'), sample_shape=(n_sources,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,n_features), name='data_std')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
    return data