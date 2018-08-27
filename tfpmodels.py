import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape

def ifa(N = 1000, K = 2, C = 3, P = 4):
    mixture_component_means = ed.Normal(loc=0., scale=1., sample_shape=(C,K), name='mixture_component_means')
    mixture_component_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(C,K), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=np.ones(K, dtype='float32'), sample_shape=(C,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=mixture_component_means, scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(N,),name='source')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(C, P), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='data_std')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
    return data

def cifa(N = 1000, K = 2, C = 3, P = 4):
    mixture_component_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(C,K), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=np.ones(K, dtype='float32'), sample_shape=(C,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(N,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(C, P), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='data_std')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
    return data


