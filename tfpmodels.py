import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

#def ifa(N, K, C, P):
#    mus = ed.Normal(loc=0., scale=1., sample_shape=(K,C), name='factor_means')
#    sigmas = ed.InverseGamma(concentration=1., rate=1., sample_shape=(K,C), name='factor_std')
#    weights = ed.Dirichlet(concentration=np.ones(K), sample_shape=(C,), name='factor_mix_weights')
#    def unimixture(mu, sigma, weight):
#        return ed.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=weight),
#                                    components_distribution=tfd.Normal(loc=mu, scale=sigma, name='component'),
#                                    sample_shape=(N,1))
#    z = tf.concat([unimixture(mu, sigma, weight) for mu, sigma, weight in zip(tf.unstack(mus), tf.unstack(sigmas), tf.unstack(weights))],axis=1)
#    B = ed.Normal(loc=0., scale=1., sample_shape=(K, P), name='factors')
#    X = tf.matmul(z, B, name='latent')
#
#    noise_sigma = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='output_noise')
#    Y = ed.Normal(loc=X, scale=noise_sigma, name='data')  
#    return Y

def ifa(N = 1000, K = 2, C = 3, P = 4):
    mixture_component_means = ed.Normal(loc=0., scale=1., sample_shape=(K,C), name='mixture_component_means')
    mixture_component_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(C,K), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=np.ones(K), sample_shape=(C,), name='mixture_weights')
    z = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=mixture_component_means, scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(N,),name='source')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(C, P), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    output_noise = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='output_noise')
    data = ed.Normal(loc=X, scale=output_noise, name='data')  
    return data

def cifa(N = 1000, K = 2, C = 3, P = 4):
    mixture_component_std = ed.InverseGamma(concentration=1., rate=1., sample_shape=(C,K), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=np.ones(K), sample_shape=(C,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(N,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(C, P), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    output_noise = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='output_noise')
    data = ed.Normal(loc=X, scale=output_noise, name='data')  
    return data


model = cifa(2000, 2, 2, 2)
    