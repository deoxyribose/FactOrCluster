import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

def ifa(N, K, C, P):
    mus = ed.Normal(loc=0., scale=1., sample_shape=(K,C), name='factor_means')
    sigmas = ed.InverseGamma(concentration=1., rate=1., sample_shape=(K,C), name='factor_std')
    weights = ed.Dirichlet(concentration=np.ones(K), sample_shape=(C,), name='factor_mix_weights')
    def unimixture(mu, sigma, weight):
        return ed.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=weight),
                                    components_distribution=tfd.Normal(loc=mu, scale=sigma, name='component'),
                                    sample_shape=(N,1))
    z = tf.concat([unimixture(mu, sigma, weight) for mu, sigma, weight in zip(tf.unstack(mus), tf.unstack(sigmas), tf.unstack(weights))],axis=1)
    B = ed.Normal(loc=0., scale=1., sample_shape=(K, P), name='factors')
    X = tf.matmul(z, B, name='latent')

    noise_sigma = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='output_noise')
    Y = ed.Normal(loc=X, scale=noise_sigma, name='data')  
    return Y

def cifa(N, K, C, P):
    sigmas = ed.InverseGamma(concentration=1., rate=1., sample_shape=(K,C), name='factor_std')
    weights = ed.Dirichlet(concentration=np.ones(K), sample_shape=(C,), name='factor_mix_weights')
    def centered_unimixture(sigma, weight):
        return ed.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=weight),
                                    components_distribution=tfd.Normal(loc=tf.zeros_like(sigma), scale=sigma, name='component'),
                                    sample_shape=(N,1))
    z = tf.concat([centered_unimixture(sigma, weight) for sigma, weight in zip(tf.unstack(sigmas), tf.unstack(weights))],axis=1)
    B = ed.Normal(loc=0., scale=1., sample_shape=(K, P), name='factors')
    X = tf.matmul(z, B, name='latent')

    noise_sigma = ed.InverseGamma(concentration=1., rate=1., sample_shape=(1,P), name='output_noise')
    Y = ed.Normal(loc=X, scale=noise_sigma, name='data')  
    return Y



model = cifa(2000, 2, 2, 2)
    