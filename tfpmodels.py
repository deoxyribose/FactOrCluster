import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape

def independentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 2, n_features = 2, mixture_component_means_mean = 0., mixture_component_means_std = 1., mixture_component_std_concentration = 1., mixture_component_std_rate=1.,mixture_weights_concentration=None,data_std_concentration=1.,data_std_rate=1.):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components_in_mixture, dtype='float32')
    mixture_component_means = ed.Normal(loc=mixture_component_means_mean, scale=mixture_component_means_std, sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_means')
    mixture_component_std = ed.Gamma(concentration=mixture_component_std_concentration, rate=mixture_component_std_rate, sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, sample_shape=(n_sources,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data_std = ed.Gamma(concentration=data_std_concentration, rate=data_std_rate, sample_shape=(1,n_features), name='data_std')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
    return data

def centeredIndependentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 2, n_features = 2, mixture_component_std_concentration = 1., mixture_component_std_rate=1.,mixture_weights_concentration=None,data_std_concentration=1.,data_std_rate=1.):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components_in_mixture, dtype='float32')
    mixture_component_std = ed.Gamma(concentration=mixture_component_std_concentration, rate=mixture_component_std_rate, sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_std')
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, sample_shape=(n_sources,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data_std = ed.Gamma(concentration=data_std_concentration, rate=data_std_rate, sample_shape=(1,n_features), name='data_std')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
    return data

def mixtureOfGaussians(n_observations = 1000, n_components = 2, n_features = 2, mixture_component_means_mean = 0., mixture_component_means_std = 1., mixture_component_covariances_cholesky_df = None, mixture_component_covariances_cholesky_scale_tril=None,mixture_weights_concentration=None):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components, dtype='float32')
    if mixture_component_covariances_cholesky_df is None:
        mixture_component_covariances_cholesky_df = n_features + 2.
    if mixture_component_covariances_cholesky_scale_tril is None:
        mixture_component_covariances_cholesky_scale_tril = (1./(mixture_component_covariances_cholesky_df - n_features - 1.))*tf.eye(n_features)
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, name='mixture_weights')
    mixture_component_means = ed.Normal(loc=mixture_component_means_mean, scale=mixture_component_means_std, sample_shape=(n_components, n_features), name='mixture_component_means')
    mixture_component_covariances_cholesky = ed.Wishart(
        df=mixture_component_covariances_cholesky_df, scale_tril=mixture_component_covariances_cholesky_scale_tril, sample_shape=(n_components), input_output_cholesky=True, name='mixture_component_covariances_cholesky')
    return ed.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mixture_weights),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mixture_component_means,
                    scale_tril=mixture_component_covariances_cholesky,
                    name='component'), sample_shape=(n_observations,), name='data')

'''
def centeredIndependentFactorAnalysisTest(n_observations, factor_loadings, mixture_weights, mixture_component_std, data_std):
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
    data = ed.Normal(loc=data_mean, scale=data_std, name='data')
    return data
'''

def centeredIndependentFactorAnalysisTest(n_observations, mc_samples, factor_loadings, mixture_weights, mixture_component_std, data_std):
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(mc_samples, n_observations),name='sources')
    data_mean = tf.einsum('bik,kj->ijb', sources, factor_loadings, name='data_mean')
    data = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=tf.ones(mc_samples)/mc_samples),
                components_distribution=tfd.Normal(loc=data_mean, scale=data_std[:,:,None], name='data'), name='mc_approx')
    return data

def mixtureOfGaussiansTest(n_observations, mixture_weights, mixture_component_means, mixture_component_covariances_cholesky):
    return ed.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mixture_weights),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mixture_component_means,
                    scale_tril=mixture_component_covariances_cholesky,
                    name='component'), sample_shape=(n_observations,), name='data')

# hyperparameters as dict
#def centeredIndependentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 2, n_features = 2, hyperparameters = {'mixture_component_std_rate':1.,'mixture_weights_concentration':1.,'data_std_concentration':1.,'data_std_rate':1.}):
#    # check that Dirichlet concentration is a vector
#    if hasattr(tmp,'shape') and hyperparameters['mixture_weights_concentration'].shape == (n_components_in_mixture,):
#        pass
#    else:
#        hyperparameters['mixture_weights_concentration'] = np.ones(n_components_in_mixture, dtype='float32')
#    mixture_component_std = ed.Gamma(concentration=hyperparameters['mixture_component_std_concentration'], rate=hyperparameters['mixture_component_std_rate'], sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_std')
#    mixture_weights = ed.Dirichlet(concentration=hyperparameters['mixture_weights_concentration'], sample_shape=(n_sources,), name='mixture_weights')
#    sources = ed.Independent(
#        tfd.MixtureSameFamily(
#            mixture_distribution=tfd.Categorical(probs=mixture_weights),
#            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_std), scale=mixture_component_std, name='mixture_component')),
#        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
#    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
#    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
#    data_std = ed.Gamma(concentration=hyperparameters['data_std_concentration'], rate=hyperparameters['data_std_rate'], sample_shape=(1,n_features), name='data_std')
#    data = ed.Normal(loc=data_mean, scale=data_std, name='data')  
#    return data
