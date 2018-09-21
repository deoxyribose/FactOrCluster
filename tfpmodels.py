import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from itertools import product
from future_features import tape

jitter = 1e-04
print(jitter)

def independentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 2, n_features = 2, mixture_component_means_mean = 0., mixture_component_means_var = 1., mixture_component_var_concentration = 1., mixture_component_var_rate=1.,mixture_weights_concentration=None,data_var_concentration=1.,data_var_rate=1.):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components_in_mixture, dtype='float64')
    mixture_component_means = ed.Normal(loc=mixture_component_means_mean, scale=tf.sqrt(mixture_component_means_var), sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_means')
    mixture_component_var = ed.Gamma(concentration=mixture_component_var_concentration, rate=tf.sqrt(mixture_component_var_rate), sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_var')
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, sample_shape=(n_sources,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_var), scale=tf.sqrt(mixture_component_var), name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings/tf.linalg.norm(factor_loadings), name='data_mean')
    data_var = ed.Gamma(concentration=data_var_concentration, rate=data_var_rate, sample_shape=(1,n_features), name='data_var')
    data = ed.Normal(loc=data_mean, scale=tf.sqrt(data_var), name='data')  
    return data

def centeredIndependentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 2, n_features = 2, mixture_component_var_concentration = 1., mixture_component_var_rate=1.,mixture_weights_concentration=None,data_var_concentration=1.,data_var_rate=1.):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components_in_mixture, dtype='float64')
    mixture_component_var = ed.Gamma(concentration=mixture_component_var_concentration, rate=mixture_component_var_rate, sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_var')
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, sample_shape=(n_sources,), name='mixture_weights')
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_var), scale=tf.sqrt(mixture_component_var), name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    data_mean = tf.matmul(sources, factor_loadings/tf.linalg.norm(factor_loadings, axis=1, keepdims=True), name='data_mean')
    data_var = ed.Gamma(concentration=data_var_concentration, rate=data_var_rate, sample_shape=(1,n_features), name='data_var')
    data = ed.Normal(loc=data_mean, scale=tf.sqrt(data_var), name='data')  
    return data


def condition_number(matrix):
    singular_values = tf.svd(matrix,compute_uv=False)
    largest_singular_value = tf.reduce_max(singular_values,axis=1)
    smallest_singular_value = tf.reduce_max(singular_values,axis=1)
    return largest_singular_value,smallest_singular_value,largest_singular_value/smallest_singular_value

def centeredMarginalizedIndependentFactorAnalysis(n_observations = 1000, n_sources = 2, n_components_in_mixture=2, n_features = 2, mixture_component_var_concentration = 1., mixture_component_var_rate=1.,mixture_weights_concentration=None,data_var_concentration=1.,data_var_rate=1.):

    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components_in_mixture, dtype='float64')
    combinations = tf.one_hot(np.array(list(
        product(range(n_components_in_mixture), repeat=n_sources))).astype('float64'), 
        depth=n_components_in_mixture, dtype='float64') # combinations x sources x component indicator
    n_combinations = combinations.shape[0]
    mixture_component_var = ed.Gamma(concentration=tf.convert_to_tensor(mixture_component_var_concentration, dtype='float64'),
        rate=tf.convert_to_tensor(mixture_component_var_rate, dtype='float64'),
        sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_var')
    mixture_weights = ed.Dirichlet(concentration=tf.convert_to_tensor(mixture_weights_concentration, dtype='float64'),
        sample_shape=(n_sources,), name='mixture_weights')
    factor_loadings = ed.Normal(loc=tf.convert_to_tensor(0., dtype='float64'), 
        scale=tf.convert_to_tensor(1., dtype='float64'), 
        sample_shape=(n_sources, n_features), name='factor_loadings')
    #factor_loadings /= tf.linalg.norm(factor_loadings, axis=1, keepdims=True)
    data_var = ed.Gamma(concentration=tf.convert_to_tensor(data_var_concentration, dtype='float64'),
        rate=tf.convert_to_tensor(data_var_rate, dtype='float64'),
        sample_shape=(n_features,), name='data_var')
    
    all_mixture_weights = tf.reduce_prod(tf.reduce_sum(combinations * mixture_weights[None, :, :], axis=-1), axis=1)
    all_mixture_vars = tf.reduce_sum(combinations * mixture_component_var[None, :, :], axis=-1)

    cov_factor = tf.einsum('sf,cs->csf', factor_loadings, all_mixture_vars)
    covmat = tf.einsum('csf,csg->cfg', cov_factor, cov_factor) + tf.diag(jitter + data_var)[None,:,:]
    
    # conditioning of covariance matrices
    #largest_magnitude_covariance = tf.reduce_max(tf.abs(covmat),axis=[1,2],keepdims=True)
    #covmat /= largest_magnitude_covariance
    #covmat = tf.cast(covmat,tf.float64)
    #covmat /= tf.Print(largest_magnitude_covariance,[largest_magnitude_covariance],summarize=100,message='\n Normalizing covmats by')

    data = ed.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=all_mixture_weights),
        components_distribution=tfd.MultivariateNormalTriL(
            loc=tf.zeros((n_combinations, n_features),dtype=tf.float64),
            scale_tril=tf.cholesky(covmat),
            #scale_tril=tf.cast(tf.sqrt(largest_magnitude_covariance),tf.float64)*tf.linalg.cholesky(tf.Print(covmat,[covmat],summarize=100)),
            #scale_tril=tf.sqrt(largest_magnitude_covariance)*tf.linalg.cholesky(tf.Print(covmat,[covmat],summarize=100)),
            #scale_tril=tf.sqrt(largest_magnitude_covariance)*tf.linalg.cholesky(tf.Print(covmat, [tf.check_numerics(covmat,message='covmat is nan or inf'),'Determinants:',tf.linalg.det(covmat),'max_singvals,min_singvals,cond_numbers::',condition_number(covmat)],summarize=100,message='\n')),
            name='data_component'),
            sample_shape=(n_observations,), name='data')
    return data


def centeredMarginalizedIndependentFactorAnalysisTest(n_observations, mixture_weights, mixture_component_var, factor_loadings, data_var):
    n_sources, n_features = factor_loadings.shape
    n_components_in_mixture = mixture_weights.shape[1]
    combinations = tf.one_hot(np.array(list(
        product(range(n_components_in_mixture), repeat=n_sources))).astype('float64'), 
        depth=n_components_in_mixture, dtype='float64') # combinations x sources x component indicator
    n_combinations = combinations.shape[0]

    all_mixture_weights = tf.reduce_prod(tf.reduce_sum(combinations * mixture_weights[None, :, :], axis=-1), axis=1)
    all_mixture_vars = tf.reduce_sum(combinations * mixture_component_var, axis=-1)
    #factor_loadings /= tf.linalg.norm(factor_loadings, axis=1, keepdims=True)
    cov_factor = tf.einsum('sf,cs->csf', factor_loadings, all_mixture_vars)
    covmat = tf.einsum('csf,csg->cfg', cov_factor, cov_factor) + tf.diag(jitter + data_var)[None,:,:]
    
    # conditioning of covariance matrices
    #largest_magnitude_covariance = tf.reduce_max(tf.abs(covmat),axis=[1,2],keepdims=True)
    #covmat /= largest_magnitude_covariance
    #covmat = tf.cast(covmat,tf.float64)
    #covmat /= tf.Print(largest_magnitude_covariance,[largest_magnitude_covariance],summarize=100,message='\n Normalizing covmats by')

    data = ed.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=all_mixture_weights),
        components_distribution=tfd.MultivariateNormalTriL(
            loc=tf.zeros((n_combinations, n_features),dtype=tf.float64),
            scale_tril=tf.linalg.cholesky(covmat),
            #scale_tril=tf.cast(tf.sqrt(largest_magnitude_covariance),tf.float64)*tf.linalg.cholesky(tf.Print(covmat,[covmat],summarize=100)),
            #scale_tril=tf.sqrt(largest_magnitude_covariance)*tf.linalg.cholesky(tf.Print(covmat,[covmat],summarize=100)),
            #scale_tril=tf.sqrt(largest_magnitude_covariance)*tf.linalg.cholesky(tf.Print(covmat, [tf.check_numerics(covmat,message='covmat is nan or inf'),'Determinants:',tf.linalg.det(covmat),'max_singvals,min_singvals,cond_numbers::',condition_number(covmat)],summarize=100,message='\n')),
            name='data_component'),
            sample_shape=(n_observations,), name='data')   
    return data




#    data = ed.MixtureSameFamily(
#        mixture_distribution=tfd.Categorical(probs=all_mixture_weights),
#        components_distribution=tfd.MultivariateNormalDiagPlusLowRank(loc=tf.zeros((n_combinations, n_features)), 
#        scale_perturb_diag=None,#jitter + all_mixture_vars, 
#        # set to None in order to use cholesky to find determinants when calculating gradients
#        # for justification, see line 253 in https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/distributions/python/ops/mvn_diag_plus_low_rank.py
#        # and line 201 in https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/linalg/linear_operator_low_rank_update.py
#        scale_perturb_factor=tf.einsum('sf,cs->cfs', factor_loadings, tf.sqrt(all_mixture_vars)), #tf.tile(tf.transpose(factor_loadings)[None,:,:], [n_combinations, 1, 1]) , 
#        scale_diag=jitter + tf.tile(data_var, [n_combinations, 1]), name='data_component'), sample_shape=(n_observations,), name='data')  
#    return data

def mixtureOfGaussians(n_observations = 1000, n_components = 2, n_features = 2, mixture_component_means_mean = 0., mixture_component_means_var = 1., mixture_component_covariances_cholesky_df = None, mixture_component_covariances_cholesky_scale_tril=None,mixture_weights_concentration=None):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components, dtype='float64')
    if mixture_component_covariances_cholesky_df is None:
        mixture_component_covariances_cholesky_df = n_features + 2.
    if mixture_component_covariances_cholesky_scale_tril is None:
        mixture_component_covariances_cholesky_scale_tril = (1./(mixture_component_covariances_cholesky_df - n_features - 1.))*tf.eye(n_features, dtype='float64')
    mixture_weights = ed.Dirichlet(concentration=tf.convert_to_tensor(mixture_weights_concentration, dtype='float64'), 
        name='mixture_weights')
    mixture_component_means = ed.Normal(loc=tf.convert_to_tensor(mixture_component_means_mean, dtype='float64'), 
        scale=tf.sqrt(tf.convert_to_tensor(mixture_component_means_var, dtype='float64')), 
        sample_shape=(n_components, n_features), name='mixture_component_means')
    mixture_component_covariances_cholesky = ed.Wishart(
        df=mixture_component_covariances_cholesky_df, 
        scale_tril=tf.convert_to_tensor(mixture_component_covariances_cholesky_scale_tril, dtype='float64'),
        sample_shape=(n_components), input_output_cholesky=True, name='mixture_component_covariances_cholesky')

    return ed.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mixture_weights),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mixture_component_means,
                    scale_tril=mixture_component_covariances_cholesky,
                    name='component'), sample_shape=(n_observations,), name='data')

def lowRankMixtureOfGaussians(n_observations = 1000, n_components = 2, n_sources = 2, n_features = 2, mixture_component_means_mean = 0., mixture_component_means_var = 1., mixture_component_covariances_cholesky_df = None, mixture_component_covariances_cholesky_scale_tril=None,mixture_weights_concentration=None, data_var_concentration=1., data_var_rate=1.):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components, dtype='float64')
    if mixture_component_covariances_cholesky_df is None:
        mixture_component_covariances_cholesky_df = n_sources + 2.
    if mixture_component_covariances_cholesky_scale_tril is None:
        mixture_component_covariances_cholesky_scale_tril = (1./(mixture_component_covariances_cholesky_df - n_sources - 1.))*tf.eye(n_sources)
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, name='mixture_weights')
    mixture_component_means = ed.Normal(loc=mixture_component_means_mean, scale=tf.sqrt(mixture_component_means_var), sample_shape=(n_components, n_sources), name='mixture_component_means')
    mixture_component_covariances_cholesky = ed.Wishart(
        df=mixture_component_covariances_cholesky_df, scale_tril=mixture_component_covariances_cholesky_scale_tril, sample_shape=(n_components), input_output_cholesky=True, name='mixture_component_covariances_cholesky')
    data_var = ed.Gamma(concentration=data_var_concentration, rate=data_var_rate, sample_shape=(1,n_features), name='data_var')
        
    data = ed.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mixture_weights),
        components_distribution=tfd.MultivariateNormalDiagPlusLowRank(loc=mixture_component_means, 
        scale_perturb_diag=tf.ones((n_components, n_sources)), 
        scale_perturb_factor=tf.einsum('sf,st', factor_loadings, mixture_component_covariances_cholesky), 
        scale_diag=tf.tile(data_var, [n_components, 1]), name='data_component'), sample_shape=(n_observations,), name='data')  
    return data

def mixtureOfFactorAnalyzers(n_observations = 1000, n_components = 2, n_sources = 2, n_features = 2, mixture_component_means_mean = 0., mixture_component_means_var = 1., mixture_component_covariances_cholesky_df = None, mixture_component_covariances_cholesky_scale_tril=None, mixture_weights_concentration=None, data_var_concentration=1., data_var_rate=1.):
    if mixture_weights_concentration is None:
        mixture_weights_concentration = np.ones(n_components, dtype='float64')
    if mixture_component_covariances_cholesky_df is None:
        mixture_component_covariances_cholesky_df = n_sources + 2.
    if mixture_component_covariances_cholesky_scale_tril is None:
        mixture_component_covariances_cholesky_scale_tril = (1./(mixture_component_covariances_cholesky_df - n_sources - 1.))*tf.eye(n_sources)
    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_components, n_sources, n_features), name='factor_loadings')
    mixture_weights = ed.Dirichlet(concentration=mixture_weights_concentration, name='mixture_weights')
    data_var = ed.Gamma(concentration=data_var_concentration, rate=data_var_rate, sample_shape=(1,n_features), name='data_var')
        
    data = ed.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=mixture_weights),
        components_distribution=tfd.MultivariateNormalDiagPlusLowRank(loc=tf.zeros((n_components, n_features)), 
        scale_perturb_diag=tf.ones((n_components, n_sources)), 
        scale_perturb_factor=factor_loadings, 
        scale_diag=tf.tile(data_var, [n_components, 1]), name='data_component'), sample_shape=(n_observations,), name='data')  
    return data

#####
# Test functions
#####


def centeredIndependentFactorAnalysisTest(n_observations, mc_samples, factor_loadings, mixture_weights, mixture_component_var, data_var):
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_var), scale=tf.sqrt(mixture_component_var), name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(mc_samples, n_observations),name='sources')
    data_mean = tf.einsum('bik,kj->ijb', sources, factor_loadings/tf.linalg.norm(factor_loadings, axis=1, keepdims=True), name='data_mean')
    data = ed.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=tf.ones(mc_samples)/mc_samples),
                components_distribution=tfd.Normal(loc=data_mean, scale=tf.sqrt(data_var[:,:,None]), name='data'), name='mc_approx')
    return data, sources, data_mean

def centeredIndependentFactorAnalysisTest2(n_observations, factor_loadings, mixture_weights, mixture_component_var, data_var):
    sources = ed.Independent(
        tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mixture_weights),
            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_var), scale=tf.sqrt(mixture_component_var), name='mixture_component')),
        reinterpreted_batch_ndims=1,sample_shape=(n_observations, ),name='sources')
    #data_mean = tf.einsum('ik,kj->ij', sources, factor_loadings, name='data_mean')
    data_mean = tf.matmul(sources, factor_loadings/tf.linalg.norm(factor_loadings, axis=1, keepdims=True), name='data_mean')
    #data_mean = tf.einsum('ik,kj->ij', sources, factor_loadings/tf.linalg.norm(factor_loadings, axis=1), name='data_mean')
    data = ed.Normal(loc=data_mean, scale=tf.sqrt(data_var), name='data')  
    return data, sources, data_mean


def mixtureOfGaussiansTest(n_observations, mixture_weights, mixture_component_means, mixture_component_covariances_cholesky):
    return ed.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mixture_weights),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mixture_component_means,
                    scale_tril=mixture_component_covariances_cholesky,
                    name='component'), sample_shape=(n_observations,), name='data')

#    data = ed.MixtureSameFamily(
#        mixture_distribution=tfd.Categorical(probs=all_mixture_weights),
#        components_distribution=tfd.MultivariateNormalDiagPlusLowRank(loc=tf.zeros((n_combinations, n_features)), 
#        scale_perturb_diag=jitter + all_mixture_vars, 
#        scale_perturb_factor=tf.tile(tf.transpose(factor_loadings)[None,:,:], [n_combinations, 1, 1]) , 
#        scale_diag=jitter + tf.tile(data_var, [n_combinations, 1]), name='data_component'), sample_shape=(n_observations,), name='data')  
#    return data

# hyperparameters as dict
#def centeredIndependentFactorAnalysis(n_observations = 1000, n_components_in_mixture = 2, n_sources = 2, n_features = 2, hyperparameters = {'mixture_component_var_rate':1.,'mixture_weights_concentration':1.,'data_var_concentration':1.,'data_var_rate':1.}):
#    # check that Dirichlet concentration is a vector
#    if hasattr(tmp,'shape') and hyperparameters['mixture_weights_concentration'].shape == (n_components_in_mixture,):
#        pass
#    else:
#        hyperparameters['mixture_weights_concentration'] = np.ones(n_components_in_mixture, dtype='float64')
#    mixture_component_var = ed.Gamma(concentration=hyperparameters['mixture_component_var_concentration'], rate=hyperparameters['mixture_component_var_rate'], sample_shape=(n_sources,n_components_in_mixture), name='mixture_component_var')
#    mixture_weights = ed.Dirichlet(concentration=hyperparameters['mixture_weights_concentration'], sample_shape=(n_sources,), name='mixture_weights')
#    sources = ed.Independent(
#        tfd.MixtureSameFamily(
#            mixture_distribution=tfd.Categorical(probs=mixture_weights),
#            components_distribution=tfd.Normal(loc=tf.zeros_like(mixture_component_var), scale=mixture_component_var, name='mixture_component')),
#        reinterpreted_batch_ndims=1,sample_shape=(n_observations,),name='sources')
#    factor_loadings = ed.Normal(loc=0., scale=1., sample_shape=(n_sources, n_features), name='factor_loadings')
#    data_mean = tf.matmul(sources, factor_loadings, name='data_mean')
#    data_var = ed.Gamma(concentration=hyperparameters['data_var_concentration'], rate=hyperparameters['data_var_rate'], sample_shape=(1,n_features), name='data_var')
#    data = ed.Normal(loc=data_mean, scale=data_var, name='data')  
#    return data
