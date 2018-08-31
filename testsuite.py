import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from mapper import Mapper

from tfpmodels import centeredIndependentFactorAnalysis, mixtureOfGaussians
from tfpmodels import centeredIndependentFactorAnalysisTest, mixtureOfGaussiansTest

import pandas as pd
import xarray as xr

from sklearn.decomposition import FastICA


def MAP_model(MAP_parameter,model,N):
    # MAP_parameter for ifa includes sources, but ifa_test doesn't take it as input
    MAP_parameter = dict(MAP_parameter)
    try:
        
        MAP_parameter.pop('sources')
    except:
        pass
    # model here is observation model
    # for MoGs, z is already collapsed, so we're evaluating int(p(x_new|theta,z)p(z),dz)
    # for ifa, z is sampled once, so we're evaluating int(p(x_new|theta,z_new)q(z),dz) where q(z) is a pointmass.
    try:
        model_MAP = model(n_observations = N, mc_samples=1000, **MAP_parameter)
    except TypeError:
        model_MAP = model(n_observations = N, **MAP_parameter)
    return model_MAP

def neg_log_lik(MAP_parameter,model,data):
    N = data.shape[0]
    model_MAP = MAP_model(MAP_parameter,model,N)
    return -tf.reduce_mean(model_MAP.distribution.log_prob(data))

if __name__ == '__main__':

    N = 1000
    Ntest = 1000

    n_features = 4
    n_restarts = 10
    n_datasets = 10

    deviations = np.logspace(-3,1,5, dtype='float32')

    models = []
    models.append(Mapper(centeredIndependentFactorAnalysis, 'cifa', observed_variable_names=['data'], n_observations=N, n_components_in_mixture = 2, n_sources=2, n_features=n_features))
    models.append(Mapper(mixtureOfGaussians, 'mog', observed_variable_names=['data'], n_observations=N, n_components=4, n_features=n_features))        
    model_names = [model.model_name for model in models]
    
    test_models = [centeredIndependentFactorAnalysisTest,mixtureOfGaussiansTest]
    train_neg_log_lik_op = []
    test_neg_log_lik_op = []
    data_train = tf.placeholder(shape=(N,n_features), dtype='float32') 
    data_test = tf.placeholder(shape=(Ntest,n_features), dtype='float32')
    for model, test_model in zip(models, test_models):
        train_neg_log_lik_op.append(neg_log_lik(model.variables,test_model,data_train))
        test_neg_log_lik_op.append(neg_log_lik(model.variables,test_model,data_test))

    ica_directions = tf.placeholder(shape=(2,n_features), dtype='float32')
    assign_defaults = [None,None]
    assign_defaults[0] = models[0].assigner(data_std=1e-3*tf.ones((1,n_features)), factor_loadings=ica_directions)
    assign_defaults[1] = models[1].assigner(mixture_component_covariances_cholesky=10*tf.tile(tf.eye(n_features)[None],[4,1,1]))

    train_neg_log_joint = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})
    train_neg_log_lik = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})
    test_neg_log_lik = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})
    MAP_parameters = {}

    placeholder_deviation = tf.placeholder(dtype='float32')

    fica = FastICA(n_components=2)
                

    with tape() as reference_tf:
        data_tf = mixtureOfGaussians(n_observations=N + Ntest, n_components=4, n_features=n_features, mixture_component_means_std=placeholder_deviation)
            
    with tf.Session() as sess:
        for deviation in deviations:
            for dataset in range(n_datasets):
                data, reference = sess.run([data_tf, reference_tf], feed_dict={placeholder_deviation: deviation})
                
                
                
                loss = {}
                opt = {}
                for model in models: 
                    loss[model.model_name], opt[model.model_name] = model.map_optimizer(data=data[:N])
                    
                for restart in range(n_restarts):        
                    sess.run(tf.global_variables_initializer())
                    sess.run(assign_defaults, feed_dict={ica_directions: fica.fit(data).mixing_.T})
                    for i,model in enumerate(models): 
                        opt[model.model_name].minimize()
                        MAP_parameter, converged_loss = sess.run([model.variables, loss[model.model_name]])
                        MAP_parameters[(model.model_name, deviation, restart, dataset)] = MAP_parameter
                        train_neg_log_joint.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = converged_loss
                        train_neg_log_lik.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(train_neg_log_lik_op[i], feed_dict={data_train: data[N:]}) #neg_log_lik(model.variables,test_models[i],sess,data[:N])
                        test_neg_log_lik.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(test_neg_log_lik_op[i], feed_dict={data_test: data[:N]})#neg_log_lik(model.variables,test_models[i],sess,data[N:])