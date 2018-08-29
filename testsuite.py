import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from mapper import Mapper

from tfpmodels import centeredIndependentFactorAnalysis, mixtureOfGaussians
from tfpmodels import independentFactorAnalysisTest, mixtureOfGaussiansTest

import pandas as pd
import xarray as xr

N = 100
Ntest = 100

n_features = 4
n_restarts = 10
n_datasets = 10

deviations = np.logspace(-3,1,5, dtype='float32')

cifa_2_2 = Mapper(centeredIndependentFactorAnalysis, 'cifa', observed_variable_names=['data'], n_observations=N, n_components_in_mixture = 2, n_sources=2, n_features=n_features)
mog_4 = Mapper(mixtureOfGaussians, 'mog', observed_variable_names=['data'], n_observations=N, n_components=4, n_features=n_features)

models = [cifa_2_2, mog_4]
model_names = [model.model_name for model in models]
test_models = [independentFactorAnalysisTest,mixtureOfGaussiansTest]

train_neg_log_joint = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})
train_neg_log_lik = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})
test_neg_log_lik = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})
MAP_parameters = {}

placeholder_deviation = tf.placeholder(dtype='float32')

def neg_log_lik(MAP_parameter,model,sess,data):
    # MAP_parameter for ifa includes sources, but ifa_test doesn't take it as input
    try:
        MAP_parameter.pop('sources')
    except:
        pass
    N = data.shape[0]
    # model here is observation model
    # for MoGs, z is already collapsed, so we're evaluating int(p(x_new|theta,z)p(z),dz)
    # for ifa, z is sampled once, so we're evaluating int(p(x_new|theta,z_new)q(z),dz) where q(z) is a pointmass.
    model_MAP = model(n_observations = N, **MAP_parameter)
    return sess.run(-tf.reduce_mean(model_MAP.distribution.log_prob(data)))

with tape() as reference_tf:
    data_tf = mixtureOfGaussians(n_observations=N + Ntest, n_components=3, n_features=n_features, mixture_component_means_mean=placeholder_deviation)
        
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
                for i,model in enumerate(models): 
                    opt[model.model_name].minimize()
                    MAP_parameter, converged_loss = sess.run([model.variables, loss[model.model_name]])
                    MAP_parameters[(model.model_name, deviation, restart, dataset)] = MAP_parameter
                    train_neg_log_joint.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = converged_loss
                    train_neg_log_lik.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = neg_log_lik(MAP_parameter,test_models[i],sess,data[:N])
                    test_neg_log_lik.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = neg_log_lik(MAP_parameter,test_models[i],sess,data[N:])