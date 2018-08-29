import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from mapper import Mapper

from tfpmodels import centeredIndependentFactorAnalysis, mixtureOfGaussians

import pandas as pd
import xarray as xr

N = 100
Ntest = 0

n_features = 4
n_restarts = 10
n_datasets = 10

deviations = np.logspace(-3,1,5, dtype='float32')

cifa_2_2 = Mapper(centeredIndependentFactorAnalysis, 'cifa', observed_variable_names=['data'], n_observations=N, n_components_in_mixture = 2, n_sources=2, n_features=n_features)
mog_4 = Mapper(mixtureOfGaussians, 'mog', observed_variable_names=['data'], n_observations=N, n_components=4, n_features=n_features)

models = [cifa_2_2, mog_4]
model_names = [model.model_name for model in models]

results = xr.DataArray(np.zeros((len(models), len(deviations), n_restarts, n_datasets)),dims=['model', 'deviation', 'restart', 'dataset'], coords={'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)})

placeholder_deviation = tf.placeholder(dtype='float32')
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
                for model in models: 
                    opt[model.model_name].minimize()
                    results.loc[{'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(loss[model.model_name])