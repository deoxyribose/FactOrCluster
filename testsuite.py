import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from map import Mapper

from tfpmodels import centeredIndependentFactorAnalysis, mixtureOfGaussians

N = 10000
Ntest = 1000

n_features = 10

cifa_2_3 = Mapper(centeredIndependentFactorAnalysis, 'cifa2', observed_variable_names=['data'], n_observations=N, n_components_in_mixture = 2, n_sources=3, n_features=n_features)
mog_3 = Mapper(mixtureOfGaussians, 'mog', observed_variable_names=['data'], n_observations=N, n_components=3, n_features=n_features)

models = [cifa_2_3, mog_3]

with tf.Session() as sess:
    with tape() as reference_tf:
        data_tf = mixtureOfGaussians(n_observations=N, n_components=3, n_features=n_features)
    data, reference = sess.run([data_tf, reference_tf])

    opt = {}
    for model in models: 
        _, opt[model.model_name] = model.map_optimizer(data=data)
    sess.run(tf.global_variables_initializer())
    for model in models: 
        opt[model.model_name].minimize()
