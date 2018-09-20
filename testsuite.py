import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from mapper import Mapper

from tfpmodels import centeredMarginalizedIndependentFactorAnalysis, mixtureOfGaussians
from tfpmodels import centeredMarginalizedIndependentFactorAnalysisTest, mixtureOfGaussiansTest

import pandas as pd
import xarray as xr

from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans

import pickle

def MAP_model(MAP_parameter,model,N):
    # MAP_parameter for ifa includes sources, but ifa_test doesn't take it as input
    MAP_parameter = dict(MAP_parameter)
    # model here is observation model
    # for MoGs, z is already collapsed, so we're evaluating int(p(x_new|theta,z)p(z),dz)
    # for ifa, z is sampled mc_samples times, so we're evaluating int(p(x_new|theta,z_new)q(z),dz) where q(z) is mc_samples pointmasses.
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

    n_features = 2
    n_clusters = 2
    n_sources = 2
    n_restarts = 7
    n_datasets = 10

    #deviations = np.logspace(-3,1,5, dtype='float32')
    deviations = np.logspace(-1,1,10, dtype='float32') # larger deviation means better snr in both models

    models = []
    models.append(Mapper(mixtureOfGaussians, 'mog', observed_variable_names=['data'], n_observations=N, n_components=n_clusters, n_features=n_features))        
    models.append(Mapper(centeredMarginalizedIndependentFactorAnalysis, 'cifa', observed_variable_names=['data'], n_observations=N, n_components_in_mixture = n_clusters, n_sources=n_sources, n_features=n_features, mixture_component_var_concentration=.1, mixture_component_var_rate=1.,data_var_concentration=.1,data_var_rate=10.))
    model_names = [model.model_name for model in models]
    data_generating_models = ['mog','cifa']
    
    test_models = [mixtureOfGaussiansTest,centeredMarginalizedIndependentFactorAnalysisTest]
    train_neg_log_lik_op = []
    test_neg_log_lik_op = []
    ppc_op = []
    loss = {}
    opt = {}

    data_train = tf.placeholder(shape=(N,n_features), dtype='float32') 
    data_test = tf.placeholder(shape=(Ntest,n_features), dtype='float32')
    for model, test_model in zip(models, test_models):
        train_neg_log_lik_op.append(neg_log_lik(model.variables,test_model,data_train))
        test_neg_log_lik_op.append(neg_log_lik(model.variables,test_model,data_test))
        ppc_op.append(MAP_model(model.variables,test_model,N))
        loss[model.model_name], opt[model.model_name] = model.bfgs_optimizer(data=data_train)
    cluster_centers = tf.placeholder(shape=(n_clusters,n_features), dtype='float32')
    ica_directions = tf.placeholder(shape=(2,n_features), dtype='float32')
    data_variance = tf.placeholder(shape=(), dtype='float32')
    assign_defaults = [None,None]
    assign_defaults[0] = models[0].assigner(mixture_component_covariances_cholesky=10*tf.tile(tf.eye(n_features)[None],[n_clusters,1,1]),mixture_component_means=cluster_centers)
    assign_defaults[1] = models[1].assigner(data_var=1e-1*data_variance*tf.ones((n_features,)), factor_loadings=ica_directions)

    
    experimental_variable_prealloc = np.zeros((len(data_generating_models), len(models), len(deviations), n_restarts, n_datasets))
    experimental_variable_dims = ['data_generating_model','model', 'deviation', 'restart', 'dataset']
    experimental_variable_coords ={'data_generating_model': data_generating_models, 'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)}
                                              
    train_neg_log_joint = xr.DataArray(experimental_variable_prealloc,dims=experimental_variable_dims,coords=experimental_variable_coords)
    train_neg_log_lik = train_neg_log_joint.copy()
    test_neg_log_lik = train_neg_log_joint.copy()
    MAP_parameters = {}
    ppc = {}
    data_store = {}

    placeholder_deviation = tf.placeholder(dtype='float32')

    fica = FastICA(n_components=n_sources)
    kmeans = KMeans(n_clusters=n_clusters)

    sess = tf.Session()

    all_init = tf.global_variables_initializer()
    for data_generating_model in data_generating_models:
        if data_generating_model == 'mog':
            with tape() as reference_tf:
                data_tf = mixtureOfGaussians(n_observations=N + Ntest, n_components=n_clusters, n_features=n_features, mixture_component_means_var=placeholder_deviation)
        else:
            with tape() as reference_tf:
                data_tf = centeredMarginalizedIndependentFactorAnalysis(n_observations=N + Ntest, n_components_in_mixture = n_clusters, n_sources=n_clusters, n_features=n_features, mixture_component_var_concentration=.1, mixture_component_var_rate=1.,data_var_concentration=.1,data_var_rate=1./placeholder_deviation)
        
        
        for deviation in deviations:
            for dataset in range(n_datasets):
                data, reference = sess.run([data_tf, reference_tf], feed_dict={placeholder_deviation: deviation})
                
                kmeans_cluster_centers = kmeans.fit(data[:N]).cluster_centers_
                fica_directions = fica.fit(data).mixing_.T
                fica_directions = fica_directions/np.linalg.norm(fica_directions,axis=1, keepdims=True)
                current_data_variance = data[:N].var()

                for restart in range(n_restarts):        
                    sess.run(all_init)
                    sess.run(assign_defaults, feed_dict={cluster_centers: kmeans_cluster_centers, ica_directions: fica_directions, data_variance: current_data_variance})
                    for i,model in enumerate(models): 
                        print('g={},x={},d={},r={},i={}'.format(data_generating_model, deviation, dataset, restart, i))
                        opt[model.model_name].minimize(feed_dict={data_train: data[:N]}, session=sess)
                        MAP_parameter, converged_loss = sess.run([model.variables, loss[model.model_name]], feed_dict={data_train: data[:N]})
                        MAP_parameters[(data_generating_model,model.model_name, deviation, restart, dataset)] = MAP_parameter
                        ppc[(data_generating_model,model.model_name, deviation, restart, dataset)] = sess.run(ppc_op[i])
                        data_store[(data_generating_model,model.model_name, deviation, restart, dataset)] = data[:N]
                        train_neg_log_joint.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = converged_loss
                        train_neg_log_lik.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(train_neg_log_lik_op[i], feed_dict={data_train: data[:N]})
                        test_neg_log_lik.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(test_neg_log_lik_op[i], feed_dict={data_test: data[N:]})
                        
    pickle.dump([MAP_parameters,train_neg_log_joint,train_neg_log_lik,test_neg_log_lik],open( "mog_ifa_MAPparamaters_and_losses_on_synth_data.p", "wb" ) )
    pickle.dump([ppc, data_store],open( "mog_ifa_MAP_ppc.p", "wb" ) )