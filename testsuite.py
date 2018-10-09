import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from mapper import Mapper, IFA_MAPEM

from tfpmodels import centeredMarginalizedIndependentFactorAnalysis, projectedMixtureOfGaussians

import pandas as pd
import xarray as xr

from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans

import pickle

store = []

def callback(parameters):
    store.append(parameters)

if __name__ == '__main__':

    N = 1000
    Ntest = 1000

    initial_direction = 'ica'

    n_features = 5
    n_clusters = 4
    n_sources = 2
    n_components_in_mixture = 2
    n_restarts = 3
    n_datasets = 2
    max_attempts = 3

    #deviations = np.logspace(-3,1,5, dtype='float32')
    deviations = np.logspace(-1,1,20, dtype='float64') # larger deviation means better snr in both models

    data_generating_models = ['mog','cifa']
    model_names = ['mog','cifa']
    models = []
    test_models = []
    assign_defaults = []
    data_train = tf.placeholder(shape=(N,n_features), dtype='float64') 
    data_test = tf.placeholder(shape=(Ntest,n_features), dtype='float64')
    cluster_centers = tf.placeholder(shape=(n_clusters,n_sources), dtype='float64')
    #cluster_centers = tf.placeholder(shape=(n_clusters,n_features), dtype='float64')
    ica_directions = tf.placeholder(shape=(2,n_features), dtype='float64')
    data_variance = tf.placeholder(shape=(), dtype='float64')
    
    for name in model_names:
        if name == 'mog':
            model = Mapper(projectedMixtureOfGaussians, 'mog', 
            #model = Mapper(mixtureOfGaussians, 'mog', 
            observed_variable_names=['data'], 
            n_observations=N, 
            n_components=n_clusters, 
            n_sources=n_sources,
            n_features=n_features)
            assign_default = model.assigner(mixture_component_covariances_cholesky=10*tf.tile(tf.eye(n_sources, dtype='float64')[None],[n_clusters,1,1]),
            #assign_default = model.assigner(mixture_component_covariances_cholesky=10*tf.tile(tf.eye(n_features, dtype='float64')[None],[n_clusters,1,1]),
                mixture_component_means=cluster_centers)
        if name == 'cifa':
            model = IFA_MAPEM(centeredMarginalizedIndependentFactorAnalysis, 'cifa', 
                observed_variable_names=['data'], 
                n_observations=N, 
                n_components_in_mixture = n_components_in_mixture, 
                n_sources=n_sources, 
                n_features=n_features, 
                mixture_component_var_concentration=3., 
                mixture_component_var_rate=1.,
                data_var_concentration=3.,
                data_var_rate=1.)
            #tril = np.tril(np.ones((n_components_in_mixture, n_components_in_mixture)))
            #scale = tf.linalg.LinearOperatorLowerTriangular(tril)
            #affine = tfp.bijectors.AffineLinearOperator(np.zeros(n_components_in_mixture), scale)
            #model.append_bijector('mixture_component_var', affine, prepend=False)
            assign_default = model.assigner(data_var=0.1*tf.ones((), dtype='float64'), 
               factor_loadings=ica_directions)
        models.append(model)    
        assign_defaults.append(assign_default)        
    
    train_neg_log_lik_op = []
    test_neg_log_lik_op = []
    ppc_op = []
    loss = {}
    opt = {}
    cg_opt = {}
    optstep = {}
    adam_opt = {}
    reset = None
    emsteps = None
    for model in models:
        train_neg_log_lik_op.append(model.neg_log_lik(data_train))
        test_neg_log_lik_op.append(model.neg_log_lik(data_test))
        ppc_op.append(model.test_model())
        loss[model.model_name], opt[model.model_name] = model.l_bfgs_optimizer(data=data_train)
        _, cg_opt[model.model_name] = model.cg_optimizer(data=data_train)
        _, adam_opt[model.model_name] = model.adam_optimizer(data=data_train)
        optstep[model.model_name] = tf.contrib.opt.ScipyOptimizerInterface(loss[model.model_name], var_list=list(model.unconstrained_variables.values()), options={'maxiter': 1})
        if model.model_name == 'cifa':
            _, reset, emsteps = model.MAPEM_optimizer(data=data_train)
            emproposal = model.MAPEMsteps(data=data_train)
    
    experimental_variable_prealloc = np.zeros((len(data_generating_models), len(models), len(deviations), n_restarts, n_datasets))
    experimental_variable_dims = ['data_generating_model','model', 'deviation', 'restart', 'dataset']
    experimental_variable_coords ={'data_generating_model': data_generating_models, 'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)}
                                              
    train_neg_log_joint = xr.DataArray(experimental_variable_prealloc,dims=experimental_variable_dims,coords=experimental_variable_coords)
    train_neg_log_lik = train_neg_log_joint.copy()
    test_neg_log_lik = train_neg_log_joint.copy()
    MAP_parameters = {}
    ppc = {}
    data_store = {}

    placeholder_deviation = tf.placeholder(dtype='float64')

    fica = FastICA(n_components=n_sources)
    pca = PCA(n_components=n_sources)
    kmeans = KMeans(n_clusters=n_clusters)

    sess = tf.Session()

    all_init = tf.global_variables_initializer()
    
    data_tf = {}
    max_attempt_registered = 1
    
    for data_generating_model in data_generating_models:
        if data_generating_model == 'mog':
            with tape() as reference_tf:
                data_tf[data_generating_model] = projectedMixtureOfGaussians(n_observations=N + Ntest, n_components=n_clusters, n_features=n_features, mixture_component_means_var=placeholder_deviation)
        else:
            with tape() as reference_tf:
                data_tf[data_generating_model] = centeredMarginalizedIndependentFactorAnalysis(n_observations=N + Ntest, n_components_in_mixture = n_components_in_mixture, n_sources=n_clusters, n_features=n_features, mixture_component_var_concentration=3., mixture_component_var_rate=1.,data_var_concentration=3.,data_var_rate=2.*placeholder_deviation)
    for data_generating_model in data_generating_models:    
        for deviation in deviations:
            for dataset in range(n_datasets):
                data, reference = sess.run([data_tf[data_generating_model], reference_tf], feed_dict={placeholder_deviation: deviation})
                
                pca_transformed_data = pca.fit_transform(data[:N])
                kmeans_cluster_centers = kmeans.fit(pca_transformed_data).cluster_centers_
                if initial_direction.lower() == 'ica':
                    init_directions = fica.fit(data).mixing_.T
                elif initial_direction.lower() == 'pca':
                    init_directions = pca.fit(data).components_.T
                else:
                    init_directions = np.random.randn(n_sources, n_features).astype('float64')
                init_directions = init_directions/np.linalg.norm(init_directions,axis=1, keepdims=True)
                current_data_variance = data[:N].var()

                for restart in range(n_restarts):        
                    sess.run(all_init)
                    sess.run(assign_defaults, feed_dict={cluster_centers: kmeans_cluster_centers, ica_directions: init_directions, data_variance: current_data_variance})
                    for i,model in enumerate(models): 
                        print('g={},m={},x={},d={},r={}'.format(data_generating_model, model.model_name, deviation, dataset, restart))
                        #for step in range(1000):
                        #    sess.run(adam_opt[model.model_name], feed_dict={data_train: data[:N]})
                        attempt = 0
                        while attempt < max_attempts:
                            if model.model_name == 'cifa':
                                for j in range(50):
                                    sess.run(reset)
                                    #emprop = sess.run(emproposal, feed_dict={data_train: data[:N]})
                                    sess.run(emsteps, feed_dict={data_train: data[:N]})
                            try:
                                opt[model.model_name].minimize(feed_dict={data_train: data[:N]}, session=sess, loss_callback=callback, fetches=[model.variables])
                                max_attempt_registered = max(attempt, max_attempt_registered)
                                break
                            except:
                                # optstep takes one bfgs step in hopes of escaping bad initialization
                                print("Previous optimization attempt failed, taking one BFGS step and trying again with L-BFGS")
                                optstep[model.model_name].minimize(feed_dict={data_train: data[:N]}, session=sess, loss_callback=callback, fetches=[model.variables])
                                attempt += 1
                            
                        #    print('retrying')
                        #    cg_opt[model.model_name].minimize(feed_dict={data_train: data[:N]}, session=sess, loss_callback=callback, fetches=[model.variables])
                        MAP_parameter, converged_loss = sess.run([model.variables, loss[model.model_name]], feed_dict={data_train: data[:N]})
                        MAP_parameters[(data_generating_model,model.model_name, deviation, restart, dataset)] = MAP_parameter
                        ppc[(data_generating_model,model.model_name, deviation, restart, dataset)] = sess.run(ppc_op[i])
                        data_store[(data_generating_model,model.model_name, deviation, restart, dataset)] = data[:N]
                        train_neg_log_joint.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = converged_loss
                        train_neg_log_lik.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(train_neg_log_lik_op[i], feed_dict={data_train: data[:N]})
                        test_neg_log_lik.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(test_neg_log_lik_op[i], feed_dict={data_test: data[N:]})
                        
    pickle.dump([MAP_parameters,train_neg_log_joint,train_neg_log_lik,test_neg_log_lik],open( "mog_ifa_MAPparameters_and_losses_on_synth_data.p", "wb" ) )
    pickle.dump([ppc, data_store],open( "mog_ifa_MAP_ppc.p", "wb" ) )