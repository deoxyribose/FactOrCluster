import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np

from future_features import tape
from mapper import Mapper, IFA_MAPEM, PMOG_MAPEM

from tfpmodels import centeredMarginalizedIndependentFactorAnalysis, projectedMixtureOfGaussians_Conjugate

import pandas as pd
import xarray as xr
import tqdm

from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans

import pickle

emstore = []
store = []
jitter = 1e-6

def callback(parameters):
    store.append(parameters)

def invgamma_from_moments(mean, variance):
    mean = tf.convert_to_tensor(mean, dtype='float64')
    variance = tf.convert_to_tensor(variance, dtype='float64')
    concentration = 2. + tf.square(mean)/variance
    rate = mean * (1. + tf.square(mean)/variance)
    return concentration, rate

def markovbound_variance(mean, boundary, containment):
    mean = tf.convert_to_tensor(mean, dtype='float64')
    boundary = tf.convert_to_tensor(boundary, dtype='float64')
    containment = tf.convert_to_tensor(containment, dtype='float64')
    return (1. - containment) * tf.square(mean - boundary)

if __name__ == '__main__':

    N = 141
    Ntest = 20#141
    label = '_emulator'

    initial_direction = 'ica'
    new_calibration = True
    n_features = 33
    n_clusters = 5
    n_sources = 2
    n_components_in_mixture = 2
    n_restarts = 3
    n_datasets = 30
    max_attempts = 3
    em_steps = 100

    cluster_mean_fraction = 0.95 # relative total variance of mean component in MoG
    first_component_fraction = 1e-2 # variance scaling in [0,1] of first sub-component in cIFA 
    #first_component_fraction = tf.placeholder(shape=(), dtype='float64', name='first_component_fraction') # variance scaling in [0,1] of first sub-component in cIFA 
    first_boundary = 1. #tf.placeholder(shape=(), dtype='float64', name='first_boundary')
    other_boundary = 1. #tf.placeholder(shape=(), dtype='float64', name='other_boundary')
    
    
    #deviations = np.logspace(-3,1,5, dtype='float32')
    deviations = np.logspace(-2,0,21, dtype='float64')[:-1] # larger deviation means worse snr in both models

    data_generating_models = ['mog','cifa']
    model_names = ['mog','cifa']
    models = []
    test_models = []
    assign_defaults = []
    data_train = tf.placeholder(shape=(N,n_features), dtype='float64') 
    data_test = tf.placeholder(shape=(Ntest,n_features), dtype='float64')
    cluster_centers = tf.placeholder(shape=(n_clusters,n_sources), dtype='float64')
    #cluster_centers = tf.placeholder(shape=(n_clusters,n_features), dtype='float64')
    ica_directions = tf.placeholder(shape=(n_sources,n_features), dtype='float64')
    data_variance = tf.placeholder(shape=(), dtype='float64')
    
    placeholder_deviation = tf.placeholder(shape=(), dtype='float64', name='deviation')

    for name in model_names:
        if name == 'mog':
            data_var_mean = placeholder_deviation
            data_var_boundary = 1.
            data_var_contained = 0.95
            data_var_concentration, data_var_rate = invgamma_from_moments(data_var_mean, 
                markovbound_variance(data_var_mean, data_var_boundary, data_var_contained))
            
            mixture_component_means_var = cluster_mean_fraction * (1. - data_var_mean) / n_sources
            marginal_variance_mean = (1. - cluster_mean_fraction) * (1. - data_var_mean) / n_sources
            marginal_variance_boundary = 1.
            marginal_variance_contained = 0.5
            marginal_variance_concentration, marginal_variance_rate = invgamma_from_moments(marginal_variance_mean, 
                markovbound_variance(marginal_variance_mean, marginal_variance_boundary, marginal_variance_contained))
            
            mixture_component_precisions_cholesky_df = n_sources - 1. + 2.*marginal_variance_concentration
            mixture_component_covariance_cholesky_scale_coeff = 2.*marginal_variance_rate
            mixture_component_precisions_scale_tril = 1./tf.sqrt(mixture_component_covariance_cholesky_scale_coeff) * tf.eye(n_sources, dtype='float64')   

            mog_param = {"mixture_component_means_mean":0., 
                         "mixture_component_means_var":mixture_component_means_var, 
                         "mixture_component_precisions_cholesky_df": mixture_component_precisions_cholesky_df, 
                         "mixture_component_precisions_cholesky_scale_tril":mixture_component_precisions_scale_tril,
                         "mixture_weights_concentration": 2. * tf.ones(n_clusters, dtype='float64'),
                         "data_var_concentration":data_var_concentration,
                         "data_var_rate":data_var_rate}

            model = PMOG_MAPEM(projectedMixtureOfGaussians_Conjugate, 'mog', 
            #model = Mapper(mixtureOfGaussians, 'mog', 
            observed_variable_names=['data'], 
            n_observations=N, 
            n_components=n_clusters, 
            n_sources=n_sources,
            n_features=n_features,
            **mog_param)
            assign_default = model.assigner(data_var=placeholder_deviation,
                mixture_component_precisions_cholesky=0.05*tf.tile(tf.eye(n_sources, dtype='float64')[None],[n_clusters,1,1]),
            #assign_default = model.assigner(mixture_component_covariances_cholesky=10*tf.tile(tf.eye(n_features, dtype='float64')[None],[n_clusters,1,1]),
                mixture_component_means=cluster_centers)
        if name == 'cifa':
            data_var_mean = placeholder_deviation
            data_var_boundary = 1.
            data_var_contained = 0.95
            data_var_concentration, data_var_rate = invgamma_from_moments(data_var_mean, 
                markovbound_variance(data_var_mean, data_var_boundary, data_var_contained))
            
            mixture_component_var_mean_majority = (1. - data_var_mean) / (n_sources * (1. - (1. - first_component_fraction) / n_components_in_mixture) )
            mixture_component_var_mean = tf.concat([first_component_fraction * mixture_component_var_mean_majority * np.ones((n_sources, 1)), mixture_component_var_mean_majority * np.ones((n_sources, n_components_in_mixture - 1))], axis=1)
            
            mixture_component_var_boundary = tf.concat([first_boundary * np.ones((n_sources, 1)), other_boundary * np.ones((n_sources, n_components_in_mixture - 1))], axis=1)
            mixture_component_var_contained = 0.95
            mixture_component_var_concentration, mixture_component_var_rate = invgamma_from_moments(mixture_component_var_mean, 
                markovbound_variance(mixture_component_var_mean, mixture_component_var_boundary, mixture_component_var_contained))

            cifa_param = {"mixture_component_var_concentration": mixture_component_var_concentration,
                          "mixture_component_var_rate": mixture_component_var_rate,
                          "mixture_weights_concentration": 2. * tf.ones(n_components_in_mixture, dtype='float64'),
                          "data_var_concentration": data_var_concentration,
                          "data_var_rate": data_var_rate}

            model = IFA_MAPEM(centeredMarginalizedIndependentFactorAnalysis, 'cifa', 
                observed_variable_names=['data'], 
                n_observations=N, 
                n_components_in_mixture = n_components_in_mixture, 
                n_sources=n_sources, 
                n_features=n_features, 
                **cifa_param)
            
            #tril = np.tril(np.ones((n_components_in_mixture, n_components_in_mixture)))
            #scale = tf.linalg.LinearOperatorLowerTriangular(tril)
            #affine = tfp.bijectors.AffineLinearOperator(np.zeros(n_components_in_mixture), scale)
            #model.append_bijector('mixture_component_var', affine, prepend=False)
            assign_default = model.assigner(data_var=placeholder_deviation*tf.ones((), dtype='float64'), 
               factor_loadings=ica_directions)
        models.append(model)  
        test_models.append(model.test_model())  
        assign_defaults.append(assign_default)        
    
    train_neg_log_lik_op = []
    test_neg_log_lik_op = []
    ppc_op = []
    loss = {}
    opt = {}
    cg_opt = {}
    optstep = {}
    adam_opt = {}
    reset = {}
    emsteps = {}
    emproposals = {}
    for model, test_model in zip(models, test_models):
        train_neg_log_lik_op.append(-tf.reduce_mean(test_model.distribution.log_prob(data_train)))
        test_neg_log_lik_op.append(-tf.reduce_mean(test_model.distribution.log_prob(data_test)))
        ppc_op.append(test_model)
        loss[model.model_name], opt[model.model_name] = model.l_bfgs_optimizer(data=data_train)
        _, cg_opt[model.model_name] = model.cg_optimizer(data=data_train)
        _, adam_opt[model.model_name] = model.adam_optimizer(data=data_train)
        optstep[model.model_name] = tf.contrib.opt.ScipyOptimizerInterface(loss[model.model_name], var_list=list(model.unconstrained_variables.values()), options={'maxiter': 1})
        
        _, reset[model.model_name], emsteps[model.model_name] = model.MAPEM_optimizer(data=data_train)
        emproposals[model.model_name] = model.MAPEMsteps(data=data_train)
    
    experimental_variable_prealloc = np.zeros((len(data_generating_models), len(models), len(deviations), n_restarts, n_datasets))
    experimental_variable_dims = ['data_generating_model','model', 'deviation', 'restart', 'dataset']
    experimental_variable_coords ={'data_generating_model': data_generating_models, 'model': model_names, 'deviation': deviations, 'restart': range(n_restarts), 'dataset': range(n_datasets)}
                                              
    train_neg_log_joint = xr.DataArray(experimental_variable_prealloc,dims=experimental_variable_dims,coords=experimental_variable_coords)
    train_neg_log_lik = train_neg_log_joint.copy()
    test_neg_log_lik = train_neg_log_joint.copy()
    MAP_parameters = {}
    ppc = {}
    data_store = {}

    perturb_from_zero = [model.assigner(mixture_weights=jitter + 
        (1.-model.variable_shapes['mixture_weights'][-1].value*jitter)*model.variables['mixture_weights']) for model in models]

    fica = FastICA(n_components=n_sources)
    pca = PCA(n_components=n_sources)
    kmeans = KMeans(n_clusters=n_clusters)

    sess = tf.Session()

    all_init = tf.global_variables_initializer()
    
    data_tf = {}
    max_attempt_registered = 1
    
    all_combinations = len(data_generating_models) * len(deviations) * n_datasets * n_restarts * len(models)
    progress = tqdm.tqdm(total=all_combinations)

    for data_generating_model in data_generating_models:
        if data_generating_model == 'mog':
            with tape() as reference_tf:
                data_tf[data_generating_model] = projectedMixtureOfGaussians_Conjugate(n_observations=N + Ntest, n_components=n_clusters, n_sources=n_sources, n_features=n_features, **mog_param)
        else:
            with tape() as reference_tf:
                data_tf[data_generating_model] = centeredMarginalizedIndependentFactorAnalysis(n_observations=N + Ntest, n_components_in_mixture = n_components_in_mixture, n_sources=n_sources, n_features=n_features, **cifa_param)
    for data_generating_model in data_generating_models:    
        for deviation in deviations:
            for dataset in range(n_datasets):
                data, reference = sess.run([data_tf[data_generating_model], reference_tf], feed_dict={placeholder_deviation: deviation})
                
                if initial_direction.lower() == 'ica':
                    init_directions = fica.fit(data).mixing_.T
                    kmeans_cluster_centers = kmeans.fit(fica.transform(data[:N])).cluster_centers_
                
                elif initial_direction.lower() == 'pca':
                    init_directions = pca.fit(data).components_.T
                    kmeans_cluster_centers = kmeans.fit(pca.transform(data[:N])).cluster_centers_
                else:
                    init_directions = np.random.randn(n_sources, n_features).astype('float64')
                    kmeans_cluster_centers = kmeans.fit(data[:N].dot(init_directions.T)).cluster_centers_
                
                init_directions = init_directions/np.linalg.norm(init_directions,axis=1, keepdims=True)
                current_data_variance = data[:N].var()

                for restart in range(n_restarts):        
                    sess.run(all_init)
                    sess.run(assign_defaults, feed_dict={placeholder_deviation: deviation, 
                                                         cluster_centers: kmeans_cluster_centers, 
                                                         ica_directions: init_directions, 
                                                         data_variance: current_data_variance})
                    for i,model in enumerate(models): 
                        #print('g={},m={},x={},d={},r={}'.format(data_generating_model, model.model_name, deviation, dataset, restart))
                        #for step in range(1000):
                        #    sess.run(adam_opt[model.model_name], feed_dict={data_train: data[:N]})
                        attempt = 0
                        #progress = []
                        while attempt < max_attempts:
                            for j in range(em_steps):
                                sess.run(reset[model.model_name], feed_dict={placeholder_deviation: deviation})
                                emstore.append(sess.run(emproposals[model.model_name], feed_dict={data_train: data[:N], placeholder_deviation: deviation}))
                                sess.run(emsteps[model.model_name], feed_dict={data_train: data[:N], placeholder_deviation: deviation})
                            sess.run(perturb_from_zero)
                                #progress.append(sess.run(loss[model.model_name], feed_dict={placeholder_deviation: deviation}))
                            try:
                                opt[model.model_name].minimize(feed_dict={data_train: data[:N], placeholder_deviation: deviation}, session=sess, loss_callback=callback, fetches=[model.variables])
                                max_attempt_registered = max(attempt, max_attempt_registered)
                                break
                            except Exception as e:
                                print("Previous optimization attempt failed with message `{}`, taking one EM step to recalibrate.".format(str(e)))
                                #sess.run(reset[model.model_name])
                                #sess.run(emsteps[model.model_name], feed_dict={data_train: data[:N], placeholder_deviation: deviation})
                            
                                # optstep takes one bfgs step in hopes of escaping bad initialization
                                #print("Previous optimization attempt failed, taking one BFGS step and trying again with L-BFGS")
                                #optstep[model.model_name].minimize(feed_dict={data_train: data[:N]}, session=sess, loss_callback=callback, fetches=[model.variables])
                                attempt += 1
                        if attempt == max_attempts:
                            raise(RuntimeError("optimizer failed to run"))    
                        #    print('retrying')
                        #    cg_opt[model.model_name].minimize(feed_dict={data_train: data[:N]}, session=sess, loss_callback=callback, fetches=[model.variables])
                        MAP_parameter, converged_loss = sess.run([model.variables, loss[model.model_name]], feed_dict={data_train: data[:N], placeholder_deviation: deviation})
                        MAP_parameters[(data_generating_model,model.model_name, deviation, restart, dataset)] = MAP_parameter
                        ppc[(data_generating_model,model.model_name, deviation, restart, dataset)] = sess.run(ppc_op[i])
                        data_store[(data_generating_model,model.model_name, deviation, restart, dataset)] = data[:N]
                        train_neg_log_joint.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = converged_loss
                        train_neg_log_lik.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(train_neg_log_lik_op[i], feed_dict={data_train: data[:N], placeholder_deviation: deviation})
                        test_neg_log_lik.loc[{'data_generating_model': data_generating_model, 'model': model.model_name, 'deviation': deviation, 'restart': restart, 'dataset': dataset}] = sess.run(test_neg_log_lik_op[i], feed_dict={data_test: data[N:], placeholder_deviation: deviation})
                        progress.update()
    pickle.dump([MAP_parameters,train_neg_log_joint,train_neg_log_lik,test_neg_log_lik],open( "mog_ifa_MAPparameters_and_losses_on_synth_data{}_N{}.pkl".format(label,N), "wb" ) )
    pickle.dump([ppc, data_store],open( "mog_ifa_MAP_ppc{}_N{}.pkl".format(label,N), "wb" ) )