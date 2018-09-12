import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import numpy as np
import matplotlib.pylab as plt
from tfpmodels import centeredIndependentFactorAnalysisTest

def plot_source_distributions(mixture_component_var,sess):
    n_sources,n_components_in_mixture = mixture_component_var.shape
    batch_event_space = np.stack((n_sources*n_components_in_mixture)*[np.linspace(-6,6,1000).astype(np.float32)],axis=1).reshape(-1,n_sources,n_components_in_mixture)
    densities = sess.run(tfd.Normal(loc=0.,scale=tf.sqrt(mixture_component_var)).prob(batch_event_space))
    for i in range(n_sources):
        plt.show()
        for j in range(n_components_in_mixture):
            plt.plot(batch_event_space[:,i,j],densities[:,i,j])
    plt.show()

def plot_ifa_parameters_and_ppc(estimated_parameters, true_parameters, sess):
    map_estimates = dict(estimated_parameters)
    #map_estimates.pop('sources')

    true_parameters_vars = true_parameters.copy()
    map_sources = map_estimates.pop('sources')
    #map_estimates = sess.run(map_estimates)

    n_observations = true_parameters['data'].shape[0]
    #testmodel,source = centeredIndependentFactorAnalysisTest2(n_observations=n_observations, **map_estimates)
    testmodel,source, data_mean = centeredIndependentFactorAnalysisTest(n_observations=n_observations, mc_samples=1, **map_estimates)
    #print(sess.run(map_estimates['data_var']))
    #print(sess.run(source.distribution.sample((5000))).var(0))
    ppc = sess.run(testmodel.distribution.sample())

    #plot_source_distributions(true_parameters['mixture_component_var'],sess)
    #plot_source_distributions(map_estimates['mixture_component_var'],sess)
    fig, ax = plt.subplots()
    plt.title('Variance of sample is {}'.format(true_parameters['data'].var()))
    plt.scatter(*true_parameters['data'].T,alpha=.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.show()

    #fig, ax = plt.subplots()
    #plt.title('Variance of sample is {}'.format(ppc.var()))
    #plt.scatter(*ppc.T, alpha=.5,c='orange')
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    #plt.show()

    fig, ax = plt.subplots()
    plt.title('Variance of sample is {}'.format(ppc.var()))
    plt.scatter(*true_parameters['data'].T,alpha=.5)
    plt.scatter(*ppc.T, alpha=.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

    fig, ax = plt.subplots()
    plt.title('True and estimated sources')
    plt.scatter(*true_parameters['sources'].T)
    #map_sources = sess.run(tf.squeeze(source.distribution.sample((n_observations))))
    n_sources = map_sources.shape[1]
    plt.scatter(*map_sources.T)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

    fig, ax = plt.subplots()
    plt.title('True and estimated data_mean')
    data_mean_s = sess.run(tf.squeeze(data_mean))
    true_data_mean = np.einsum('ik,kj->ij', true_parameters['sources'], true_parameters['factor_loadings']/np.linalg.norm(true_parameters['factor_loadings'], axis=1, keepdims=True))
    plt.scatter(*true_data_mean.T)
    plt.scatter(*data_mean_s.T)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

    n_components_in_mixture = true_parameters['mixture_component_var'].shape[1]

    for source in range(n_sources):
        plt.title('mixture component variances, true and estimated')
        plt.bar(np.arange(n_components_in_mixture),true_parameters_vars['mixture_component_var'][source,:])
        plt.bar(np.arange(n_components_in_mixture),map_estimates['mixture_component_var'][source,:],alpha=.5)
        plt.show()

    for source in range(n_sources):
        plt.title('mixing weights, true and estimated')
        plt.bar(np.arange(n_components_in_mixture),true_parameters_vars['mixture_weights'][source,:])
        plt.bar(np.arange(n_components_in_mixture),map_estimates['mixture_weights'][source,:],alpha=.5)
        plt.show()

    fgen = true_parameters['factor_loadings']
    fpred = map_estimates['factor_loadings']
    fig, ax = plt.subplots()
    ax.scatter(*true_parameters['data'].T,alpha=0.3)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for fg in fgen:
        plt.plot(fg[0]*np.array([1,-1]),fg[1]*np.array([1,-1]),color='y',label='true')

    for fg in fpred:
        plt.plot(fg[0]*np.array([1,-1]),fg[1]*np.array([1,-1]),color='r', linestyle='-.', label='predicted')

    #for fg in fica_n:
    #    plt.plot(fg[0]*np.array([1,-1]),fg[1]*np.array([1,-1]),color='k',linestyle='--',label='initial')

    plt.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim);