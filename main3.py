# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:40:01 2019

@author: giess
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:01:21 2019

@author: giess
"""
#import matplotlib
#matplotlib.use('GTKAgg')
#print(matplotlib.get_backend())

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
#import scipy.stats as st
import os
#import warnings
import math
import numexpr as ne
ne.set_num_threads(4)
#from numba import jit


A_lim = [0.1, 1.2, 0.1]
mu_lim = [-30, 30, 1]
sig_lim = [.1, 45, 5]

#stimuli_range = np.arange(-120, 125, 5)
stimuli_range = np.linspace(-120, 120, 49)
response_range = np.linspace(-0.5,1.5,21)
#response_range = np.round(np.arange(-0.5, 1.6, 0.1), 1) # range of response values

# list of paramter values
A = np.arange(A_lim[0], A_lim[1]+A_lim[2], A_lim[2])
mu = np.arange(mu_lim[0], mu_lim[1] + mu_lim[2], mu_lim[2])
sig = np.arange(sig_lim[0], sig_lim[1] + sig_lim[2], sig_lim[2])
#sig = np.setdiff1d(sig,np.array([0])) # remove sigma value of 0 as this breaks the gaussian - divide by zero error

# true model parameters
A_gen = 0.9
mu_gen = 0
sig_gen = 15.1

def gaussian(x, A, mu, sig):
    """
    Gaussian distribution with an amplitude factor, sigma scale factor, and a noise component.
    """
    return (A  * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
#    return (A * np.exp( ( -(x-mu)*(x-mu)) / ( 2 * (sig * sig)))) #* np.sin(abs(x) * .05)

#@jit
def response_model(x, mu , sig = 0.1):
    """
    Model across the response values
    """
    return np.exp((-(x - mu) * (x-mu)) / (2 * (sig * sig)))


def response_model_pdf(x, mu, sig = 0.1):
    """Probability density function for the response model"""
#    return (1 / (2 * np.pi * (sig * sig)) * np.exp( (-(x - mu) * (x-mu)) / (2 * (sig * sig))) ) / sig

    x = (x - mu) / sig
#    pdf =  ( np.exp( - x**2 / 2) / np.sqrt(2 * np.pi) ) / sig
    pdf = ( np.exp( - (x*x) / 2) / math.sqrt(2 * np.pi) ) / sig
    return pdf/np.sum(pdf)

#    pdf = st.norm.pdf(x, mu, sig)
#
#    pdf=pdf/np.sum(pdf)
#    return pdf


def gen_prior(A_ = [], mu_ = [], sig_ = []):
    """
    Input :
        'A' parameter values as list
        'mu' parameter values as list
        'sig' parameter values as list

    Generate the prior distribution of the model parameters

    A : amplitude (height)
    mu : mean (center)
    sig : standard deviation
    """
    prob = 1/( len(A_) * len(mu_) * len(sig_) )

    prior = [(i, j, k, prob) for i in A_
                 for j in mu_
                 for k in sig_]

    prior = np.asarray(prior)

    # Samples the prior probabilities from a gaussian so no longer uniform prior
#    x = len(A_) * len(mu_) * len(sig_)
#    prob_d = gaussian(np.arange(0, x), 100, int(x * 0.8), int(x * 0.8))
#    prob_d_sum = sum(prob_d)
#
#    len_p = len(prior)
#    for p in range(0, len_p):
#        prior[p,3]= prob_d[p] / prob_d_sum

    return prior

def lookuptable_func(prior, stimuli_range, response_range):
    """
    Input:
        prior : np array : prior probability of a model
        stimuli_range : list : range of stimuli values
        response_range : list : range of response values

    Calculates, for each model, the response probabilities given a stimuli

    Output:
        lookuptable : np.array : (models, stimuli, response) shaped array
    """
    lookuptable = np.zeros((len(prior), len(stimuli_range), len(response_range)))

    len_prior = len(prior)

    for model in range(0, len_prior):
        A, mu, sig = prior[model,0], prior[model,1], prior[model,2]

        AI = gaussian(stimuli_range, A, mu, sig) # all responses across stimuli for a given model

#        rp = np.asarray(list(map(lambda x: response_model(response_range, x), AI)))
        rp = np.array([response_model(response_range, x) for x in AI])

        response_probability = (rp/rp.sum(axis=0))

        lookuptable[model] = response_probability

    return lookuptable
#def lookuptable_func(prior, stimuli_range, response_range):
#    lookuptable = np.zeros((len(prior), len(stimuli_range), len(response_range)),dtype=np.float32)
#
##    len_prior = len(prior)
#
##    for model in range(0, len_prior):
#    A, mu, sig = prior[:,0], prior[:,1], prior[:,2]
#
#    stimuli_r = np.tile(stimuli_range, (len(A), 1))
#
#    AI = gaussian(stimuli_r, A, mu, sig) # all responses across stimuli for a given model
#
#    print(AI.shape)
#    import sys
#    sys.exit(0)
#
#    rp = np.array([response_model(response_range, x) for x in AI])
#
#
#    response_probability = (rp/rp.sum(axis=0))
#
#    lookuptable[model] = response_probability
#
#    return lookuptable

def response_given_stimuli(prior, lookuptable):
    """
    Input:
        prior : np.array - holds the models parameters + prior probability
        lookuptable : np.array- holds the response values for a given model and stimuli

    We first multiply each models response values by that models prior probability,
    then we sum across models.

    p(r|x) = sum p(r|m, x)p(m)

    This creates a table similar to the lookuptable but without the model dimension
    """
    m, s, r = lookuptable.shape
#    prob_rx = np.zeros((m,s,r), dtype=np.float16)

#    prob_rx = lookuptable * prior[:, None, None, 3]

    prob_rx = ne.evaluate("l * p", {'p':prior[:, None, None, 3], 'l':lookuptable})

    prob_rx = np.sum(prob_rx, axis = 0)
    return prob_rx

#    x = ne.evaluate("sum(p, axis = 0)", {'p':prob_rx})
#    return x


def calc_posterior(prior, lookuptable, prob_rx):
    """
    Input:
        lookuptable : np.array - holds the response values for a given model and stimuli
        prior : np.array - holds model parameters + model prior probability
        prob_rx : np.array - response values * by model prior, and summed across models

    Here we apply Bayes rule to calculate the posterior of each model
    """

    m, s, r = lookuptable.shape
    posterior = np.zeros((m, s, r), dtype=np.float16)
#
#    return lookuptable * prior[:, None, None, 3] / prob_rx
    posterior = ne.evaluate("l * p / rx", {'p': prior[:,None,None,3], 'l':lookuptable, 'rx': prob_rx})
    return posterior


def sample_gen_model(x_t_1):
    """
    Input:
        x_t_1 : int : single stimuli index - related to the stimuli_range list

    Given a single stimuli we sample our generative model and return a single response.

    Output:
        x_t_1
        response_idx : int : the index of the response value in the response_range list
    """

    x = stimuli_range[x_t_1] # convert index to actual value

    true_r = round(gaussian(x, A_gen, mu_gen, sig_gen), 1) # sample stimuli from generative model

    responses_probability = response_model_pdf(response_range, true_r) # equivalent to the above 2 steps

    observed_response = rnd.choices(response_range, responses_probability, k = 1) # pick a response

    response_idx = np.where( response_range == observed_response )[0][0] # index of the observed value in the response range

    return x_t_1, response_idx

def update_posterior(prior, posterior, x_t_1, response_idx, overwrite_prior = True):
    """
    Input:
        prior : np.array : prior probabilities
        x_t_1 : int : stimuli index
        response_idx : int : response index
        overwrite_true : boolean : whether to replace the prior or create new

    Given a sampled stimuli and its response - sampled from the generative model -
    we update the posterior of each model.

    If overwrite_prior is True then we replace the current prior matrix with new values
    otherwise we return a new matrix which has a different shape (ie. only holds % value no parameter values).

    """

    if overwrite_prior:
        prior[:,3] = posterior[:, x_t_1, response_idx]
        return prior
    else:
        new_posterior = np.zeros(len(posterior))

        for model in range(0, len(posterior)):
            new_posterior[model] = posterior[model, x_t_1, response_idx]

        return new_posterior

def expected_entropy(posterior):
    """
    ------------UNUSED----------
    H(x,r) = -sum posterior * log(posterior)
    """

#    H_xr = np.sum(-(posterior * np.log(posterior)), axis = (0,2))
    Pr = np.mean(posterior, axis = 0)
    Hr = -np.sum(np.multiply(Pr, np.log(Pr)))
    Hrtheta = -np.mean(np.sum(np.multiply(posterior, np.log(posterior))))
    MI = Hr - Hrtheta
    return MI


def calc_rmse(posterior):
    """
    Take the posterior, mean across the models,
    subtract the new window from each model to get the difference, square that difference
    boom u've got your rmse
    """

    mean_ = np.mean(posterior, axis = 0)

#    diffs = np.zeros_like(posterior, dtype=np.float16)
#    np.subtract(posterior, mean_, out=diffs)

    diffs = ne.evaluate("p - m", {'p':posterior, 'm':mean_})

    rmse = np.sum(diffs, axis = 0)
    rmse = np.sum(rmse, axis = 1)
#    rmse = sum(rmse)

    return -rmse

#    x = ne.evaluate("sum(sum(p - m, axis=0), axis=1)", {'p':posterior, 'm':mean_})
#    return x

#def calc_rmse_with_hist(posterior, hist_rmse):
#    """
#    Same idea as rmse but store a r*s matrix which stores weights for each
#    response-stimulus pair. Multiply the calculated rmse matrix by this and then update it.
#    """
#
#    mean_ = np.mean(posterior, axis = 0)
#    diffs = ne.evaluate("p - m", {'p':posterior, 'm':mean_})
#
#    rmse = np.sum(diffs, axis = 0)
#
#    rmse = rmse * hist_rmse
#    hist_rmse = rmse  # update the hist_rmse
#
#    rmse = np.sum(rmse, axis = 0)
#
#    return -rmse, hist_rmse



def calc_infomax(posterior, prior):

    m, s, r = posterior.shape

    u = np.log( np.divide(posterior, prior[:, None,None, 3]) )

    u = np.sum(u, axis = 0)
    u = np.sum(u, axis = 1)

    return u

def calc_expected_model(prior):
    """
    We want to plot the expected model
    We get the parameters of this model by multiplying all parameter combinations by their models prior
    """

    # Estimated Gaussian Function - init at 0,0,0.1
    estimate_function = [0, 0, 0.1]

#    len_prior = len(prior)
#    for model in range(0, len_prior):
#        estimate_function[0] += prior[model,0] * prior[model,3]
#        estimate_function[1] += prior[model,1] * prior[model,3]
#        estimate_function[2] += prior[model,2] * prior[model,3]

    estimate_function[0] = np.dot(prior[:,0], prior[:,3])
    estimate_function[1] = np.dot(prior[:,1], prior[:,3])
    estimate_function[2] = np.dot(prior[:,2], prior[:,3])

    return estimate_function


def est_model_accuracy(est_func_responses, true_model_responses):
    """
    Since we know the true model we can compare it to the estimated model to get an accuracy value
    Effectively its the same RMSE calculation as used previously
    """

    len_r = len(est_func_responses)
    sum_ = 0

    for r in range(0, len_r):
        sum_ += (est_func_responses[r] - true_model_responses[r])**2

    accuracy = (sum_ / len_r) ** (0.5)

    return accuracy

def est_model_accuracy_r2(est_func_responses, true_model_responses):
    """
    Another way of comparing the true and the estimate model is by taking the r-squared error,
    for this we use the sklearn package
    r2 = 1 - (explained variance / total variance)
    """
    return r2_score(true_model_responses, est_func_responses)

def trial(prior, lookuptable, trial_idx, random_sample = False):
    """
    A single trial

    Input:
        prior : np.array: the current prior probability of each model - (also holds the models parameters)
        lookuptable : np.array : the lookuptable - doesn't change across trials
        trial_idx : int : the trial that is currently being run
        random_sample : bool : whether to use random sampling or not

    1. a single trial includes first calculating the probability of a response given a stimuli summed across models
    2. then we calculate the posterior of models given stimuli and responses
    3. then we calculate the rmse across models, marginalizing out models and responses, so that we get a value
    for each stimuli
    4. we select the stimuli with max rmse value
    5. we sample the selected stimuli against the generative model to get a single response
    6. we pick that stimuli, response pair form the posterior(2) for each model and assign that value as the
    new prior for that model
    """
    # Integral across models
    prob_rx = response_given_stimuli(prior, lookuptable)

    # Posterior dist. of models
    posterior = calc_posterior(prior, lookuptable, prob_rx)

    # RMSE calculation
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#       rmse = calc_infomax(posterior, prior)
    rmse = calc_rmse(posterior)
#    rmse, hist_rmse = calc_rmse_with_hist(posterior, hist_rmse)

    # Stimuli selection and model sampling
    if random_sample or trial_idx < 3: # trial_idx < 0 means no random sample at start
        """Random stimuli"""
        x_t_1 = rnd.randint(0, len(stimuli_range)-1)

        _, response_idx = sample_gen_model(x_t_1)
    else:
        # Stimuli to present in a trial
        x_t_1 = np.argmax(rmse)

        # Response of presented stimuli
        _, response_idx = sample_gen_model(x_t_1)


    # Update prior probability of models
    new_prior = update_posterior(prior, posterior, x_t_1, response_idx, True)

    return new_prior, rmse, posterior, x_t_1, response_idx


def run():

    prior = gen_prior(A, mu, sig)
    start = time.time()
    lookuptable = lookuptable_func(prior, stimuli_range, response_range)
    print("lookuptable", time.time() - start)

    true_model = gaussian(stimuli_range, A_gen, mu_gen, sig_gen)

    ntests = 1
    ntrials = 100

    live_plot = 0
    after_plots = 0
    random_sampling = True
    corner_plot_ = False
    save_prior = 0  # save the prior each trial/exp only for the adap. method
    save_accuracy = 0  # save the accuracies
    save_accuracy_rnd = 0
    save_xr = 0  # save the stimuli and response value idx'

    if ntests > 1: live_plot = 0
    if ntests > 1: after_plots = 0
    if save_accuracy_rnd: random_sampling = True

    # Store data from adaptive sampling trials
#    rmse_data = np.zeros((ntests, ntrials, len(stimuli_range)))
    prior_data = np.zeros((ntests, ntrials, len(prior), len(prior[0])), dtype = np.float16)
#    posterior_data = np.zeros((ntests, ntrials, len(lookuptable), len(stimuli_range), len(response_range))) # Too big to store ~ 3.5 Billion floats ~ 13GB once full
    x_samp_data = np.zeros((ntests, ntrials), dtype = int)
    r_samp_data = np.zeros((ntests, ntrials), dtype = int)
    accuracy_data = np.zeros((ntests, ntrials), dtype = np.float16)
    est_model_data = np.zeros((ntests, ntrials, 3), dtype = np.float16)

    # Store data from random sampling trials
#    rnd_rmse_data = np.zeros((ntests, ntrials, len(stimuli_range)))
    rnd_prior_data = np.zeros((ntests, ntrials, len(prior), len(prior[0])), dtype = np.float16)
#    rnd_posterior_data = np.zeros((ntests, ntrials, len(lookuptable), len(stimuli_range), len(response_range)))
    rnd_x_samp_data = np.zeros((ntests, ntrials), dtype = int)
    rnd_r_samp_data = np.zeros((ntests, ntrials), dtype = int)
    rnd_accuracy_data = np.zeros((ntests, ntrials), dtype = np.float16)
    rnd_est_model_data = np.zeros((ntests, ntrials, 3), dtype = np.float16)

    for exp in range(ntests):
        start_exp = time.time()

        if live_plot:
            fig = plt.figure(figsize = (10, 10))

        # Run adaptive sampling trials
        running_prior = np.copy(prior)
#        start_adp = time.time()
        for i in range(ntrials):
#            start = time.time()
            running_prior, running_rmse, _, x_t, response_idx = trial(running_prior, lookuptable, i)

            prior_data[exp,i] = running_prior

            x_samp_data[exp,i] = x_t
            r_samp_data[exp,i] = response_idx

            # Get the estimated Gaussian model
            est_model_param = calc_expected_model(running_prior)
            est_model = gaussian(stimuli_range, est_model_param[0], est_model_param[1], est_model_param[2])
            est_model_data[exp,i] = est_model_param

            # Compare the estimated model to the true model
            accuracy = est_model_accuracy_r2(est_model, true_model)
            accuracy_data[exp,i] = accuracy

            if live_plot:
                plt.clf() # clear current graphs
#                plt.gcf()
                fig.set_size_inches(10,10) # needed when saving figures

                fig.suptitle('Trial: %i' % (i+1))
                plot_rmse(running_rmse)
                plot_prior(running_prior[:,3])
                plot_sampled(true_model, est_model, accuracy, x_samp_data[exp], r_samp_data[exp], i)
#                plt.savefig("./Figures/LivePlot/liveplot_trial_" + str(i) + ".png", dpi = 95)
                plt.pause(0.1) # pause and redraw new graphs

#            print("Adaptive trial:", i, "|", round(time.time() - start, 5))
#        print("Adaptive sampling done:", exp+1, "|", round(time.time() - start_adp, 5))
        if live_plot:
            plt.close(fig)

        # Run random sampling trials
        if random_sampling:
            rnd_prior = np.copy(prior)
#            start = time.time()
            for i in range(ntrials):
                rnd_prior, rnd_rmse, _, rnd_x_t, rnd_response_idx = trial(rnd_prior, lookuptable, i, True)

                rnd_prior_data[exp,i] = rnd_prior

                rnd_x_samp_data[exp,i] = rnd_x_t
                rnd_r_samp_data[exp,i] = rnd_response_idx



                """ Estimated Gaussian model - from random sampled points """
                rnd_est_model_param = calc_expected_model(rnd_prior)
                rnd_est_model = gaussian(stimuli_range, rnd_est_model_param[0], rnd_est_model_param[1], rnd_est_model_param[2])
                rnd_est_model_data[exp,i] = rnd_est_model_param

                """Compare random estimate model to true model"""
                rnd_accuracy = est_model_accuracy_r2(rnd_est_model, true_model)
                rnd_accuracy_data[exp,i] = rnd_accuracy

#            print("Random sampling done:  ",exp+1, "|", round(time.time() - start, 5))

        if after_plots:
            # After simulation data plots
            fig2, axx2 = plt.subplots(nrows = 2, ncols = 4)
            fig2.set_size_inches(10,7)
            fig2.suptitle("Top: Adaptive sampling\nBottom: Random sampling")

            # Adaptive sample plots
            plot_accuracy(axx2[0][0], accuracy_data[exp])
            plot_sampled_upto(axx2[0][1], true_model, est_model_data[exp], accuracy_data[exp], x_samp_data[exp], r_samp_data[exp], int(ntrials * 1/3))
            plot_sampled_upto(axx2[0][2], true_model, est_model_data[exp], accuracy_data[exp], x_samp_data[exp], r_samp_data[exp], int(ntrials * 2/3))
            plot_sampled_upto(axx2[0][3], true_model, est_model_data[exp], accuracy_data[exp], x_samp_data[exp], r_samp_data[exp], ntrials)

            if random_sampling:
                # Random sample plots
                plot_accuracy(axx2[1][0], rnd_accuracy_data[exp])
                plot_sampled_upto(axx2[1][1], true_model, rnd_est_model_data[exp], rnd_accuracy_data[exp], rnd_x_samp_data[exp], rnd_r_samp_data[exp], int(ntrials * 1/3))
                plot_sampled_upto(axx2[1][2], true_model, rnd_est_model_data[exp], rnd_accuracy_data[exp], rnd_x_samp_data[exp], rnd_r_samp_data[exp], int(ntrials * 2/3))
                plot_sampled_upto(axx2[1][3], true_model, rnd_est_model_data[exp], rnd_accuracy_data[exp], rnd_x_samp_data[exp], rnd_r_samp_data[exp], ntrials)

        print("Simulation", exp + 1, "|", round(time.time() - start_exp, 5))

    # Plot accuracy over trials
    fig3, axx3 = plt.subplots(nrows = 3, ncols = 1)  # chaning nrows from 2 to 3 to get mean plot
    fig3.suptitle("Accuracy of model estimation \n %i simulations %i trials each" % (ntests, ntrials))
    plot_all_accuracy(axx3, accuracy_data.T, rnd_accuracy_data.T)


    if corner_plot_:
        fig4, axx4 = plt.subplots(nrows = 2, ncols = 3)
        fig4.suptitle("Corner plots")

        corner_plot(axx4[0], prior_data[0,0], A, mu, sig)
        corner_plot(axx4[1], prior_data[0,-1], A, mu, sig)
#        corner_plot(axx4[2], prior, A, mu, sig)

    folder_name = "6"
    data_set_num = "2"
    cwd = os.getcwd()
    prior_data_name = "./Figures/" + folder_name + "/prob_data" + data_set_num
    accuracy_data_name = "./Figures/" + folder_name + "/accuracy_data" + data_set_num
    accuracy_data_rnd_name = "./Figures/" + folder_name + "/accuracy_data_rnd" + data_set_num
    xr_data_name = "./Figures/" + folder_name + "/stim_resp_data" + data_set_num

    info_string = "\n".join(["ntests " + str(ntests), "ntrials " + str(ntrials),"Alim: " + str(A_lim), "mulim: " + str(mu_lim), "siglim: " + str(sig_lim),
                             "stim rnge: " + str(stimuli_range), "response rnge: " + str(response_range),
                             "gen model: " + str(A_gen) + " " + str(mu_gen) + " " + str(sig_gen)])

    other_info = "Uniform prior \n\
\n\
3 random samples at start for the adaptive sampling method \n\
\
    "

    if save_prior:
        np.savez_compressed(prior_data_name, a=prior, b=prior_data, c=A, d=mu, e=sig)

        with open("Figures/" + folder_name + "/info" + data_set_num + ".txt", 'w+') as file:
            file.write(info_string + "\n" + other_info)

    if save_accuracy:
        np.savez_compressed(accuracy_data_name, a=accuracy_data)

    if save_accuracy_rnd:
        np.savez_compressed(accuracy_data_rnd_name, a=rnd_accuracy_data)

    if save_xr:
        np.savez_compressed(xr_data_name, a = x_samp_data, b = r_samp_data)

    return locals()





def plot_sampled_upto(ax, true_model, est_model_data, acc_data, x_data, r_data, trial):
    """
    Plot the true model, and sampled point responses up to a given trial
    """

    est_model = gaussian(stimuli_range, est_model_data[trial-1][0], est_model_data[trial-1][1], est_model_data[trial-1][2])

    ax.plot(stimuli_range, true_model, label='true')
    ax.plot(stimuli_range[x_data[0:trial]], response_range[r_data[0:trial]], 'bx', markersize=6, label='sampled responses')
    ax.plot(stimuli_range, est_model, label='estimated')

    text = "acc: {:0.4f}".format(acc_data[trial-1])
    ax.annotate(text, xy = (stimuli_range[0], A_gen))

    ax.set_title('Trial: %i' %trial)
#    ax5.legend()

def plot_accuracy(ax, acc_data):
    """
    Plot the accuracy of the esitmated model too the true model
    """
    ax.plot(np.arange(0, len(acc_data)), acc_data)
    ax.set_title('Accuracy')
    ax.set_ylabel("R-squared accuracy")
    ax.set_xlabel("Trial")

def plot_all_accuracy(ax, all_accuracy_data, all_accuracy_data_rnd):
    ax[0].plot(np.arange(0, len(all_accuracy_data)), all_accuracy_data)
    ax[0].set_title("Adaptive Sampling")
#    ax[0].set_xlabel("Trials")
    ax[0].set_ylabel("Accuracy - R-squared")

    ax[1].plot(np.arange(0, len(all_accuracy_data_rnd)), all_accuracy_data_rnd)
    ax[1].set_title("Random Sampling")
#    ax[1].set_xlabel("Trials")
    ax[1].set_ylabel("Accuracy - R-squared")

    x_str = "x\n[" + str(stimuli_range[0]) + " " + str(stimuli_range[-1]) + " " + str(stimuli_range[1] - stimuli_range[0]) + "]"
    textstr = "\n".join( ("A", str(A_lim), "mu", str(mu_lim), "sig", str(sig_lim), x_str))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

    ax[1].text(len(all_accuracy_data) - 2, 0.80, textstr, fontsize=7,
        verticalalignment='top', bbox = props)

#    ax[0].text(left, bottom, textstr, fontsize = 7,
#        horizontalalignment='left',
#        verticalalignment='bottom',
#        transform=ax.transAxes)

    if len(ax) == 3:
        ax[2].plot(np.mean(all_accuracy_data.T, axis = 0), label = "adaptive")
        ax[2].plot(np.mean(all_accuracy_data_rnd.T, axis = 0), label = "random")

        ax[2].set_title("Mean accuracy")
        ax[2].set_xlabel("Trials")
        ax[2].set_ylabel("Mean Accuracy - R-squared")
        ax[2].legend()


""" Live plotting """
def plot_rmse(rmse):
    """Plot rmse"""
    ax1 = plt.subplot(2,2,1)
    ax1.plot(stimuli_range, rmse)
    ax1.set_title("Σ MMSE")

    xmax = stimuli_range[np.argmax(rmse)]
    ymax = rmse.max()

    text= "x={:.0f}".format(xmax)
    ax1.annotate(text, xy=(xmax, ymax), xytext=(xmax, ymax))

def plot_prior(prior_prob):
    """Plot the prior probability of models"""
    ax2 = plt.subplot(2,2,2)
    ax2.plot(prior_prob)
    ax2.set_title("current prior")

def plot_sampled(true_model, est_func, accuracy, x_samp, r_samp, trial):
    """Plot the true model, and sampled point responses"""
    ax3 = plt.subplot(2,1,2)
    # Plot the true model
    ax3.plot(stimuli_range, true_model, label='true')
    # Plot the sampled points
    ax3.plot(stimuli_range[x_samp[0:trial]], response_range[r_samp[0:trial]], 'bx', label='sampled responses')
    # Plot the estimated model
    ax3.plot(stimuli_range, est_func, label='estimated')

    text = "acc: {:0.4f}".format(accuracy)
    ax3.annotate(text, xy = (stimuli_range[0], A_gen))

    ax3.set_title('Sampled points')
    ax3.legend()



def corner_plot(ax, prior, A_, mu_, sig_):
    """
    Input:
        param_dist : np.array (A, mu, sig) : % values for each parameter combo
        param : int : idx of the parameter to sum out leaving the other two to be plotted

    Given the distribution of probabilities for each parameter value
    plot a corner plot by summing out a given parameter

    top=0.93,
    bottom=0.055,
    left=0.075,
    right=0.935,
    hspace=0.14,
    wspace=0.12
    """

    param_dist = np.reshape(prior[:,3], [len(A_), len(mu_), len(sig_)])  # get a 3d representation of the parameters

    # marginalise out each parameter

    data0 = np.sum(param_dist, axis = 0)

    data1 = np.sum(param_dist, axis = 1)

    data2 = np.sum(param_dist, axis = 2)

    A_ = np.array([round(i,1) for i in A_])

    ax[0].contourf(data0)
    ax[0].set_xlabel("sig")
    ax[0].set_ylabel("mu")
#    ax[0].set_xlim([sig_.min(), sig_.max()])
#    ax[0].set_ylim([mu_.min(), mu_.max()])
#    ax[0].set_xticklabels(sig_)
#    ax[0].set_yticklabels(mu_)

    ax[1].contourf(data1)
    ax[1].set_xlabel("sig")
    ax[1].set_ylabel("A")
#    ax[1].set_xlim([sig_.min(), sig_.max()])
#    ax[1].set_ylim([A_.min(), A_.max()])
#    ax[1].set_xticklabels(sig_)
#    ax[1].set_yticklabels(A_)

    ax[2].contourf(data2)
    ax[2].set_xlabel("mu")
    ax[2].set_ylabel("A")
#    ax[2].set_xlim([mu_.min(), mu_.max()])
#    ax[2].set_ylim([A_.min(), A_.max()])
#    ax[2].set_xticklabels(mu_)
#    ax[2].set_yticklabels(A_)

    return ax

if __name__ == "__main__":
    #run()
    # locals().update(run())
    prior = gen_prior(A, mu, sig)
    start = time.time()
    lookuptable = lookuptable_func(prior, stimuli_range, response_range)
    print(lookuptable[0].shape)
    print(len(lookuptable[0]))
    print("lookuptable", time.time() - start)
