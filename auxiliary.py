# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:09:39 2019

@author: giess
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.animation import FuncAnimation



"""
Plot accuracy and mean accuracy for all tests and trials for adaptive and random.

"""  
def mean_with_error(data):
    
    mean_ = np.mean(data, axis = 0)
    
#    print("data", data.shape)
#    print("mean_", mean_.shape)
    
    err = (data - mean_) ** 2
    
    err = np.mean(err, axis = 0)
    
    return mean_, err

#%%

path = "C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"
folder = "6"

adap_acc_data = np.load(path + folder + "/accuracy_data1.npz")['a']
rnd_acc_data = np.load(path + folder + "/accuracy_data_rnd1.npz")['a']

adap_acc_data2 = np.load(path + folder + "/set2/accuracy_data2.npz")['a']

ntests = len(adap_acc_data)
ntrials = len(adap_acc_data[0]) 

#%%
    

def plot_all_accuracy(ax, all_accuracy_data, all_accuracy_data_rnd):
    ax[0].plot(np.arange(0, len(all_accuracy_data)), all_accuracy_data)
    ax[0].set_title("Adaptive Sampling")
#    ax[0].set_xlabel("Trials")
    ax[0].set_ylabel("Accuracy - R-squared")
    
    ax[1].plot(np.arange(0, len(all_accuracy_data_rnd)), all_accuracy_data_rnd)
    ax[1].set_title("Random Sampling")
#    ax[1].set_xlabel("Trials")
    ax[1].set_ylabel("Accuracy - R-squared")
    
    if len(ax) == 3:
        x = np.arange(0, ntrials)
#        ax[2].plot(np.mean(all_accuracy_data.T, axis = 0), label = "adaptive")
#        ax[2].plot(np.mean(all_accuracy_data_rnd.T, axis = 0), label = "random")
        
        mean_adap, err_adap = mean_with_error(all_accuracy_data.T)
        mean_rnd, err_rnd = mean_with_error(all_accuracy_data_rnd.T)
        
#        plt.fill_between(x, mean_adap-err_adap, mean_adap+err_adap)
#        plt.fill_between(x, mean_rnd - err_rnd, mean_rnd + err_rnd)
        
        ax[2].errorbar(x, mean_adap, yerr=err_adap, label = "adaptive")
        ax[2].errorbar(x, mean_rnd, yerr=err_rnd, label = "random")
        
        ax[2].set_title("Mean accuracy")
        ax[2].set_xlabel("Trials")
        ax[2].set_ylabel("Mean Accuracy - R-squared")
        ax[2].legend()


concat_adap_acc_data = np.concatenate((adap_acc_data, adap_acc_data2))
fig3, axx3 = plt.subplots(nrows = 3, ncols = 1)  # chaning nrows from 2 to 3 to get mean plot
fig3.suptitle("Accuracy of model estimation \n %i simulations %i trials each" % (ntests, ntrials))
plot_all_accuracy(axx3, concat_adap_acc_data.T, rnd_acc_data.T)
    
plt.show()

#%%
"""
Interactive Contour plot for all 3 variable combinations
"""

#loaded = np.load("prob_data.npz")
#prior = loaded['a']
#prior_data = loaded['b']
#A = loaded['c']
#mu = loaded['d']
#sig = loaded['e']

path = "C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"
folder = "6"

loaded = np.load(path + folder + "/prob_data1.npz")
prior = loaded['a']
prior_data = loaded['b'] # ntests, ntrials, #models, 4
A = loaded['c']
mu = loaded['d']
sig = loaded['e']

#%%
def calc_data(data, A_, mu_, sig_):
    param_dist = np.reshape(data[:,3], [len(A_), len(mu_), len(sig_)])
    data0 = np.sum(param_dist, axis = 0)
    data1 = np.sum(param_dist, axis = 1)
    data2 = np.sum(param_dist, axis = 2)
    return data0, data1, data2
   
#%%

"""Part of the above"""

plt.ion()
plt.figure(1, figsize=(20, 10))
data0, data1, data2 = calc_data(prior, A, mu, sig)
plt.subplot(131)
plt.contourf(data0); plt.colorbar()
plt.subplot(132)
plt.contourf(data1); plt.colorbar()
plt.subplot(133)
plt.contourf(data2); plt.colorbar()
for k in range(10):
    data0, data1, data2 = calc_data(prior_data[0,k], A, mu, sig)
    plt.clf(); plt.subplot(131);
    plt.suptitle('Corner plot @ trial ' + str(k+1))
    plt.contourf(data0)
    plt.xlabel("sig")
    plt.ylabel("mu")
    plt.subplot(132)
    plt.contourf(data1)
    plt.xlabel("sig")
    plt.ylabel("A")
    plt.subplot(133)
    plt.contourf(data2)
    plt.colorbar()
    plt.xlabel("mu")
    plt.ylabel("A")
    plt.draw()
    plt.pause(1)



#%%
"""Load our data"""

path = "C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"
folder = "7/set1"

loaded = np.load(path + folder + "/prob_data1.npz")

prior = loaded['a']
prior_data = loaded['b']
A = loaded['c']
mu = loaded['d']
sig = loaded['e']

#%%
"""Method to calculate the contour plot data for A, mu, sig"""

def calc_data(data, A_, mu_, sig_):
    param_dist = np.reshape(data[:,3], [len(A_), len(mu_), len(sig_)])
#    print("param:", param_dist.shape)
    data0 = np.sum(param_dist, axis = 0) # mu & sig
    data1 = np.sum(param_dist, axis = 1) # A & sig
    data2 = np.sum(param_dist, axis = 2) # A & mu
    return data0, data1, data2

#%% Mu | Sig

"""Contour + Marginal plots for Mu and Sig"""

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_contourf = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# Calculate our contour data (we can use this for the next plots as well)
test_idx = 0
trial_idx = 0
data0, data1, data2 = calc_data(prior_data[test_idx,trial_idx], A, mu, sig)

plt.figure(figsize=(8,8))

ax_contourf = plt.axes(rect_contourf)
ax_contourf.tick_params(direction='in', top=True, right=True)

ax_histx = plt.axes(rect_histx)
ax_histx.tick_params(direction='in', labelbottom=False)
ax_histx.margins(x=0)
ax_histx.spines['top'].set_visible(False)
ax_histx.spines['right'].set_visible(False)
#ax_histx.spines['left'].set_visible(False)

ax_histy = plt.axes(rect_histy)
ax_histy.tick_params(direction='in', labelleft=False)
ax_histy.margins(y=0)
ax_histy.spines['top'].set_visible(False)
ax_histy.spines['right'].set_visible(False)
#ax_histy.spines['bottom'].set_visible(False)

title_obj = plt.suptitle("Contour + Marginal plots of Mu and Sig @ trial " + str(trial_idx+1))
#plt.setp(title_obj, color = "blue")

# set axis label
labelx_obj = ax_contourf.set_xlabel("sig")
labely_obj = ax_contourf.set_ylabel("mu")

#plt.setp(labelx_obj, color = "blue"); plt.setp(labely_obj, color = "blue")

# set Y tick labels for contourf plot
ax_contourf.set_yticks(np.arange(0, 61, 1))
ax_contourf.set_yticklabels([str(x) for x in np.arange(-30, 31, 1)])

# set X tick labels for contourf plot
ax_contourf.set_xticklabels([str(x) for x in np.arange(0.1, 45.1, 5.0)])

# remove every nth tick on Y axis
labels = ax_contourf.yaxis.get_ticklabels()
for label in range(0, len(labels)):
    if label % 2 ==0:
        labels[label].set_visible(False)

# plot contourf data
ax_contourf.contourf(data0, cmap = 'viridis')

# calculate histogram data
sig_bar = data0.sum(axis = 0)
mu_bar = data0.sum(axis = 1)

# adjust histogram X axis for TOP plot 
#sig_bins = np.arange(0.1, 50.1, 5.0)
#sig_bins = len(sig)

# adjust histogram Y axis for RIGHT plot
#mu_bins = np.arange(-30, 31, 1)
#mu_bins = len(mu)

# plot histograms
#ax_histx.hist(sig_bar, bins = sig_bins)
#ax_histy.hist(mu_bar, bins = mu_bins, orientation='horizontal')
ax_histx.plot(np.arange(0.1, 50.1, 5.0), sig_bar)

# rotate the marginal mu plot
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(-90)
ax_histy.plot(np.arange(31, -30, -1), mu_bar, transform= rot + base)

plt.show()

#plt.savefig("C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"+folder+"/marginalplot/trial_" + str(trial_idx+1) + ".png", 
#            dpi = 95)
            #transparent = True)
            
#%%
loaded_stim_data = np.load(path + folder + "/stim_resp_data1.npz")

stimuli_mu_sig_cont = loaded_stim_data['a']

print(stimuli_mu_sig_cont[0,0])

#%% A | mu
            
def A_mu_contour(trial_idx):
    
    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    
    rect_contourf = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # Calculate our contour data (we can use this for the next plots as well)
    test_idx = 0
    _, _, data = calc_data(prior_data[test_idx,trial_idx], A, mu, sig)
    
    plt.figure(figsize=(8,8))
    
    ax_contourf = plt.axes(rect_contourf)
    ax_contourf.tick_params(direction='in', top=True, right=True)
    
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histx.margins(x=0)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_histy.margins(y=0)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)    
    
    # set axis label
    plt.suptitle("Contour + Marginal plots of A and Mu @ trial " + str(trial_idx+1))

    x_ticks_values = np.arange(-30, 31, 1)
    y_ticks_values = [round(x, 1) for x in np.arange(0.2, 1.3, 0.1)]
    ax_contourf.set_ylabel("A")
    ax_contourf.set_xlabel("mu")
    
    # set Y tick labels for contourf plot
    ax_contourf.set_yticks(np.arange(1, 13, 1))
    ax_contourf.set_yticklabels([str(x) for x in y_ticks_values])
    
    # set X tick labels for contourf plot
    ax_contourf.set_xticks(np.arange(0, 61, 1))
    ax_contourf.set_xticklabels([str(x) for x in x_ticks_values ])
    
    # remove even values on X-axis ( MU )
    labels = ax_contourf.xaxis.get_ticklabels()
    for label in range(0, len(labels)):
        if label % 5 != 0:
            labels[label].set_visible(False)
            
    ax_contourf.contourf(data, cmap = 'viridis')
    
    # calculate histogram data    
    x_bar  = data.sum(axis = 0)
    y_bar = data.sum(axis = 1)
    
    # plot histograms
    ax_histx.plot(np.arange(0, 61, 1), x_bar)
    
    # rotate the marginal mu plot
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)
    ax_histy.plot(np.arange(1, 13, 1), y_bar[::-1], transform= rot + base)
#    
#    plt.savefig("C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures
#/"+folder+"/marginalplot_A_mu/trial_" + str(trial_idx+1) + ".png", 
#            dpi = 95)


for i in range(0, 25):
    print(i)
    A_mu_contour(i)
            
            
#%%

"""function for plotting contour + marginals"""

def contour_marginals(trial_idx, dataset):
    
    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    
    rect_contourf = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # Calculate our contour data (we can use this for the next plots as well)
    test_idx = 0
    
    if dataset == 0:
        data, _, _ = calc_data(prior_data[test_idx,trial_idx], A, mu, sig)
    elif dataset == 1:
        _, data, _ = calc_data(prior_data[test_idx,trial_idx], A, mu, sig)
    elif dataset == 2:
        _, _, data = calc_data(prior_data[test_idx,trial_idx], A, mu, sig)
    
    plt.figure(figsize=(8,8))
    
    ax_contourf = plt.axes(rect_contourf)
    ax_contourf.tick_params(direction='in', top=True, right=True)
    
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histx.margins(x=0)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    #ax_histx.spines['left'].set_visible(False)
    
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_histy.margins(y=0)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    #ax_histy.spines['bottom'].set_visible(False)
    
    
    # set axis label
    if dataset == 0:
        plt.suptitle("Contour + Marginal plots of Mu and Sig @ trial " + str(trial_idx+1))
        
        x_ticks = np.arange(0.1, 50.1, 5.0)
        x_ticks_values = x_ticks
        y_ticks = np.arange(0, 61, 1)
        y_ticks_values = np.arange(-30, 31, 1)
        ax_contourf.set_xlabel("sig")
        ax_contourf.set_ylabel("mu")
    elif dataset == 1:
        plt.suptitle("Contour + Marginal plots of A and Sig @ trial " + str(trial_idx+1))
    
        x_ticks = np.arange(0.1, 50.1, 5.0)
        x_ticks_values = x_ticks
        y_ticks = np.arange(1, 13, 1)
        y_ticks_values = [round(x, 1) for x in np.arange(0.1, 1.3, 0.1)]
        ax_contourf.set_xlabel("sig")
        ax_contourf.set_ylabel("A")
    elif dataset == 2:
        plt.suptitle("Contour + Marginal plots of Mu and A @ trial " + str(trial_idx+1))
    
        x_ticks = np.arange(0, 61, 1)
        x_ticks_values = np.arange(-30, 31, 1)
        y_ticks = np.arange(1, 13, 1)
        y_ticks_values = [round(x, 1) for x in np.arange(0.1, 1.3, 0.1)]
        ax_contourf.set_ylabel("A")
        ax_contourf.set_xlabel("mu")
        
    # set Y tick labels for contourf plot
    ax_contourf.set_yticks(y_ticks)
    ax_contourf.set_yticklabels([str(x) for x in y_ticks_values])
    
    # set X tick labels for contourf plot
    ax_contourf.set_xticklabels([str(x) for x in x_ticks_values ])
    
    # remove even values on Y-axis if MU is being plotted
    if dataset == 0:
        labels = ax_contourf.yaxis.get_ticklabels()
        for label in range(0, len(labels)):
            if label % 2 ==0:
                labels[label].set_visible(False)
#    if dataset == 2:
#        labels = ax_contourf.xaxis.get_ticklabels()
#        for label in range(0, len(labels)):
#            if label % 2 ==0:
#                labels[label].set_visible(False)            
    
    # plot contourf data
    ax_contourf.contourf(data, cmap = 'viridis')
    
    # calculate histogram data    
#    if dataset == 2:
#        x_bar = data.sum(axis = 1)
#        y_bar = data.sum(axis = 0)
#    else:
    x_bar  = data.sum(axis = 0)
    y_bar = data.sum(axis = 1)
    
    # plot histograms
    ax_histx.plot(y_ticks, y_bar)
    
    # rotate the marginal mu plot
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)
    ax_histy.plot(x_ticks, x_bar, transform= rot + base)
    
#    plt.show()
#    plt.savefig("C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/
#"+folder+"/marginalplot_A_sig/trial_" + str(trial_idx+1) + ".png", 
#            dpi = 95)
    
 
# 0 : mu & sig
# 1 : A & sig
# 2 : A & mu
for i in range(0, 2):
    print(i)
    contour_marginals(i, 0)

#%%

"""Plot Stimuli and Response pairs"""

path = "C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"
folder = "6"

loaded_stimuli_data = np.load(path + folder + "/stim_resp_data1.npz")

stimuli_indicies = loaded_stimuli_data['a']
response_indicies = loaded_stimuli_data['b']

stimuli_range = np.arange(-120, 120, 5)

sim_idx = 5

stimuli_idx_test = stimuli_indicies[sim_idx]
stimuli_trials = stimuli_range[stimuli_idx_test]

fig, ax = plt.subplots()
ax.plot(stimuli_trials, marker='o', linestyle='')
ax.axhline(0, color = 'black', linewidth = 0.8)
ax.axvline(2, color = 'black', linewidth = 0.2)

fig.suptitle("Stimuli values sampled during one simulation\n3 rnd samples at start")
ax.set_xlabel("trial")
ax.set_ylabel("Stimuli (° offset from 0)")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# set cell here to plot stimuli seperately
"""Plot responses"""

ax2 = ax.twinx() 

response_range = np.arange(-0.5, 1.6, 0.1)

response_idx_test = response_indicies[sim_idx]
response_trials = response_range[response_idx_test]

ax2.plot(response_trials, color = 'Green', marker='x', linestyle='')

ax2.set_ylabel('Response', color= 'Green')
ax2.tick_params(axis='y', labelcolor='Green')

ax2.spines['top'].set_visible(False)


#%%

import seaborn as sns


data0, data1, data2 = calc_data(prior_data[40,20], A, mu, sig)

#sns.jointplot(x=data0[0], y=data0[1], kind='scatter')
#sns.jointplot(x=data0[0], y=data0[1], kind='hex')
sns.jointplot(x=data0[0], y=data0[1], kind='kde')
fig, ax = plt.subplots()

for i in range(0, 2):
    data0, _, _ = calc_data(prior_data[40,i], A, mu, sig)
    plot = sns.jointplot(x=data0[0], y=data0[1], kind='kde')
    plt.title("Contour plot trial " + str(i+1))


#%%

"""Folder 7 | Plot accuracy of 3 rnd at start vs 5 rnd at start"""

path = "C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"

folder = "7/set1"
folder2 = "7/set2"
folder3 = "7/set3"

loaded_set1 = np.load(path + folder + "/accuracy_data1.npz")
loaded_set_rnd1 = np.load(path + folder + "/accuracy_data_rnd1.npz")

loaded_set2 = np.load(path + folder2 + "/accuracy_data2.npz")
loaded_set_rnd2 = np.load(path + folder2 + "/accuracy_data_rnd2.npz")

loaded_set3 = np.load(path + folder3 + "/accuracy_data3.npz")
loaded_set_rnd3 = np.load(path + folder3 + "/accuracy_data_rnd3.npz")

loaded_set4 = np.load(path + "8/set1" + "/accuracy_data1.npz") # gauss prior | 5 rnd samps

loaded_set5 = np.load(path + "8/set2" + "/accuracy_data2.npz") # gauss prior | 2 rnd samps

loaded_set6 = np.load(path + "8/set3" + "/accuracy_data3.npz") # gauss prior | 3 rnd samps

#%%
"""PART OF THE ABOVE"""

accuracy_set1 = loaded_set1['a']
accuracy_set2 = loaded_set2['a']
accuracy_set3 = loaded_set3['a']

accuracy_set4 = loaded_set4['a']
accuracy_set5 = loaded_set5['a']
accuracy_set6 = loaded_set6['a']

accuracy_rnd_set1 = loaded_set_rnd1['a']
accuracy_rnd_set2 = loaded_set_rnd2['a']
accuracy_rnd_set3 = loaded_set_rnd3['a']

ntests = len(accuracy_set1)
ntrials = len(accuracy_set1[0]) 

# combine the 3 sets of random samples into 1 for averaging
accuracy_rnd_set = np.concatenate((accuracy_rnd_set1, accuracy_rnd_set2, accuracy_rnd_set3))

    
def plot_set_accuracy(ax, acc_set1, acc_set2, acc_set3, acc_set4, acc_set5, acc_set6, acc_rnd_set):

    x = np.arange(0, ntrials)
#        ax[2].plot(np.mean(all_accuracy_data.T, axis = 0), label = "adaptive")
#        ax[2].plot(np.mean(all_accuracy_data_rnd.T, axis = 0), label = "random")
    
    mean_set1, err_set1 = mean_with_error(acc_set1)
    mean_set2, err_set2 = mean_with_error(acc_set2)
    mean_set3, err_set3 = mean_with_error(acc_set3)
    
    mean_rnd, err_rnd = mean_with_error(acc_rnd_set)
    
    mean_set4, err_set4 = mean_with_error(acc_set4)
    mean_set5, err_set5 = mean_with_error(acc_set5)
    mean_set6, err_set6 = mean_with_error(acc_set6)
    
    
#        plt.fill_between(x, mean_adap-err_adap, mean_adap+err_adap)
#        plt.fill_between(x, mean_rnd - err_rnd, mean_rnd + err_rnd)
    
#    ax.errorbar(x, mean_set3, yerr=err_set3, capsize  = 2, label = "2 rnd samples")
#    ax.errorbar(x, mean_set1, yerr=err_set1, capsize  = 2, label = "3 rnd samples")
#    ax.errorbar(x, mean_set2, yerr=err_set2, capsize  = 2, label = "5 rnd samples")
#    ax.errorbar(x, mean_rnd, yerr=err_rnd, capsize  = 2, label = "all rnd samples")
#    ax.plot(x, mean_set3, label = "2 rnd samples")
#    ax.plot(x, mean_set1, label = "3 rnd samples")
#    ax.plot(x, mean_set2, label = "5 rnd samples")
    ax.plot(x, mean_set4, label = "5 rnd samples + gaus prior")
    ax.plot(x, mean_set5, label = "2 rnd samples + gaus prior")
    ax.plot(x, mean_set6, label = "3 rnd samples + gaus prior")
    ax.plot(x, mean_rnd, label = "all rnd samples", linestyle='dashed')
    
    ax.set_title("Accuracy for different # of rnd samples at start")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Mean Accuracy - R-squared")
    ax.legend()


fig3, axx3 = plt.subplots(nrows = 1, ncols = 1)  # chaning nrows from 2 to 3 to get mean plot
fig3.suptitle("Accuracy of model estimation\n%i simulations %i trials each" % (ntests, ntrials))
plot_set_accuracy(axx3, accuracy_set1, accuracy_set2, accuracy_set3, \
                  accuracy_set4, accuracy_set5, accuracy_set6, accuracy_rnd_set)
    
plt.show()

#%%

"""Plot distribution of mu parameter across trials for random and adaptive"""

path = "C:/Users/giess/OneDrive/Documents/University/FourthYear/Thesis/Code/Thesis/Figures/"

folder = "7/set1"

loaded = np.load(path + folder + "/prob_data1.npz")

prior = loaded['a']
prior_data = loaded['b']
A = loaded['c']
mu = loaded['d']
sig = loaded['e']

#%%

fig, ax = plt.subplots(1, 1)

def marginal_mu(test_idx, trial_idx):

    data, _, _ = calc_data(prior_data[test_idx,trial_idx], A, mu, sig)

    mu_bar = data.sum(axis = 1)
    print("shape", mu_bar.shape)
    
    x_ticks = np.arange(-30, 31, 1)
    x_ticks_values = x_ticks
    
    ax.set_ylabel("probability")
    ax.set_xlabel("mu parameter values")
    ax.plot(x_ticks, mu_bar, label = "trial: " + str(trial_idx+1))
    ax.legend()
    plt.show()
    
    
test = 25
marginal_mu(test, 0)
marginal_mu(test, 1)
marginal_mu(test, 2)
marginal_mu(test, 3)
marginal_mu(test, 4)
marginal_mu(test, 5)
#marginal_mu(test, 6)
#marginal_mu(test, 7)
#marginal_mu(test, 8)
#marginal_mu(test, 9)

    
    
#%%
    
"""
#######################
    Other Code
#######################
"""

    

def gaussian(x, A, mu, sig):
    """
    Gaussian distribution with an amplitude factor, sigma scale factor, and a noise component.
    """
    return (A  * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

xx = np.arange(-120, 120, 5)
rr = gaussian(xx, 100, 0, 30)
rr = [r/sum(rr) for r in rr]

plt.plot(rr)
#%%

vals = [np.random.normal(i, 0.005) for i in rr]
plt.plot(xx, vals)
plt.xlabel("Stimuli ( degrees )")
plt.ylabel("Accuracy in compensating for env force ( *1e1 )")

#%%

def parameter_probabilities(prior, A_, mu_, sig_):
    """
    Input:
        prior : np.array (m,4) : prior distribution
        ntest : int : idx of the test we want to plot
        ntrial : int : idx of the trial we want to plot
        
    Takes data of shape ( ntests, ntrials, m, 4 ) which holds the prior probability 
    distribution for m models across ntrials and ntests
    
    Use this data to calculate a 3D matrix of probability values for each parameter 
    value ( A, mu, sig ) which will be used for making corner plots
    """
    
    dist = np.zeros((len(A_), len(mu_), len(sig_)))
    
    # TODO: might be able to do this with np.reshape on the prior
    for a in range(0, len(dist)):
        for mu in range(0, len(dist[a])):
            for sig in range(0, len(dist[a,mu])):
                i = a * mu * sig
                dist[a,mu,sig] = prior[i,3]
    
    return dist
    
    
def corner_plot(ax, prior, A_, mu_, sig_):
    """
    Input:
        param_dist : np.array (A, mu, sig) : % values for each parameter combo
        param : int : idx of the parameter to sum out leaving the other two to be plotted
        
    Given the distribution of probabilities for each parameter value
    plot a corner plot by summing out a given parameter
    """
    
    param_dist = parameter_probabilities(prior, A_, mu_, sig_)
    
    data0 = np.sum(param_dist, axis = 0)
    
    data1 = np.sum(param_dist, axis = 1)
    
    data2 = np.sum(param_dist, axis = 2)
    
    A_ = [round(i,1) for i in A_]
    
    
    ax[0].contourf(data0)
    ax[0].set_xlabel("sig")
    ax[0].set_ylabel("mu")
#    ax[0].set_xticklabels(sig_)
#    ax[0].set_yticklabels(mu_)
    
    ax[1].contourf(data1)
    ax[1].set_xlabel("sig")
    ax[1].set_ylabel("A")
#    ax[1].set_xticklabels(sig_)
#    ax[1].set_yticklabels(A_)
    
    ax[2].contourf(data2)
    ax[2].set_xlabel("mu")
    ax[2].set_ylabel("A")
#    ax[2].set_xticklabels(mu_)
#    ax[2].set_yticklabels(A_)
    
    return



"""
Save and Load data
"""
def save_data(A, B, file_name = "model_data"):
    np.savez_compressed(file_name, a=A, b=B)
    
def load_data(file_name):
    loaded = np.load(file_name)
    return loaded['a'], loaded['b']


