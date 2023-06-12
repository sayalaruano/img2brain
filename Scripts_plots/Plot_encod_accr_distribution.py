#%%
# Imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data of enconding accuracy data for the six models with numpy
acc_lr = np.loadtxt('Results/Evaluation_allmodels/corr_lrbasemodel.csv', delimiter=',')
acc_ridge = np.loadtxt('Results/Evaluation_allmodels/corr_ridge.csv', delimiter=',')
acc_lasso = np.loadtxt('Results/Evaluation_allmodels/corr_lasso.csv', delimiter=',')
acc_enet = np.loadtxt('Results/Evaluation_allmodels/corr_enet.csv', delimiter=',')
acc_knn = np.loadtxt('Results/Evaluation_allmodels/corr_kneig.csv', delimiter=',')
acc_dt = np.loadtxt('Results/Evaluation_allmodels/corr_dt.csv', delimiter=',')

#%% Simple plot test
plt.figure(figsize=(10, 8))
plt.hist(acc_lasso, bins=80)
plt.xlabel('Correlation coefficient')
plt.ylabel('Number of voxels')
plt.title('Correlation coefficients across voxels of the lasso regression model')
plt.show()

#%%
# Plotting the distribution of the accuracy of the six models in the same figure 
# with matplotlib
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(10, 8))
plt.hist(acc_lr, bins=80, alpha=0.4, label='Linear regression', color = 'orchid')
plt.hist(acc_ridge, bins=80, alpha=0.4, label='Ridge regression', color = 'orange')
plt.hist(acc_lasso, bins=80, alpha=0.4, label='Lasso regression', color = 'mediumpurple')
plt.hist(acc_enet, bins=80, alpha=0.4, label='Elastic net regression', color = 'lightgreen')
plt.hist(acc_knn, bins=80, alpha=0.4, label='K-nearest neighbors', color = 'lightcoral')
plt.hist(acc_dt, bins=80, alpha=0.4, label='Decision tree', color = 'steelblue')
plt.xlabel('Encoding accuracy (correlation coefficient)')
plt.ylabel('Number of voxels')
plt.legend()
sns.despine()
# Save figure
plt.savefig('Results/Evaluation_allmodels/Corrcoef_allmodels.png', dpi=300)  

# Show figure and close plot
plt.show() 
plt.close()

# %%
