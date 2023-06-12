#%%
# Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec
from matplotlib import style
import seaborn as sns

# Load the validation data from a csv file
fmri_val = np.loadtxt('Results/Predictions/fmri_val.csv', delimiter=',')

# Load the predicted data from a csv file
fmri_val_pred = np.loadtxt('Results/Predictions/fmri_val_pred.csv', delimiter=',')

#%% Combined figure of the predicted and actual BOLD signal for the top 1 voxel and the voxel with the lowest correlation coefficient
# Using the style for the plot
plt.style.use('seaborn-ticks')

# Function to create a subtitle for the subplots
def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontsize=15)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

# Create a figure with 2x2 subplots
rows = 2
cols = 2
fig, axs = plt.subplots(ncols=2, nrows=2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(11, 9))

# Plot the fmri predictions vs the fmri values for the top 1 voxel
axs[0, 0].plot(fmri_val[:,2377], label='Actual data', color='steelblue')
axs[0, 0].plot(fmri_val_pred[:,2377], label='Predicted data', color='lightcoral')
axs[0, 0].set_xlabel('Image')
axs[0, 0].set_ylabel('BOLD')
axs[0, 0].legend(fontsize=11, loc='lower right')
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].text(-0.1, 1.1, "A)", transform=axs[0, 0].transAxes, size=13)

# Plot the scatter plot of the predicted and actual BOLD signal for the top 1 voxel
axs[0, 1].scatter(fmri_val[:,2377], fmri_val_pred[:,2377], color='steelblue')
axs[0, 1].set_xlabel('Actual BOLD')
axs[0, 1].set_ylabel('Predicted BOLD')
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].text(-0.1, 1.1, "B)", transform=axs[0, 1].transAxes, size=13)

# Plot the fmri predictions vs the fmri values for the voxel with the lowest correlation coefficient
axs[1, 0].plot(fmri_val[:,3275], label='Actual data', color='steelblue')
axs[1, 0].plot(fmri_val_pred[:,3275], label='Predicted data', color='lightcoral')
axs[1, 0].set_xlabel('Image')
axs[1, 0].set_ylabel('BOLD')
axs[1, 0].legend(fontsize=11, loc='lower right')
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].text(-0.1, 1.1, "C)", transform=axs[1, 0].transAxes, size=13)

# Plot the scatter plot of the predicted and actual BOLD signal for the voxel with the lowest correlation coefficient
axs[1, 1].scatter(fmri_val[:,3275], fmri_val_pred[:,3275], color='steelblue')
axs[1, 1].set_xlabel('Actual BOLD')
axs[1, 1].set_ylabel('Predicted BOLD')
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].text(-0.1, 1.1, "D)", transform=axs[1, 1].transAxes, size=13)

# Create grid for the subplots and add the subtitles
grid = plt.GridSpec(rows, cols)
create_subtitle(fig, grid[0, ::], 'Voxel 2377')
create_subtitle(fig, grid[1, ::], 'Voxel 3275')
fig.tight_layout()
fig.set_facecolor('w')

# Save the figure
plt.savefig('Results/Evaluation_allmodels/fmri_pred_actual.png', dpi=300, bbox_inches='tight')

# %%
