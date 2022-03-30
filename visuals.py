###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
from sklearn.metrics import f1_score, accuracy_score
from time import time
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#
# Display inline matplotlib plots with IPython
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
###########################################


def distribution(data, transformed=False):
    """
    Visualization code for displaying skewed distributions of features
    """

    # Create figure
    fig = plt.figure(figsize=(11, 5))

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins=25, color='#00A0A0')
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features",
                     fontsize=16, y=1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features",
                     fontsize=16, y=1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = plt.subplots(2, 3, figsize=(11, 8))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):

                # Creative plot code
                ax[j//3, j % 3].bar(i+k*bar_width, results[learner]
                                    [i][metric], width=bar_width, color=colors[k])
                ax[j//3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j % 3].set_xlabel("Training Set Size")
                ax[j//3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0,
                     linewidth=1, color='k', linestyle='dashed')
    ax[1, 1].axhline(y=accuracy, xmin=-0.1, xmax=3.0,
                     linewidth=1, color='k', linestyle='dashed')
    ax[0, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1,
                     color='k', linestyle='dashed')
    ax[1, 2].axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1,
                     color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    plt.legend(handles=patches, bbox_to_anchor=(-.80, 2.53),
               loc='upper center', borderaxespad=0., ncol=3, fontsize='x-large')

    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models",
                 fontsize=16, x=0.63, y=1.05)
    # Tune the subplot layout
    # Refer - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html for more details on the arguments
    plt.subplots_adjust(left=0.125, right=1.2, bottom=0.1,
                        top=0.9, wspace=0.2, hspace=0.3)
    plt.tight_layout()
    plt.show()


def feature_plot(importances, X_train, y_train):

    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Create the plot
    fig = plt.figure(figsize=(18, 5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    plt.bar(np.arange(5), values, width=0.2, align="center", color='#00A000',
            label="Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(values), width=0.2, align="center", color='#00A0A0',
            label="Cumulative Feature Weight")
    plt.xticks(np.arange(5), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize=12)
    plt.xlabel("Feature", fontsize=12)

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()


def visualize_crosstabs(df: pd.DataFrame, categorical_features: list, target: str):
    """
    Visualize the crosstab of the target variable and the categorical features.

    Args:
        df (pd.DataFrame): input dataframe
        categorical_features (list): list of categorical features
        target (str): target variable
    """
    import matplotlib.pyplot as plt
    import math

    columns_per_row = 3
    plot_rows = math.ceil(len(categorical_features) / columns_per_row)

    fig, axes = plt.subplots(plot_rows,
                             columns_per_row,
                             figsize=(15, plot_rows * 4),
                             constrained_layout=True)
    # Create crosstabs to show distribution the values of each categorical against income
    i = j = 0
    for feat in categorical_features:
        table = pd.crosstab(df[feat], df[target])
        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',
                                                           ax=axes[j],
                                                           stacked=True)
        j += 1
        if j == columns_per_row:
            j = 0
            i += 1


def visualize_numerical_features(df: pd.DataFrame, numerical_features: list, target: str = None, kind: str = 'hist'):
    """
    Visualize the distribution of numerical features.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        numerical_features (list): The list of numerical features to visualize.
        criteria (str): The target column. Assumes categorical and binary target.
        kind: The type of plot to use.
    Returns: None
    """
    if target != None:
        columns_per_row = 2
    else:
        columns_per_row = 1
    plot_rows = len(numerical_features)

    if kind == 'hist':
        sharey = True
    elif kind == 'box':
        sharey = False
    else:
        raise ValueError('kind must be either hist or box')

    fig, axes = plt.subplots(plot_rows,
                             columns_per_row,
                             figsize=(8, plot_rows * 3),
                             constrained_layout=False,
                             sharey=sharey)

    plt.subplots_adjust(hspace=0.5)

    # Create crosstabs to show distribution the values of each categorical against income
    col = 0
    if target != None:
        for target_val in df[target].unique():
            row = 0
            query_df = df.query(f"{target} == {target_val}")
            for feat in numerical_features:
                if kind == 'hist':
                    query_df[feat].plot(kind=kind, ax=axes[row, col])
                elif kind:
                    query_df[feat].plot(kind=kind, ax=axes[row, col], vert=False)
                plt.setp(axes[row, col], xlabel=f"{feat} for {target} == {target_val}")
                row += 1
                
            col += 1
    else:
        row = 0
        query_df = df
        for feat in numerical_features:
            if kind == 'hist':
                query_df[feat].plot(kind=kind, ax=axes[row])
            elif kind:
                query_df[feat].plot(kind=kind, ax=axes[row], vert=False)
            plt.setp(axes[row], xlabel=f"{feat}")
            row += 1
