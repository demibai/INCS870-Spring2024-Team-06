import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from train_eval import load_dataset, cat_to_num

# Dataset visualization

OUTPUT_PATH = "figures/dataset_plots/"

def plot_histogram(data, feature):
    dir_path = os.path.join(OUTPUT_PATH, "histograms")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Plotting histogram for", feature)
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(8, 8))
    file_prefix = feature + "_hist"
    sns.histplot(data[feature], kde=True)

    plt.axvline(data[feature].mean(), color='r', linestyle='--')
    plt.annotate(f'Mean: {data[feature].mean():.2f}', xy=(data[feature].mean(), 0), xytext=(data[feature].mean(), 0.1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.title(f'Histogram of {feature}')

    plt.savefig(os.path.join(dir_path, file_prefix + ".png"))

def plot_count(data, feature):
    dir_path = os.path.join(OUTPUT_PATH, "count_plots")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Plotting count for", feature)
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(8, 8))
    file_prefix = feature + "_count"
    feature_count = data[feature].value_counts()
    feature_count = feature_count.reset_index()
    feature_count.columns = [feature, 'count']
    sns.barplot(x='count', y=feature, data=feature_count.head(10))
    plt.title(f'Count of {feature}')
    plt.savefig(os.path.join(dir_path, file_prefix + ".png"))
    
def plot_box(data, feature, against):
    dir_path = os.path.join(OUTPUT_PATH, "box_plots")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Plotting boxplot for", against, "against", feature)
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(12, 12))
    file_prefix = feature + "_box"
    sns.boxplot(x=feature, y=against, data=data)
    plt.title(f'Boxplot of {against} against {feature}')
    plt.savefig(os.path.join(dir_path, file_prefix + ".png"))

def plot_scatter(data, feature1, feature2):
    dir_path = os.path.join(OUTPUT_PATH, "scatter_plots")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print("Plotting scatter for", feature1, "against", feature2)
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(12, 12))
    file_prefix = feature1 + "_" + feature2 + "_scatter"
    sns.scatterplot(x=feature1, y=feature2, data=data)
    plt.title(f'Scatter of {feature1} against {feature2}')
    plt.savefig(os.path.join(dir_path, file_prefix + ".png"))


def plot_correlation(data):
    dir_path = os.path.join(OUTPUT_PATH, "correlation_matrix")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data = cat_to_num(data)
    print("Plotting correlation matrix")
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(20, 20))
    file_prefix = "correlation_matrix"
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation matrix')
    plt.savefig(os.path.join(dir_path, file_prefix + ".png"))

def plot_pair(data):
    dir_path = os.path.join(OUTPUT_PATH, "pair_plots")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data = cat_to_num(data)
    print("Plotting pairplot")
    plt.clf()
    plt.rcParams.update(plt.rcParamsDefault)
    file_prefix = "pair_plot"
    sns.pairplot(data)
    plt.title(f'Pairplot')
    plt.savefig(os.path.join(dir_path, file_prefix + ".png"))



def main():
    train_data, test_data = load_dataset()
    # plot_histogram(train_data, 'dur')
    # plot_histogram(train_data, 'sbytes')
    # plot_histogram(train_data, 'dbytes')

    # plot_count(train_data, 'proto')
    # plot_count(train_data, 'service')
    # plot_count(train_data, 'state')

    # plot_box(train_data, 'dur', 'attack_cat')
    # plot_box(train_data, 'sbytes', 'attack_cat')
    # plot_box(train_data, 'dbytes', 'attack_cat')

    # plot_scatter(train_data, 'dur', 'sbytes')
    # plot_scatter(train_data, 'sbytes', 'dbytes')

    # plot_correlation(train_data)

    plot_pair(train_data)
if __name__ == '__main__':
    main()