from matplotlib import pyplot as plt
import pandas as pd
from train_eval import load_dataset
import os
import sys

# Features to plot
CAT_FEATURES = ['attack_cat', 'label']
NUM_FEATURES = ['dttl', 'ct_dst_sport_ltm']

OUTPUT_PATH = "figures/dataset_plots/"

def main(**kwargs):
    train_data, test_data = load_dataset()
    dataset_label = None
    if 'set' not in kwargs:
        data = train_data
        dataset_label = "train"
    else:
        data = train_data if kwargs['set'] == 'train' else test_data
        dataset_label = "train" if kwargs['set'] == 'train' else "test"

    path = os.path.join(OUTPUT_PATH, dataset_label + "/")
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot categorical features
    for feature in CAT_FEATURES:
        plt.figure(figsize=(6, 6))
        data[feature].value_counts().plot(kind='pie', title='feature: ' + feature, fontsize=6)
        plt.savefig(path + feature + "_pie.png")

    # Plot numerical features
    for feature in NUM_FEATURES:
        plt.figure()
        data[feature].plot(kind='hist', title='feature: ' + feature, fontsize=6)
        plt.savefig(path + feature + "_hist.png")

    print("Plots saved to " + path)

if __name__ == "__main__":
    main(**dict(arg.split('=') for arg in sys.argv[1:]) if len(sys.argv) > 1 else {})