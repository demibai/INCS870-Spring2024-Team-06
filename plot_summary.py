import os
import matplotlib.pyplot as plt
import csv
import time

REPORTS_PATH = "reports/"
OUTPUT_PATH = "figures/"

def extract_one(data, key):
    for line in data:
        if key in line:
            return line[1]

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    models = []
    fc_methods = []
    Ks = []
    accuracies = []

    for report in os.listdir(REPORTS_PATH):
        with open(REPORTS_PATH + report, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            if report.find("baseline") != -1:
                models.append("baseline")
                fc_methods.append("baseline-none")
                Ks.append(46)
                accuracies.append(float(extract_one(data, 'accuracy')))
            else:
                models.append(extract_one(data, "model_name"))
                fc_methods.append(extract_one(data, "feature_selection_method"))
                Ks.append(float(extract_one(data, "k")))
                accuracies.append(float(extract_one(data, 'accuracy')))

    color_dict = {
        "baseline-none": "red",
        "chi2": "blue",
        "rfe": "green",
        "anova": "yellow",
        "mutual_information": "purple",
        "variance_threshold": "orange"
    }

    timestamp = str(int(time.time()))
    plt.figure(figsize=(12, 6))
    plt.scatter(fc_methods, accuracies, c=[color_dict[method] for method in fc_methods])
    
    for i, v in enumerate(models):
        if v == "baseline":
            plt.axhline(y=accuracies[i], color='red', linestyle='--')
            plt.annotate("baseline", (fc_methods[i], accuracies[i]), fontsize=8, xytext=(-10, -10), textcoords='offset points')

    for i, txt in enumerate(accuracies):
        txt = str(round(txt, 3))
        txt += ", k=" + str(int(Ks[i]))
        if i % 2 == 0:
            plt.annotate(txt, (fc_methods[i], accuracies[i]), fontsize=8, xytext=(0, 5), textcoords='offset points')
        else:
            plt.annotate(txt, (fc_methods[i], accuracies[i]), fontsize=8)
    plt.xlabel("Feature selection method")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(OUTPUT_PATH, timestamp + "_summary.png"))
    print("Summary plot saved to", os.path.join(OUTPUT_PATH, timestamp + "_summary.png"))
    
            
if __name__ == "__main__":
    main()