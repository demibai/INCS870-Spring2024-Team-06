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
    tasks = []
    fc_methods = []
    k_s = []
    accuracies = []

    for report in os.listdir(REPORTS_PATH):
        with open(REPORTS_PATH + report, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
            if report.find("baseline") != -1:
                if extract_one(data, "task") == "multi":
                    models.append("baseline-multi")
                    fc_methods.append("baseline-none")
                    k_s.append(46)
                    accuracies.append(float(extract_one(data, 'accuracy')))
                    tasks.append(extract_one(data, "task"))
                elif extract_one(data, "task") == "binary":
                    models.append("baseline-binary")
                    fc_methods.append("baseline-none")
                    k_s.append(46)
                    accuracies.append(float(extract_one(data, 'accuracy')))
                    tasks.append(extract_one(data, "task"))
            else:
                models.append(extract_one(data, "model_name"))
                fc_methods.append(extract_one(data, "feature_selection_method"))
                k_s.append(float(extract_one(data, "k")))
                accuracies.append(float(extract_one(data, 'accuracy')))
                tasks.append(extract_one(data, "task"))

    color_dict = {
        "binary": "red",
        "multi": "blue",
    }

    timestamp = str(int(time.time()))
    plt.figure(figsize=(12, 6))
    plt.scatter(fc_methods, accuracies, c=[color_dict[i] for i in tasks])
    
    for i, v in enumerate(models):
        if v == "baseline-binary":
            plt.axhline(y=accuracies[i], color='red', linestyle='--')
            plt.annotate("binary classification baseline", (fc_methods[i], accuracies[i]), fontsize=8, xytext=(-30, -10), textcoords='offset points')
        elif v == "baseline-multi":
            plt.axhline(y=accuracies[i], color='blue', linestyle='--')
            plt.annotate("multiclass classification baseline", (fc_methods[i], accuracies[i]), fontsize=8, xytext=(-30, -10), textcoords='offset points')
    for i, txt in enumerate(accuracies):
        txt = str(round(txt, 3))
        txt += ", k=" + str(float(k_s[i])) + ", " + tasks[i]
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