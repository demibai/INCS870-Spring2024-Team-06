import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import time
import joblib

DATASET_PATH = "unsw_nb15/"
FIGURES_PATH = "figures/"
MODELS_PATH = "models/"
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"


def load_dataset(dataset_path=DATASET_PATH, train_file=TRAIN_FILE, test_file=TEST_FILE):
    train_set_path = os.path.join(dataset_path, train_file)
    test_set_path = os.path.join(dataset_path, test_file)
    train_data = pd.read_csv(train_set_path)
    test_data = pd.read_csv(test_set_path)

    return train_data, test_data


# def load_model_from_pickle(model_path):
#     model = joblib.load(model_path)
#     return model


def cat_to_num(data):
    # protocol
    data["proto"] = data["proto"].astype("category")
    data["proto"] = data["proto"].cat.codes

    # service
    data["service"] = data["service"].astype("category")
    data["service"] = data["service"].cat.codes

    # state
    data["state"] = data["state"].astype("category")
    data["state"] = data["state"].cat.codes

    # attack category
    data["attack_cat"] = data["attack_cat"].astype("category")
    data["attack_cat"] = data["attack_cat"].cat.codes

    return data


def main():
    # Load dataset
    train_data, test_data = load_dataset()
    train_data = cat_to_num(train_data)
    test_data = cat_to_num(test_data)

    print("Train dataset shape:", train_data.shape)
    print("Test dataset shape:", test_data.shape)

    y_train = train_data["attack_cat"]
    X_train = train_data.drop(["id", "label", "attack_cat"], axis=1)

    y_test = test_data["attack_cat"]
    X_test = test_data.drop(["id", "label", "attack_cat"], axis=1)

    # Model training
    start_time = time.time()
    model = XGBClassifier()
    model.fit(X_train, y_train)
    end_time = time.time()
    print("Training time:", round(end_time - start_time, 2), "seconds")
    y_pred = model.predict(X_test)

    # Freeze model to disk
    timestamp = int(time.time())
    model_name = model.__class__.__name__.lower()
    joblib.dump(model, MODELS_PATH + str(timestamp) + "_" + model_name + ".pkl")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Classification report
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("reports/" + str(timestamp) + "_report.csv")

    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.savefig("figures/" + str(timestamp) + "_importance.png")


if __name__ == "__main__":
    main()
    print("Done!")
