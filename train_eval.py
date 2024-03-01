import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from configparser import ConfigParser
import os
import time
import joblib
import sys

constants = ConfigParser()
constants.read("constants.ini")

dataset_path = constants.get("CONSTANTS", "DATASET_PATH")
figures_path = constants.get("CONSTANTS", "FIGURES_PATH")
models_path = constants.get("CONSTANTS", "MODELS_PATH")
train_file = constants.get("CONSTANTS", "TRAIN_FILE")
test_file = constants.get("CONSTANTS", "TEST_FILE")


# Load the training dataset and testing dataset
def load_dataset():
    train_set_path = os.path.join(dataset_path, train_file)
    test_set_path = os.path.join(dataset_path, test_file)
    train_data = pd.read_csv(train_set_path)
    test_data = pd.read_csv(test_set_path)

    return train_data, test_data


# Load a pre-trained machine learning model stored in a pickle file.
def load_model_from_pickle(model_path):
    model = joblib.load(model_path)
    return model


# Preprocess the dataset
# Transform original data type into a categorical data type.
# Then encode the categories as integer codes.
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

def feature_selection(x_train, y_train, x_test, method, k=None, cv=None):
    # Feature selection sub-routine
    feature_selection_time = 0
    if method == "rfe" or method == "rfecv":
        # Recursive Feature Elimination
        if not k:
            k = 20
        else:
            k = int(k)
        print("[Feature Selection] Using Recursive Feature Elimination,", "selecting", str(k), "features.")
        start_time = time.time()
        model = XGBClassifier()
        if method == "rfecv":
            cv = 5 if not cv else int(cv)
            print("[Feature Selection] Using RFECV, splitting dataset into %d folds." % cv)
            rfe = RFECV(model, step=1, min_features_to_select=k,cv=cv)
        else:
            rfe = RFE(model, n_features_to_select=k)
        fit = rfe.fit(x_train, y_train)
        selected_features = fit.get_support(indices=True)
        x_train = x_train[x_train.columns[selected_features]]
        x_test = x_test[x_test.columns[selected_features]]
        end_time = time.time()
        feature_selection_time = round(end_time - start_time, 2)
        print("[Feature Selection] RFE took", feature_selection_time, "seconds")
    elif method == "variance_threshold":
        if not k:
            k = 0.0
        print("[Feature Selection] Using Variance Threshold", "using", str(k), "as threshold.")
        start_time = time.time()
        vt = VarianceThreshold(threshold=k)
        fit = vt.fit(x_train, y_train)
        selected_features = fit.get_support(indices=True)
        x_train = x_train[x_train.columns[selected_features]]
        x_test = x_test[x_test.columns[selected_features]]
        end_time = time.time()
        feature_selection_time = round(end_time - start_time, 2)
        print("[Feature Selection] Variance Threshold took", feature_selection_time, "seconds")
    elif method == "chi2":
        if not k:
            k = 20
        else:
            k = int(k)
        print("[Feature Selection] Using SelectKBest,", "selecting", str(k), "features.")
        start_time = time.time()
        skb = SelectKBest(chi2, k=k)
        fit = skb.fit(x_train, y_train)
        selected_features = fit.get_support(indices=True)
        x_train = x_train[x_train.columns[selected_features]]
        x_test = x_test[x_test.columns[selected_features]]
        end_time = time.time()
        feature_selection_time = round(end_time - start_time, 2)
        print("[Feature Selection] SelectKBest took", feature_selection_time, "seconds")
    elif method == "anova":
        if not k:
            k = 20
        else:
            k = int(k)
        print("[Feature Selection] Using SelectKBest,", "selecting", str(k), "features.")
        start_time = time.time()
        skb = SelectKBest(f_classif, k=k)
        fit = skb.fit(x_train, y_train)
        selected_features = fit.get_support(indices=True)
        x_train = x_train[x_train.columns[selected_features]]
        x_test = x_test[x_test.columns[selected_features]]
        end_time = time.time()
        feature_selection_time = round(end_time - start_time, 2)
        print("[Feature Selection] SelectKBest took", feature_selection_time, "seconds")
    elif method == "mutual_information":
        if not k:
            k = 20
        else:
            k = int(k)
        print("[Feature Selection] Using SelectKBest,", "selecting", str(k), "features.")
        start_time = time.time()
        skb = SelectKBest(mutual_info_classif, k=k)
        fit = skb.fit(x_train, y_train)
        selected_features = fit.get_support(indices=True)
        x_train = x_train[x_train.columns[selected_features]]
        x_test = x_test[x_test.columns[selected_features]]
        end_time = time.time()
        feature_selection_time = round(end_time - start_time, 2)
        print("[Feature Selection] SelectKBest took", feature_selection_time, "seconds")
    else:
        print("[Feature Selection] No feature selection method specified. Using all features.")
    print("[Feature Selection] Train dataset shape after feature selection:", x_train.shape)
    print("[Feature Selection] Test dataset shape after feature selection:", x_test.shape)

    return x_train, x_test, feature_selection_time, k


def main(**kwargs):
    # Load dataset
    train_data, test_data = load_dataset()
    train_data = cat_to_num(train_data)
    test_data = cat_to_num(test_data)

    print("[Dataset] Train dataset shape:", train_data.shape)
    print("[Dataset] Test dataset shape:", test_data.shape)

    y_train = train_data["attack_cat"]
    x_train = train_data.drop(["id", "label", "attack_cat"], axis=1)

    y_test = test_data["attack_cat"]
    x_test = test_data.drop(["id", "label", "attack_cat"], axis=1)

    k = kwargs.get("k", None)
    x_train, x_test, fs_time, k = feature_selection(x_train, y_train, x_test,
                                                    kwargs.get("method", None), float(k) if k else None)

    if kwargs.get("model_path", None):
        # Load model from local file
        model = load_model_from_pickle(kwargs.get("model_path"))
        print("[Model] Model loaded from", kwargs.get("model_path"))
        model_features = model.get_booster().feature_names
        y_pred = model.predict(x_test[model_features])
        VERBOSE = False
    else:
        # Model training
        start_time = time.time()
        xgboost_params = {
            "objective": "multi:softprob",
            "min_child_weight": 1,
            "max_depth": 6,
            "num_class": 10,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "reg_lambda": 1,
            "reg_alpha": 0
        }
        model = XGBClassifier(**xgboost_params)
        model.fit(x_train, y_train)
        end_time = time.time()
        model_training_time = round(end_time - start_time, 2)
        print("[Model] Training time:", model_training_time, "seconds")
        y_pred = model.predict(x_test)
        VERBOSE = True

    timestamp = int(time.time())
    if not kwargs.get("model_path", None):
        # Freeze model to disk
        print("[Model] Freezing params to disk")
        model_name = model.__class__.__name__.lower()
        file_prefix = str(timestamp) + "_" + model_name + "_" + kwargs.get("method", "none") + "_"
        file_prefix += str(k) if k else "all"
        joblib.dump(model, models_path + file_prefix + "_model" + ".pkl")
        print("[Model] Model saved to", models_path + file_prefix + ".pkl")
    else:
        file_prefix = str(timestamp) + "_loaded_model"

    accuracy = accuracy_score(y_test, y_pred)
    print("[Model] Accuracy: %.2f%%" % (accuracy * 100.0))

    # Classification report
    print("[Model] Classification report")
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    if VERBOSE:
        # Verbose report
        report_df.loc[""] = ""
        report_df.loc["model_name"] = model_name
        report_df.loc["model_training_time"] = model_training_time
        report_df.loc["feature_selection_method"] = kwargs.get("method", "none")
        report_df.loc["k"] = k
        report_df.loc["feature_selection_time"] = fs_time
        report_df.loc["accuracy"] = accuracy
        report_df.loc["recall"] = report_df.loc["weighted avg"]["recall"]
        report_df.loc["precision"] = report_df.loc["weighted avg"]["precision"]
        report_df.loc["f1_score"] = report_df.loc["weighted avg"]["f1-score"]

    # Save report to disk
    report_df.to_csv("reports/" + file_prefix + "_report.csv")

    # Plot feature importance
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, x_train.columns[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.savefig("figures/" + file_prefix + "_importance.png")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]) if len(sys.argv) > 1 else {})

