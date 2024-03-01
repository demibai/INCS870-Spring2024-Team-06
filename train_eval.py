import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE, RFECV, VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from configparser import ConfigParser
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os
import time
import joblib
import sys

constants = ConfigParser()
constants.read("constants.ini")


# Load the training dataset and testing dataset
def load_dataset(pca=False):
    train_set_path = os.path.join(constants.get("CONSTANTS", "DATASET_PATH"), constants.get("CONSTANTS", "TRAIN_FILE"))
    test_set_path = os.path.join(constants.get("CONSTANTS", "DATASET_PATH"), constants.get("CONSTANTS", "TEST_FILE"))
    train_data = pd.read_csv(train_set_path)
    test_data = pd.read_csv(test_set_path)

    return train_data, test_data


def normalize_data(data):
    cnt = 0
    for feature in data.columns:
        if data[feature].dtype == "float64":
            data[feature] = np.log1p(data[feature])
            cnt += 1
    print("[Normalization] Normalized", cnt, "numerical features.")
    return data


# Load a pre-trained machine learning model stored in a pickle file.
def load_model_from_pickle(model_path):
    model = joblib.load(model_path)
    return model


# Preprocess the dataset
# Transform original data type into a categorical data type.
# Then encode the categories as integer codes.
def cat_to_num(data):
    # protocol
    data["proto"] = LabelEncoder().fit_transform(data["proto"])

    # service
    data["service"] = LabelEncoder().fit_transform(data["service"])

    # state
    data["state"] = LabelEncoder().fit_transform(data["state"])

    # attack category
    data["attack_cat"] = LabelEncoder().fit_transform(data["attack_cat"])


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
    task = kwargs.get("task", "multi")
    if task not in ["multi", "binary"]:
        raise ValueError("Invalid task. Use 'multi' or 'binary'.")

    # Load dataset
    train_data, test_data = load_dataset()
    train_data = cat_to_num(train_data)
    test_data = cat_to_num(test_data)

    print("[Dataset] Train dataset shape:", train_data.shape)
    print("[Dataset] Test dataset shape:", test_data.shape)

    corr_matrix = train_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    y_train = train_data["attack_cat"] if task == "multi" else train_data["label"]
    x_train = train_data.drop(["id", "label", "attack_cat"] + to_drop, axis=1)

    y_test = test_data["attack_cat"] if task == "multi" else test_data["label"]
    x_test = test_data.drop(["id", "label", "attack_cat"] + to_drop, axis=1)

    print("[Dataset] Train dataset shape after dropping highly correlated features:", x_train.shape)
    print("[Dataset] Test dataset shape after dropping highly correlated features:", x_test.shape)

    # Normalize data
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)

    k = kwargs.get("k", None)
    x_train, x_test, fs_time, k = feature_selection(x_train, y_train, x_test, kwargs.get("method", None), float(k) if k else None)

    # Dimensionality reduction
    if kwargs.get("pca", None):
        # Scale data
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train))
        X_test = pd.DataFrame(scaler.transform(x_test))
        pca = PCA(n_components=int(kwargs.get("pca")) if kwargs.get("pca") else x_train.shape[1])
        x_train = pd.DataFrame(pca.fit_transform(x_train))
        x_test = pd.DataFrame(pca.transform(X_test))
        print("[PCA] Train dataset shape after PCA:", x_train.shape)
        print("[PCA] Test dataset shape after PCA:", X_test.shape)

    if kwargs.get("model_path", None):
        # Load model from local file
        model = load_model_from_pickle(kwargs.get("model_path"))
        print("[Model] Model loaded from", kwargs.get("model_path"))
        model_features = model.get_booster().feature_names
        y_pred = model.predict(X_test[model_features])
        verbose_output = False
    else:
        # Model training
        grid_search = True if kwargs.get("grid_search", "false") == "true" else False
        if grid_search:
            print("[Model] Using GridSearchCV to find best hyperparameters")
            xgboost_params = {
                "objective": ["multi:softmax"] if task == "multi" else ["reg:squaredlogerror"],
                "eval_metric": ["mlogloss", "auc"] if task == "multi" else ["logloss", "auc"],
                "min_child_weight": [1],
                "max_depth": [6, 8],
                "num_class": [10] if task == "multi" else [1],
                "learning_rate": [0.01, 0.3],
                "n_estimators": [100, 1000],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            }
            model = XGBClassifier()
            grid = GridSearchCV(model, xgboost_params, cv=StratifiedKFold(n_splits=5), n_jobs=5, verbose=2, scoring="accuracy")
            start_time = time.time()
            print("[Model] Training started")
            grid.fit(x_train, y_train)
            end_time = time.time()
            model_training_time = round(end_time - start_time, 2)
            print("[Model] Training time:", model_training_time, "seconds")
            print("[Model] Best parameters found by GridSearchCV:", grid.best_params_)
            model = grid.best_estimator_
        else:
            xgboost_params = {
                "objective": "multi:softmax" if task == "multi" else "reg:squaredlogerror",
                "eval_metric": "mlogloss" if task == "multi" else "logloss",
                "min_child_weight": 1,
                "max_depth": 6,
                "num_class": 10 if task == "multi" else 1,
                "learning_rate": 0.3,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            model = XGBClassifier(**xgboost_params)
            start_time = time.time()
            print("[Model] Training started")
            model.fit(x_train, y_train)
            end_time = time.time()
            model_training_time = round(end_time - start_time, 2)
            print("[Model] Training time:", model_training_time, "seconds")
        y_pred = model.predict(x_test)
        verbose_output = True

    timestamp = int(time.time())
    if not kwargs.get("model_path", None):
        # Freeze model to disk
        print("[Model] Freezing params to disk")
        model_name = model.__class__.__name__.lower()
        file_prefix = str(timestamp) + "_" + model_name + "_" + kwargs.get("method", "none") + "_"
        file_prefix += (str(k) if k else "all") + "_" + task
        joblib.dump(model, constants.get("CONSTANTS", "MODELS_PATH") + file_prefix + "_model" + ".pkl")
        print("[I/O] Model saved to", constants.get("CONSTANTS", "MODELS_PATH") + file_prefix + ".pkl")
        file_prefix += str(k) if k else "all"
        joblib.dump(model, constants.get("CONSTANTS", "MODELS_PATH") + file_prefix + "_model" + ".pkl")
        print("[Model] Model saved to", constants.get("CONSTANTS", "MODELS_PATH") + file_prefix + ".pkl")
    else:
        file_prefix = str(timestamp) + "_loaded_model"

    accuracy = accuracy_score(y_test, y_pred)
    print("[Model] Accuracy: %.2f%%" % (accuracy * 100.0))

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    if verbose_output:
        report_df.loc[""] = ""
        report_df.loc["model_name"] = model_name
        report_df.loc["task"] = task
        report_df.loc["pca"] = kwargs.get("pca", "none")
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
    print("[I/O] Report saved to", "reports/" + file_prefix + "_report.csv")

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
    print("[I/O] Feature importance plot saved to", "figures/" + file_prefix + "_importance.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test))
    cm_display.plot(cmap="Oranges")
    plt.savefig("figures/" + file_prefix + "_confusion_matrix.png")
    print("[I/O] Confusion matrix plot saved to", "figures/" + file_prefix + "_confusion_matrix.png")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]) if len(sys.argv) > 1 else {})