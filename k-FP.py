import random
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from RF_fextract import kfp_features, kfp_feature_labels
from contextlib import redirect_stdout
from itertools import combinations


def load_dataset(input_dir: str, max_length=8000):
    """
    Process all dataset .tsv files in a directory and return numpy arrays.

    Each .tsv file represents a single trace and follows the format:
        <timestamp>\t<±size>

    The dataset directory contains multiple .tsv files in the format:
        <label>-<instance>.tsv
    where:
        - <label> is an integer representing the website class.
        - <instance> is a unique index for traces of that website.

    dataset/
    --------------------
    ├── 0-0.tsv
    ├── 0-1.tsv
    ├── ...
    ├── 0-98.tsv
    ├── 0-99.tsv
    ├── 1-0.tsv
    ├── 1-1.tsv
    ├── ...
    ├── 1-98.tsv
    └── 1-99.tsv

    0-0.tsv
    --------------------
    0.000000        74
    0.006548        -74
    0.006570        66
    0.007009        264
    0.018557        -66
    ...
    40.443012       -66
    40.519793       -66
    40.519808       66
    40.519815       -66
    40.519819       66

    Parameters:
        - input_dir (str): Path to the directory containing .tsv files.
        - max_length (int): Maximum sequence length. Shorter traces are padded.

    Returns:
        - X (np.ndarray): Packet size sequences (+ve for outgoing, -ve for incoming), shape (N, max_length), padded with 0.
        - T (np.ndarray): Packet timestamps, shape (N, max_length), padded with -1.
        - y (np.ndarray): Integer website labels, shape (N,).

    Example Output:
        - X: Packet size sequences for traces, each of length 5000, padded with 0
        >>> X.shape
        (20910, 5000)
        >>> X[:5]
        array([
            [   74.,   -74.,    66., ...,     0.,     0.,     0.],
            [   74.,   -74.,    66., ...,   594.,   582.,   591.],
            [   74.,   -74.,    66., ...,     0.,     0.,     0.],
            [   74.,   -74.,    66., ..., -1469.,    66.,   -66.],
            [   74.,   -74.,    66., ...,     0.,     0.,     0.]
        ], shape=(5, 5000), dtype=float32)

        - T: Corresponding packet timestamps, each of length 5000, padded with -1
        >>> T.shape
        (20910, 5000)
        >>> T[:5]
        array([
            [ 0. , 0.162228, 0.23113 , ..., -1.      , -1.      , -1.      ],
            [ 0. , 0.122405, 0.189939, ...,  4.60121 ,  4.601213,  4.601215],
            [ 0. , 0.21627 , 0.284489, ..., -1.      , -1.      , -1.      ],
            [ 0. , 0.212338, 0.283137, ..., 11.033474, 11.033477, 11.069212],
            [ 0. , 0.12259 , 0.189454, ..., -1.      , -1.      , -1.      ]
        ], shape=(5, 5000))

        - y: Website labels corresponding to each trace
        >>> y.shape
        (20910,)
        >>> y[:5]
        array([80., 40.,  8., 50., 97.])
    """

    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".tsv")])

    X_list, T_list, y_list = [], [], []

    for file_name in file_list:
        label = int(file_name.split("-")[0])  # Extract label from file name
        file_path = os.path.join(input_dir, file_name)

        data = np.loadtxt(file_path)

        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        T = data[:, 0]  # First column: timestamps
        X = data[:, 1]  # Second column: ±size, +ve (outgoing), -ve (incoming)
        # X = np.sign(
        #     data[:, 1]
        # )  # Convert packet sizes to direction: 1 (outgoing), -1 (incoming)

        # Pad sequences to max_length
        T_padded = np.full(max_length, -1.0, dtype=np.float32)
        X_padded = np.zeros(max_length, dtype=np.float32)

        length = min(len(T), max_length)
        T_padded[:length] = T[:length]
        X_padded[:length] = X[:length]

        X_list.append(X_padded)
        T_list.append(T_padded)
        y_list.append(label)

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    T = np.array(T_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, T, y


def RF_closedworld(X, T, y, feature_flags, num_trees=1000, seed=None):
    """
    Train and evaluate Random Forest model for closed-world website fingerprinting.
    Only uses `scikit-learn` classification - does not do additional k-NN.

    Parameters:
        - X (np.ndarray): Packet size sequences, shape (N, max_length).
        - T (np.ndarray): Packet timestamps, shape (N, max_length).
        - y (np.ndarray): Website class labels, shape (N,).
        - feature_flags (dict): Feature extraction configuration flags.
        - num_trees (int): Number of decision trees in the forest (default: 1000).
        - seed (int): Random seed for reproducibility (default: None).
    """

    def get_data_label(X, T, y):
        """
        Extract features data and class label from the dataset.
        """
        feature_set = kfp_features(X, T, y, seed, **feature_flags)
        # `feature_set` shape: [[n features], (class label, instance)]
        data, label = zip(*feature_set)
        data, label = list(data), [l[0] for l in label]
        return data, label

    # Split dataset into training and test set
    # 90% of data is used for training, remaining 10% is the test set
    # Stratification prevents class imbalance
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=0.1, random_state=seed, stratify=y
    )

    train_data, train_label = get_data_label(X_train, T_train, y_train)
    test_data, test_label = get_data_label(X_test, T_test, y_test)

    # Perform k-FP on 90/10 split
    model = RandomForestClassifier(
        n_jobs=-1, n_estimators=num_trees, oob_score=True, random_state=seed
    )
    model.fit(train_data, train_label)

    def compute_accuracy_bounds(model, data, labels, n_samples=1000, ci=0.95, seed=None):
        """Compute mean accuracy and confidence bounds using bootstrapping."""
        rng = np.random.default_rng(seed)
        accuracies = []

        for _ in range(n_samples):
            indices = rng.integers(0, len(data), len(data))
            X_sample = [data[i] for i in indices]
            y_sample = [labels[i] for i in indices]
            acc = model.score(X_sample, y_sample)
            accuracies.append(acc)

        mean = np.mean(accuracies)
        lower = np.percentile(accuracies, ((1 - ci) / 2) * 100)
        upper = np.percentile(accuracies, (1 - (1 - ci) / 2) * 100)
        return mean, lower, upper

    # Compute accuracy with 95% CI
    test_mean, test_low, test_high = compute_accuracy_bounds(model, test_data, test_label, seed=seed)

    print(f"TESTING ACCURACY : {test_mean:.4f} ± {(test_high - test_low)/2:.4f} (95% CI: [{test_low:.4f}, {test_high:.4f}])")
    print()

    feature_labels = kfp_feature_labels(**feature_flags)
    assert len(feature_labels) == len(train_data[0]), f"Feature label size ({len(feature_labels)}) does not match feature vector size ({len(train_data[0])})"

    feature_importance_scores = model.feature_importances_

    # print("FEATURE LABELS")
    # for label, feature in list(zip(feature_labels, tr_data[0])):
    #     if feature.is_integer():
    #         print(f"{int(feature):>20}    {label:<30}")
    #     else:
    #         print(f"{feature:20.10f}    {label:<30}")
    # print()

    print("FEATURE IMPORTANCE SCORES")
    for score, label in sorted(
        list(zip(feature_importance_scores, feature_labels)), reverse=True
    )[:25]:
        print(f"{score:>1.10f}    {label}")
    print()

    permutation_importance_scores = permutation_importance(
        model, test_data, test_label, n_repeats=10, random_state=seed
    )

    print("PERMUTATION IMPORTANCE SCORES")
    for score, label in sorted(
        list(zip(permutation_importance_scores.importances_mean, feature_labels)), reverse=True
    )[:25]:
        print(f"{score:>1.10f}    {label}")
    print()


def kfp(dataset_directory: str, features):
    """
    Load dataset and train Random Forest model for website fingerprinting.
    
    Parameters:
        - dataset_directory (str): Path to directory containing website trace .tsv files.
        - features (list): List of feature types to use ('num', 'size', 'time', 'alt').
    """

    # Set `random` seed value for reproducibility
    seed = 0
    random.seed(seed)

    # Load the `numpy` arrays
    X, T, y = load_dataset(dataset_directory)

    feature_flags = {
        'time_features': 'time' in features,
        'number_features': 'num' in features,
        'size_features': 'size' in features,
        'alternate_features': 'alt' in features and 'num' in features
    }

    RF_closedworld(
        X, T, y, feature_flags, seed=seed
    )


def main():
    experiment_date = "2026-01-01"

    dataset_directory = f"/Users/rajat/website-fingerprinting/results/{experiment_date}/"

    crs_list = ["slitheen", "waterfall"]

    dataset_list = ["dataset-tcp", "dataset-tls"]

    sites = ["example", "neverssl", "overt"]
    site_list = ["-".join(combination) for combination in combinations(sorted(sites), 2)]

    features = ["num", "size", "time", "alt"]

    for crs in crs_list:
        for dataset_type in dataset_list:
            for site in site_list:
                data_type = dataset_type.split("-")[1]

                directory = dataset_directory + f"{crs}/{dataset_type}/{site}"

                features_str = "+".join(sorted(features))
                print(f"\n{crs.upper()} / {data_type.upper()} / {site.upper()} / {features_str.upper()}\n")

                log_file_path = f"./logs/{crs}/{data_type}/{features_str}/{experiment_date}_kfp_{features_str}_{crs}_{data_type}_{site}.txt"

                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, "w", encoding="utf-8") as f:
                    with redirect_stdout(f):
                        kfp(directory, features)


if __name__ == "__main__":
    main()
