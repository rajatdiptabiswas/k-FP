import code
import random
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from RF_fextract import kfp_features, kfp_feature_labels


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


def RF_closedworld(train_set, test_set, num_trees=1000, seed=None):
    """
    Closed world Random Forest classification of data.
    Only uses `scikit-learn` classification - does not do additional k-NN.

    `train_set`, `test_set` shape
    [[n features], (class label, instance)]
    """

    tr_data, tr_label = zip(*train_set)
    tr_data, tr_label = list(tr_data), [label[0] for label in tr_label]

    te_data, te_label = zip(*test_set)
    te_data, te_label = list(te_data), [label[0] for label in te_label]

    print("Training...\n")
    model = RandomForestClassifier(
        n_jobs=-1, n_estimators=num_trees, oob_score=True, random_state=seed
    )
    model.fit(tr_data, tr_label)

    print(f"TESTING ACCURACY    : {model.score(te_data, te_label)}")
    print()

    feature_labels = kfp_feature_labels()
    assert len(feature_labels) == len(tr_data[0]), f"Feature label size ({len(feature_labels)}) does not match feature vector size ({len(tr_data[0])})"

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
        list(zip(feature_importance_scores, feature_labels))[:25], reverse=True
    ):
        print(f"{score:>1.10f}    {label}")
    print()

    permutation_importance_scores = permutation_importance(
        model, te_data, te_label, n_repeats=10, random_state=seed
    )

    print("PERMUTATION IMPORTANCE SCORES")
    for score, label in sorted(
        list(zip(permutation_importance_scores.importances_mean, feature_labels))[:25], reverse=True
    ):
        print(f"{score:>1.10f}    {label}")
    print()


def kfp(dataset_directory: str):
    # Set `random` seed value for reproducibility
    seed = 0
    random.seed(seed)

    # Load the sample `numpy` arrays
    # X: Packet direction (-1 for incoming, +1 for outgoing, 0 for padding)
    # T: Packet timestamp (relative to the first packet, padded with -1)
    # y: Website label (categorical identifier for each website)
    # with open("./X.dill", "rb") as file:
    #     X = dill.load(file)
    # with open("./T.dill", "rb") as file:
    #     T = dill.load(file)
    # with open("./y.dill", "rb") as file:
    #     y = dill.load(file)

    X, T, y = load_dataset(dataset_directory)

    """
    X: Packet size sequences for 20910 traces, each of length 5000, padded with 0
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

    --------------------

    T: Corresponding packet timestamps, each of length 5000, padded with -1
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

    --------------------

    y: Website labels corresponding to each trace
    >>> y.shape
    (20910,)
    >>> y[:5]
    array([80., 40.,  8., 50., 97.])
    """

    # Split dataset into training and test set
    # 90% of data is used for training, remaining 10% is the test set
    # Stratification prevents class imbalance
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=0.1, random_state=seed, stratify=y
    )

    # Generate the k-FP features
    train_set = kfp_features(X_train, T_train, y_train, seed)
    test_set = kfp_features(X_test, T_test, y_test, seed)

    # Open interactive shell for debugging
    # code.interact(local=locals())

    # Free up memory by deleting the raw `numpy` arrays after feature extraction
    del X, T, y
    del X_train, X_test, T_train, T_test, y_train, y_test

    # Train and evaluate Random Forest model for closed-world website fingerprinting
    RF_closedworld(
        train_set=train_set, test_set=test_set, seed=seed
    )


def main():
    dataset_directory = "/Users/rajat/website-fingerprinting/"

    crs_list = ["slitheen"]
    site_list = ["example-neverssl", "example-overt", "neverssl-overt"]

    for crs in crs_list:
        for site in site_list:
            directory = dataset_directory + f"{crs}/dataset/{site}"
            print(f"\n{crs.upper()} / {site.upper()}\n")
            kfp(directory)


if __name__ == "__main__":
    main()
