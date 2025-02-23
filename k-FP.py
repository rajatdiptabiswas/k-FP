import code
import random
import dill
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from RF_fextract import kfp_features


def RF_closedworld(train_set, valid_set, test_set, num_trees=1000, seed=None):
    """
    Closed world Random Forest classification of data.
    Only uses `scikit-learn` classification - does not do additional k-NN.
    """

    tr_data, tr_label = list(zip(*train_set))
    tr_label = list(zip(*tr_label))[0]

    val_data, val_label = list(zip(*valid_set))
    val_label = list(zip(*val_label))[0]

    te_data, te_label = list(zip(*test_set))
    te_label = list(zip(*te_label))[0]

    print("Training...")
    model = RandomForestClassifier(
        n_jobs=-1, n_estimators=num_trees, oob_score=True, random_state=seed
    )
    model.fit(tr_data, tr_label)

    print(f"Feature importance scores:\n{model.feature_importances_}")

    print(f"Validation Accuracy: {model.score(val_data, val_label)}")
    print(f"Testing Accuracy   : {model.score(te_data, te_label)}")


if __name__ == "__main__":
    # Set `random` seed value for reproducibility
    seed = 0
    random.seed(seed)

    # Load the sample `numpy` arrays
    # X: Packet direction (-1 for incoming, +1 for outgoing, 0 for padding)
    # T: Packet timestamp (relative to the first packet, padded with -1)
    # y: Website label (categorical identifier for each website)
    with open("./X.dill", "rb") as file:
        X = dill.load(file)
    with open("./T.dill", "rb") as file:
        T = dill.load(file)
    with open("./y.dill", "rb") as file:
        y = dill.load(file)

    """
    X: Packet direction sequences for 20910 traces, each of length 5000, padded with 0
    >>> X.shape
    (20910, 5000)
    >>> X[:5]
    array([
        [ 1., -1.,  1., ...,  0.,  0.,  0.],
        [ 1., -1.,  1., ..., -1., -1., -1.],
        [ 1., -1.,  1., ...,  0.,  0.,  0.],
        [ 1., -1.,  1., ..., -1., -1.,  1.],
        [ 1., -1.,  1., ...,  0.,  0.,  0.]
    ], shape=(5, 5000))

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

    y: Website labels corresponding to each trace
    >>> y.shape
    (20910,)
    >>> y[:5]
    array([80., 40.,  8., 50., 97.])
    """

    # Split dataset into training, validation, and test sets
    # 80% of data is used for training
    # Remaining 20% is split equally into validation and test sets
    X_train, X_test_val, T_train, T_test_val, y_train, y_test_val = train_test_split(
        X, T, y, test_size=0.2, random_state=seed
    )
    X_test, X_val, T_test, T_val, y_test, y_val = train_test_split(
        X_test_val, T_test_val, y_test_val, test_size=0.5, shuffle=False
    )

    # Generate the k-FP features
    train_set = kfp_features(X_train, T_train, y_train, seed)
    valid_set = kfp_features(X_val, T_val, y_val, seed)
    test_set = kfp_features(X_test, T_test, y_test, seed)

    # Open interactive shell for debugging
    # code.interact(local=locals())

    # Free up memory by deleting the raw `numpy` arrays after feature extraction
    del X, T, y
    del X_train, X_test_val, T_train, T_test_val, y_train, y_test_val
    del X_test, X_val, T_test, T_val, y_test, y_val

    # Train and evaluate Random Forest model for closed-world website fingerprinting
    RF_closedworld(
        train_set=train_set, valid_set=valid_set, test_set=test_set, seed=seed
    )
