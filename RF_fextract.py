import code
import math
import random
from itertools import groupby, chain
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

# Global variables for sharing X and T.
global_X = None
global_T = None

TIME_FEATURES = True
NUMBER_FEATURES = True
SIZE_FEATURES = True
ALTERNATE_FEATURES = True and NUMBER_FEATURES

CHUNK_NUM_ALT_CONC = 70
CHUNK_NUM_ALT_PER_SEC = 20

ORIGINAL_FEATURES_MAX_SIZE = 175

FEATURES_MAX_SIZE = 0
if TIME_FEATURES:
    FEATURES_MAX_SIZE += 26 - 4 - 4
if NUMBER_FEATURES:
    FEATURES_MAX_SIZE += 24 - 1
if SIZE_FEATURES:
    FEATURES_MAX_SIZE += 14 - 4
if ALTERNATE_FEATURES:
    FEATURES_MAX_SIZE += CHUNK_NUM_ALT_CONC
    FEATURES_MAX_SIZE += CHUNK_NUM_ALT_PER_SEC
    FEATURES_MAX_SIZE += 2    # sum(alt_conc), sum(alt_per_sec)
    FEATURES_MAX_SIZE += 20   # conc, per_sec


def init_worker(X_shared, T_shared):
    """
    This initializer is called once per worker process.
    It sets the global X and T arrays so that each worker can access them.
    """
    global global_X, global_T
    global_X = X_shared
    global_T = T_shared


def process_instance(args):
    """
    Processes one instance (a row in X and T).
    Finds the first zero in the row, prepares the list_data, and computes features.
    """
    idx, site, instance_no = args
    global global_X, global_T

    row = global_X[idx]
    # Find the first occurrence of zero in the row.
    zero_cells = np.where(row == 0)[0]
    if zero_cells.size > 0:
        last_cell_index = int(zero_cells[0])
    else:
        last_cell_index = row.shape[0]

    # Create list_data
    list_data = list(
        zip(global_T[idx][:last_cell_index], row[:last_cell_index].astype(int))
    )
    features = TOTAL_FEATURES(list_data)
    return ([features], (int(site), instance_no))


def chunks(labels, features):
    """
    Groups the labels and features by the first element of the label tuple.
    This helper assumes the labels are already sorted by site.
    """
    grouped = groupby(zip(labels, features), key=lambda x: x[0][0])
    labels_chunked, features_chunked = [], []
    for _, group in grouped:
        lbls, feats = zip(*group)
        labels_chunked.append(list(lbls))
        features_chunked.append(list(feats))
    return labels_chunked, features_chunked


def kfp_features(X, T, y, seed=None):
    """
    Parallelized version of the kfp_features function using multiprocessing.Pool.
    X and T are large numpy arrays that are shared with worker processes.
    """
    data_dict = {"feature": [], "label": []}
    unique_sites = np.unique(y)

    # Precompute instance indices for each site.
    site_to_indices = {site: np.flatnonzero(y == site) for site in unique_sites}
    tasks = []
    for site in unique_sites:
        indices = site_to_indices[site]
        for instance_no, idx in enumerate(indices):
            tasks.append((idx, site, instance_no))

    # Use a multiprocessing Pool with an initializer to share X and T.
    # Using imap with tqdm for a progress bar.
    with mp.Pool(initializer=init_worker, initargs=(X, T)) as pool:
        # If a seed is set, we need to use ordered imap to make sure
        # the results order is always predictable
        if seed != None:
            pbar = tqdm(
                pool.imap(process_instance, tasks),
                total=len(tasks),
                desc="Generating k-FP Features",
            )
        else:
            pbar = tqdm(
                pool.imap_unordered(process_instance, tasks),
                total=len(tasks),
                desc="Generating k-FP Features",
            )

        results = list(pbar)

    # Sequential processing.
    # global global_X, global_T
    # global_X, global_T = X, T

    # results = []
    # for task in tqdm(tasks, desc="[SEQUENTIAL] Generating k-FP Features"):
    #     results.append(process_instance(task))

    # Collect results.
    for feat, lab in results:
        data_dict["feature"].append(feat)
        data_dict["label"].append(lab)

    # Group data by site.
    split_target, split_data = chunks(data_dict["label"], data_dict["feature"])
    set_data = []
    set_label = []
    for lbl_chunk, feat_chunk in zip(split_target, split_data):
        temp = list(zip(feat_chunk, lbl_chunk))
        random.shuffle(temp)
        data, label = zip(*temp)
        set_data.extend(data)
        set_label.extend(label)

    # Flatten the feature lists.
    flat_set_data = [list(chain.from_iterable(data)) for data in set_data]
    set_features = list(zip(flat_set_data, set_label))
    return set_features


"""Feeder functions"""


def neighborhood(iterable):
    iterator = iter(iterable)
    prev = 0
    item = iterator.__next__()  # throws StopIteration if empty.
    for next in iterator:
        yield (prev, item, next)
        prev = item
        item = next
    yield (prev, item, None)


def chunkIt(seq, num):
    """
    Usually returns `num` chunks but can return `num + 1` chunks when
    `len(seq)` is not divisible by `num`.

    >>> len(chunkIt([x for x in range(13)], 4))
    4
    >>> len(chunkIt([x for x in range(13)], 5))
    5
    >>> len(chunkIt([x for x in range(13)], 6))
    7
    >>> chunkIt([x for x in range(13)], 6)
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12]]
    >>> 13 / 6
    2.1666666666666665
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg
    return out


"""Non-feeder functions"""


def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        direction = int(np.sign(p[1]))
        if direction == -1:
            In.append(p)
        elif direction == 1:
            Out.append(p)
    return In, Out


############### TIME FEATURES #####################


def inter_pkt_time(list_data):
    times = [x[0] for x in list_data]
    temp = []
    for elem, next_elem in list(zip(times, times[1:] + [times[0]])):
        temp.append(next_elem - elem)
    return temp[:-1]


def interarrival_times(list_data):
    In, Out = In_Out(list_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(list_data)
    return IN, OUT, TOTAL


def interarrival_maxminmeansd_stats(list_data):
    interstats = []
    interstats_labels = [
        "interarrival_times_max_in",
        "interarrival_times_max_out",
        "interarrival_times_max_total",
        "interarrival_times_avg_in",
        "interarrival_times_avg_out",
        "interarrival_times_avg_total",
        "interarrival_times_std_in",
        "interarrival_times_std_out",
        "interarrival_times_std_total",
        "interarrival_times_75th_percentile_in",
        "interarrival_times_75th_percentile_out",
        "interarrival_times_75th_percentile_total",
    ]
    In, Out, Total = interarrival_times(list_data)
    if In and Out:
        avg_in = sum(In) / float(len(In))
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.extend(
            [
                max(In),
                max(Out),
                # max(Total),
                avg_in,
                avg_out,
                # avg_total,
                np.std(In),
                np.std(Out),
                # np.std(Total),
                np.percentile(In, 75),
                np.percentile(Out, 75),
                # np.percentile(Total, 75),
            ]
        )
    elif Out and not In:
        avg_out = sum(Out) / float(len(Out))
        avg_total = sum(Total) / float(len(Total))
        interstats.extend(
            [
                0,
                max(Out),
                # max(Total),
                0,
                avg_out,
                # avg_total,
                0,
                np.std(Out),
                # np.std(Total),
                0,
                np.percentile(Out, 75),
                # np.percentile(Total, 75),
            ]
        )
    elif In and not Out:
        avg_in = sum(In) / float(len(In))
        avg_total = sum(Total) / float(len(Total))
        interstats.extend(
            [
                max(In),
                0,
                # max(Total),
                avg_in,
                0,
                # avg_total,
                np.std(In),
                0,
                # np.std(Total),
                np.percentile(In, 75),
                0,
                # np.percentile(Total, 75),
            ]
        )
    else:
        interstats.extend([0] * 12)  # 12 features, not 15
    return interstats


def time_percentile_stats(Total):
    In, Out = In_Out(Total)
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in Total]
    stats = []
    stats_labels = [
        "time_25th_percentile_in",
        "time_50th_percentile_in",
        "time_75th_percentile_in",
        "time_100th_percentile_in",
        "time_25th_percentile_out",
        "time_50th_percentile_out",
        "time_75th_percentile_out",
        "time_100th_percentile_out",
        "time_25th_percentile_total",
        "time_50th_percentile_total",
        "time_75th_percentile_total",
        "time_100th_percentile_total",
    ]
    if In1:
        stats.append(np.percentile(In1, 25))  # return 25th percentile
        stats.append(np.percentile(In1, 50))
        stats.append(np.percentile(In1, 75))
        stats.append(np.percentile(In1, 100))
    if not In1:
        stats.extend(([0] * 4))
    if Out1:
        stats.append(np.percentile(Out1, 25))  # return 25th percentile
        stats.append(np.percentile(Out1, 50))
        stats.append(np.percentile(Out1, 75))
        stats.append(np.percentile(Out1, 100))
    if not Out1:
        stats.extend(([0] * 4))
    # if Total1:
    #     stats.append(np.percentile(Total1, 25))  # return 25th percentile
    #     stats.append(np.percentile(Total1, 50))
    #     stats.append(np.percentile(Total1, 75))
    #     stats.append(np.percentile(Total1, 100))
    # if not Total1:
    #     stats.extend(([0] * 4))
    return stats


def number_pkt_stats(Total):
    In, Out = In_Out(Total)
    stats = [
        len(In),
        len(Out),
        # len(Total)
    ]
    stats_labels = [
        "number_packets_in",
        "number_packets_out",
        "number_packets_total",
    ]
    return stats


def first_and_last_30_pkts_stats(Total):
    first30 = Total[:30]
    last30 = Total[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[1] == -1:
            first30in.append(p)
        if p[1] == 1:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[1] == -1:
            last30in.append(p)
        if p[1] == 1:
            last30out.append(p)
    stats = []
    stats_labels = [
        "num_first_30_packets_in",
        "num_first_30_packets_out",
        "num_last_30_packets_in",
        "num_last_30_packets_out",
    ]
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats


# Concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(Total):
    chunks = [Total[x : x + 20] for x in range(0, len(Total), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] == 1:
                c += 1
        concentrations.append(c)
    stats = [
        sum(concentrations) / float(len(concentrations)),
        np.std(concentrations),
        np.percentile(concentrations, 50),
        min(concentrations),
        max(concentrations),
        concentrations,
    ]
    stats_labels = [
        "packet_concentration_avg",
        "packet_concentration_std",
        "packet_concentration_median",
        "packet_concentration_min",
        "packet_concentration_max",
        "packet_concentration_list",
    ]
    return stats


# Average number packets sent and received per second
def number_per_sec(Total):
    last_time = Total[-1][0]
    last_second = math.ceil(last_time)
    temp = []
    l = []
    for i in range(1, int(last_second) + 1):
        c = 0
        for p in Total:
            if p[0] <= i:
                c += 1
        temp.append(c)
    for prev, item, next in neighborhood(temp):
        x = item - prev
        l.append(x)
    avg_number_per_sec = sum(l) / float(len(l))
    stats = [avg_number_per_sec, np.std(l), np.percentile(l, 50), min(l), max(l), l]
    stats_labels = [
        "packets_per_second_avg",
        "packets_per_second_std",
        "packets_per_second_median",
        "packets_per_second_min",
        "packets_per_second_max",
        "packets_per_second_list",
    ]
    return stats


# Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
def avg_pkt_ordering_stats(Total):
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] == 1:  # outgoing
            temp1.append(c1)
        c1 += 1
        if p[1] == -1:  # incoming
            temp2.append(c2)
        c2 += 1
    avg_in = sum(temp1) / float(len(temp1))
    avg_out = sum(temp2) / float(len(temp2))
    stats = [avg_in, avg_out, np.std(temp1), np.std(temp2)]
    stats_labels = [
        "packet_ordering_in_avg",
        "packet_ordering_out_avg",
        "packet_ordering_in_std",
        "packet_ordering_out_std",
    ]
    return stats


def perc_inc_out(Total):
    In, Out = In_Out(Total)
    percentage_in = len(In) / float(len(Total))
    percentage_out = len(Out) / float(len(Total))
    stats = [percentage_in, percentage_out]
    stats_labels = [
        "percentage_in",
        "percentage_out",
    ]
    return stats


############### SIZE FEATURES #####################


def total_size(list_data):
    stat = sum([abs(x[1]) for x in list_data])
    stat_label = "packet_size_sum"
    return stat


def in_out_size(list_data):
    In, Out = In_Out(list_data)
    size_in = sum([abs(x[1]) for x in In])
    size_out = sum([abs(x[1]) for x in Out])
    stats = [size_in, size_out]
    stats_labels = [
        "packet_size_sum_in",
        "packet_size_sum_out",
    ]
    return stats


def average_total_pkt_size(list_data):
    stat = np.mean([abs(x[1]) for x in list_data])
    stat_label = "packet_size_avg"
    return stat


def average_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    average_size_in = np.mean([abs(x[1]) for x in In])
    average_size_out = np.mean([abs(x[1]) for x in Out])
    stats = [average_size_in, average_size_out]
    stats_labels = [
        "packet_size_avg_in",
        "packet_size_avg_out",
    ]
    return stats


def variance_total_pkt_size(list_data):
    stat = np.var([abs(x[1]) for x in list_data])
    stat_label = "packet_size_var"
    return stat


def variance_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    var_size_in = np.var([abs(x[1]) for x in In])
    var_size_out = np.var([abs(x[1]) for x in Out])
    stats = [var_size_in, var_size_out]
    stats_labels = [
        "packet_size_var_in",
        "packet_size_var_out",
    ]
    return stats


def std_total_pkt_size(list_data):
    stat = np.std([abs(x[1]) for x in list_data])
    stat_label = "packet_size_std"
    return stat


def std_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    std_size_in = np.std([abs(x[1]) for x in In])
    std_size_out = np.std([abs(x[1]) for x in Out])
    stats = [std_size_in, std_size_out]
    stats_labels = [
        "packet_size_std_in",
        "packet_size_std_out",
    ]
    return stats


def max_in_out_pkt_size(list_data):
    In, Out = In_Out(list_data)
    max_size_in = max([abs(x[1]) for x in In])
    max_size_out = max([abs(x[1]) for x in Out])
    stats = [max_size_in, max_size_out]
    stats_labels = [
        "packet_size_max_in",
        "packet_size_max_out",
    ]
    return stats


def unique_pkt_lengths(list_data):
    return list(set([abs(x[1]) for x in list_data]))


############### FEATURE FUNCTION #####################


# If size information available add them in to function below
def TOTAL_FEATURES(list_data, time_features=TIME_FEATURES, number_features=NUMBER_FEATURES, size_features=SIZE_FEATURES, alternate_features=ALTERNATE_FEATURES, max_size=FEATURES_MAX_SIZE):
    ALL_FEATURES = []

    list_size_data      = list_data
    list_direction_data = [(time, np.sign(size)) for (time, size) in list_data]

    if time_features:
        intertimestats = interarrival_maxminmeansd_stats(list_direction_data)
        timestats = time_percentile_stats(list_direction_data)

        ALL_FEATURES.extend(intertimestats)
        ALL_FEATURES.extend(timestats)

        ALL_FEATURES.append(sum(intertimestats))
        ALL_FEATURES.append(sum(timestats))

    if number_features:
        number_pkts = number_pkt_stats(list_direction_data)
        thirty_pkts = first_and_last_30_pkts_stats(list_direction_data)
        [avg_conc, std_conc, med_conc, min_conc, max_conc, conc] = (
            pkt_concentration_stats(list_direction_data)
        )
        [avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec] = (
            number_per_sec(list_direction_data)
        )
        [avg_order_in, avg_order_out, std_order_in, std_order_out] = (
            avg_pkt_ordering_stats(list_direction_data)
        )
        [perc_in, perc_out] = perc_inc_out(list_direction_data)

        ALL_FEATURES.extend(number_pkts)
        ALL_FEATURES.append(sum(number_pkts))

        ALL_FEATURES.extend(thirty_pkts)

        # pkt_concentration_stats()
        ALL_FEATURES.append(avg_conc)
        ALL_FEATURES.append(std_conc)
        ALL_FEATURES.append(med_conc)
        ALL_FEATURES.append(min_conc)
        ALL_FEATURES.append(max_conc)

        # number_per_sec()
        ALL_FEATURES.append(avg_per_sec)
        ALL_FEATURES.append(std_per_sec)
        ALL_FEATURES.append(med_per_sec)
        ALL_FEATURES.append(min_per_sec)
        ALL_FEATURES.append(max_per_sec)

        # avg_pkt_ordering_stats()
        ALL_FEATURES.append(avg_order_in)
        ALL_FEATURES.append(avg_order_out)
        ALL_FEATURES.append(std_order_in)
        ALL_FEATURES.append(std_order_out)

        # perc_inc_out()
        ALL_FEATURES.append(perc_in)
        ALL_FEATURES.append(perc_out)

    if alternate_features:
        alt_conc = [sum(x) for x in chunkIt(conc, CHUNK_NUM_ALT_CONC)]
        alt_per_sec = [sum(x) for x in chunkIt(per_sec, CHUNK_NUM_ALT_PER_SEC)]
        if len(alt_conc) == CHUNK_NUM_ALT_CONC:
            alt_conc.append(0)
        if len(alt_per_sec) == CHUNK_NUM_ALT_PER_SEC:
            alt_per_sec.append(0)

        ALL_FEATURES.extend(alt_conc)
        ALL_FEATURES.extend(alt_per_sec)

        ALL_FEATURES.append(sum(alt_conc))
        ALL_FEATURES.append(sum(alt_per_sec))

    # ------SIZE--------

    if size_features:
        tot_size = total_size(list_size_data)
        [in_size, out_size] = in_out_size(list_size_data)
        avg_total_size = average_total_pkt_size(list_size_data)
        [avg_size_in, avg_size_out] = average_in_out_pkt_size(list_size_data)
        var_total_size = variance_total_pkt_size(list_size_data)
        [var_size_in, var_size_out] = variance_in_out_pkt_size(list_size_data)
        std_total_size = std_total_pkt_size(list_size_data)
        [std_size_in, std_size_out] = std_in_out_pkt_size(list_size_data)
        [max_size_in, max_size_out] = max_in_out_pkt_size(list_size_data)

        # total_size()
        # ALL_FEATURES.append(tot_size)

        # in_out_size()
        ALL_FEATURES.append(in_size)
        ALL_FEATURES.append(out_size)

        # average_total_pkt_size()
        # ALL_FEATURES.append(avg_total_size)

        # average_in_out_pkt_size()
        ALL_FEATURES.append(avg_size_in)
        ALL_FEATURES.append(avg_size_out)

        # variance_total_pkt_size()
        # ALL_FEATURES.append(var_total_size)

        # variance_in_out_pkt_size()
        ALL_FEATURES.append(var_size_in)
        ALL_FEATURES.append(var_size_out)

        # std_total_pkt_size()
        # ALL_FEATURES.append(std_total_size)

        # std_in_out_pkt_size()
        ALL_FEATURES.append(std_size_in)
        ALL_FEATURES.append(std_size_out)

        # max_in_out_pkt_size()
        ALL_FEATURES.append(max_size_in)
        ALL_FEATURES.append(max_size_out)

    if alternate_features:
        # This is optional, since all other features are of equal size this gives the first n features
        # of this particular feature subset, some may be padded with 0's if too short.
        ALL_FEATURES.extend(conc)
        ALL_FEATURES.extend(per_sec)

    while len(ALL_FEATURES) < max_size:
        ALL_FEATURES.append(0)
    features = ALL_FEATURES[:max_size]

    return tuple(features)


def kfp_feature_labels(time_features=TIME_FEATURES, number_features=NUMBER_FEATURES, size_features=SIZE_FEATURES, alternate_features=ALTERNATE_FEATURES):
    labels = []

    if time_features:
        labels += [
            # interarrival_maxminmeansd_stats() × 12-4
            "interarrival_times_max_in",
            "interarrival_times_max_out",
            # "interarrival_times_max_total",
            "interarrival_times_avg_in",
            "interarrival_times_avg_out",
            # "interarrival_times_avg_total",
            "interarrival_times_std_in",
            "interarrival_times_std_out",
            # "interarrival_times_std_total",
            "interarrival_times_75th_percentile_in",
            "interarrival_times_75th_percentile_out",
            # "interarrival_times_75th_percentile_total",

            # time_percentile_stats() × 12-4
            "time_25th_percentile_in",
            "time_50th_percentile_in",
            "time_75th_percentile_in",
            "time_100th_percentile_in",
            "time_25th_percentile_out",
            "time_50th_percentile_out",
            "time_75th_percentile_out",
            "time_100th_percentile_out",
            # "time_25th_percentile_total",
            # "time_50th_percentile_total",
            # "time_75th_percentile_total",
            # "time_100th_percentile_total",

            "sum_interarrival_times",
            "sum_time",
        ]

    if number_features:
        labels += [
            # number_pkt_stats() × 3-1
            "number_packets_in",
            "number_packets_out",
            # "number_packets_total",

            "sum_number_packets",

            # first_and_last_30_pkts_stats() × 4
            "num_first_30_packets_in",
            "num_first_30_packets_out",
            "num_last_30_packets_in",
            "num_last_30_packets_out",

            # pkt_concentration_stats() × 5
            "packet_concentration_avg",
            "packet_concentration_std",
            "packet_concentration_median",
            "packet_concentration_min",
            "packet_concentration_max",

            # number_per_sec() × 5
            "packets_per_second_avg",
            "packets_per_second_std",
            "packets_per_second_median",
            "packets_per_second_min",
            "packets_per_second_max",

            # avg_pkt_ordering_stats() × 4
            "packet_ordering_in_avg",
            "packet_ordering_out_avg",
            "packet_ordering_in_std",
            "packet_ordering_out_std",

            # perc_inc_out() × 2
            "percentage_in",
            "percentage_out",
        ]

    if alternate_features:
        labels.extend(
            [f"alt_packet_concentration_{i}" for i in range(CHUNK_NUM_ALT_CONC)]
        )
        labels.extend(
            [f"alt_packets_per_second_{i}" for i in range(CHUNK_NUM_ALT_PER_SEC)]
        )

        labels += [
            "sum_alt_packet_concentration",
            "sum_alt_packets_per_second",
        ]

    if size_features:
        labels += [
            # total_size() × 1-1
            # "packet_size_sum",

            # in_out_size() × 2
            "packet_size_sum_in",
            "packet_size_sum_out",

            # average_total_pkt_size() × 1-1
            # "packet_size_avg",

            # average_in_out_pkt_size() × 2
            "packet_size_avg_in",
            "packet_size_avg_out",

            # variance_total_pkt_size() × 1-1
            # "packet_size_var",

            # variance_in_out_pkt_size() × 2
            "packet_size_var_in",
            "packet_size_var_out",

            # std_total_pkt_size() × 1-1
            # "packet_size_std",

            # std_in_out_pkt_size() × 2
            "packet_size_std_in",
            "packet_size_std_out",

            # max_in_out_pkt_size() × 2
            "packet_size_max_in",
            "packet_size_max_out",
        ]

    fixed_features_size = len(labels)

    labels.extend(
        [f"features_{i}" for i in range(FEATURES_MAX_SIZE - fixed_features_size)]
    )

    return labels


if __name__ == "__main__":
    pass
