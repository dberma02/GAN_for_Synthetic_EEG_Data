import numpy as np
import pandas as pd
from scipy.io import loadmat

# Each matlab file saves data in an object 'o', which has the following fields:
#     id: A unique alphanumeric identifier of the record
#     nS: Number of EEG data samples
#     sampFreq: Sampling frequency of the EEG data
#     marker: The eGUI [interaction record of the recording session]
#             0 --> Nothing displayed
#             1 --> Left hand
#             2 --> Right hand
#             3 --> Passive / neutral
#     data: The Raw [EEG data of the recording session]

# Returns DF with EEG data for each channel, as well as column for marker
def load_patient(file_path, keep_channels=None):
    raw = loadmat(file_path)['o']
    # Make sure sampFreq = 200
    # 200 hz --> period of 0.005 seconds
    assert(raw['sampFreq'][0][0][0][0] == 200)
    marker = pd.Series(raw['marker'][0][0].flatten())
    chnames = [elem[0][0] for elem in raw['chnames'][0][0]]
#     print("CHANNEL NAMES: ",chnames)
    data = raw['data'][0][0]
    df = pd.DataFrame(data, columns=chnames)
    if keep_channels != None:
        df = df.loc[:,keep_channels]
    return df, marker

# returns array of same len as marker, labeling each entry with membership to a specific
# trial, with a number from 0 - # trials
def trial_membership(marker, interval_lens):
    i = 0
    start = 0
    membership = np.full(shape=marker.shape, fill_value=-1)
    for test_instance in interval_lens:
        end = start + test_instance
        membership[start:end] = i
        i += 1
        start = end
    return membership

# keep_channels is a list of channels to keep.
# trial_len is the length of trial in seconds at which trials should be trimmed.  If a trial
# is shorter then trial_len, then results will be zero padded at all values after conclusion of that trial.
# returns (data, labels) where data is 3d array shape(trials, num_chans, num_observations)
# and labels 1d array of length trials, each entry corresponding to the label of trial in data at that index
def get_data(file_path, trial_len, keep_channels=None):
    patient, marker = load_patient("../data/CLASubjectA1601083StLRHand.mat", keep_channels)
    num_chans = patient.shape[1]

    # convert trial_len to num obvs at 200hz
    trim_len = int(trial_len / 0.005)
    
    # diff_mask is true at start of every new test condition
    diff_mask = (marker.diff(1) != 0)
    diffs = marker[diff_mask]

    start_indexes = pd.Series(diffs.index)
    interval_lens = start_indexes.diff(1)[1:].astype(int)
    # Add last interval to end
    interval_lens = interval_lens.append(pd.Series(len(marker) - start_indexes.iloc[-1]))

    # Because at beginning, long period of 0 --> Nothing displayed
    # so want everything to start with beginning of first trial
    labels = diffs.values[1:]
    marker = marker.values[interval_lens.values[0]:]
    patient = patient[interval_lens.values[0]:]

    interval_lens = interval_lens.values[1:]

    # Because each trial has image displayed on screen for ~1 second,
    # and then a few seconds of blank screen where the patient is still
    # conducting the task, we want to combine the period where image shown
    # and the following blank screen
    non_zero_int_lens = interval_lens[labels != 0]
    zero_int_lens = interval_lens[labels == 0]
    labels = labels[labels != 0]
    interval_lens = non_zero_int_lens + zero_int_lens

    assert(labels.shape[0] == interval_lens.shape[0])

    membership = trial_membership(marker, interval_lens)
    
    # Format results as 3d array shape(trials, num_chans, num_observations)
    results = np.zeros(shape=(len(interval_lens), num_chans, trim_len))
    for i in range(0,len(interval_lens)):
        trial = patient[membership == i].values
        if trial.shape[0] > trim_len:
            trial = trial[:trim_len,:]
        results[i] = results[i] + trial.transpose()

    return (results, labels)

#  return ERP from the specified window after stimulus is viewed
def trim_intervals(X, offset, new_interval_len):
    frequency = 200
    conv_factor = 1/200
    # Convert from seconds to step at frequency
    offset = int(offset / conv_factor)
    new_interval_len = int(new_interval_len / conv_factor)
    orig_interval_len = X.shape[2]
    if orig_interval_len < offset + new_interval_len:
        raise ValueError("offset + new interval len must be less than len of original interval")
    return X[:,:,offset : offset + new_interval_len]
