import logging
import time
from copy import copy
import sys

import numpy as np
import scipy
from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)

from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor
from sklearn.metrics import roc_auc_score

import logging
import re
import numpy as np
import glob
import os.path 

import mne
log = logging.getLogger(__name__)
log.setLevel('DEBUG')
# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications
data_folders = [
    'v2.0.0/normal',
    'v2.0.0/abnormal']
n_recordings = None  # set to an integer, if you want to restrict the set size
sensor_types = ["EEG"]
n_chans = 21
max_recording_mins = None # exclude larger recordings from training set
sec_to_cut = 60  # cut away at start of each recording
duration_recording_mins = 20  # how many minutes to use per recording#20earlier
test_recording_mins = 20#20earlier
max_abs_val = 800  # for clipping
sampling_freq = 100
divisor = 1  # divide signal by this######################## 10 earlier
test_on_eval = True  # test on evaluation set or on training set
# in case of test on eval, n_folds and i_testfold determine
# validation fold in training set for training until first stop
n_folds = 10
i_test_fold = 9
shuffle = True
model_name = 'deep'#shallow/deep for DNN (deep terminal local 1)
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 6000
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 1e-3
batch_size = 64
max_epochs = 35 # until first stop, the continue train on train+valid
cuda = False
def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        if True:
            edf_file = mne.io.read_raw_edf(file_path, montage = None, eog = ['FP1', 'FP2', 'F3', 'F4',
                                                                             'C3', 'C4',  'P3', 'P4','O1', 'O2','F7', 'F8',
                                                                             'T3', 'T4', 'T5', 'T6','PZ','FZ', 'CZ','A1', 'A2'], verbose='error')
        else:
            edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
        # fix_header(file_path)
        # try:
        #     edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        #     logging.warning("Fixed it!")
        # except ValueError:
        #     return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration


def load_data(fname, sensor_types=['EEG']):

    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:, int(sec_to_cut * fs):-int(
            sec_to_cut * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (data[:, :int(test_recording_mins * 60 * fs)], fs))
    if max_abs_val is not None:
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val), fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))

    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(fname)
    log.info("Load data..."+fname)
    ##edit to get on gpu device
    #torch.cuda.set_device(1)
    #print("--------------------------------" + torch.cuda.get_device_name(1))

    cnt.load_data()
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                        'FP2', 'FZ', 'O1', 'O2',
                        'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                if ((' ' + wanted_part + '-' in ch_name) or (wanted_part == ch_name)):#if ' ' + wanted_part + '-' in ch_name:
                    wanted_found_name.append(ch_name)
            print(wanted_found_name)####Comment out
            assert len(wanted_found_name) == 1
            selected_ch_names.append(wanted_found_name[0])
    if 'EKG' in sensor_types:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if 'EKG' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.pick_channels(selected_ch_names)

    #assert np.array_equal(cnt.ch_names, selected_ch_names)
    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 21
    if 'EKG' in sensor_types:
        n_sensors += 1

    assert len(cnt.ch_names)  == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    # change from volt to mikrovolt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.info("Preprocessing...")
    if data.shape[1] < 120000:
        return None
    for fn in preproc_functions:
        log.info(fn)
        print(data.shape)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)

    return data
def get_model(input_file):
    n_classes = 2
    model = Deep4Net(n_chans, n_classes,
                            n_filters_time=n_start_chans,
                            n_filters_spat=n_start_chans,
                            input_time_length=input_time_length,
                            n_filters_2 = int(n_start_chans * n_chan_factor),
                            n_filters_3 = int(n_start_chans * (n_chan_factor ** 2.0)),
                            n_filters_4 = int(n_start_chans * (n_chan_factor ** 3.0)),
                            final_conv_length=final_conv_length,
                            stride_before_pool=True).create_network()
    # model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
    #                                 n_filters_time=n_start_chans,
    #                                 n_filters_spat=n_start_chans,
    #                                 input_time_length=input_time_length,
    #                                 final_conv_length=final_conv_length).create_network()


    # device = th.device('cuda:1')
    state_dict = th.load('deep.pt',map_location='cpu')
    model.load_state_dict(state_dict)
    # model.to(device)
    model.eval()

    fname = input_file
    X = []
    X.append(load_data(fname))
    y = np.array([1])

    test_set = SignalAndTarget(X, y)


    if cuda:
        model.cuda()
    # determine output size
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    log.info("In shape: {:s}".format(str(test_input.cpu().data.numpy().shape)))

    out = model(test_input)
    log.info("Out shape: {:s}".format(str(out.cpu().data.numpy().shape)))
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    log.info("{:d} predictions per input/trial".format(n_preds_per_input))
    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                    input_time_length=input_time_length,
                                    n_preds_per_input=n_preds_per_input)

    if cuda:
        preds_per_batch = [var_to_np(model(np_to_var(b[0]).cuda()))
                        for b in iterator.get_batches(test_set, shuffle=False)]
    else:
        preds_per_batch = [var_to_np(model(np_to_var(b[0])))
                        for b in iterator.get_batches(test_set, shuffle=False)]
    preds_per_trial = compute_preds_per_trial(
        preds_per_batch, test_set,
        input_time_length=iterator.input_time_length,
        n_stride=iterator.n_preds_per_input)
    mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                            preds_per_trial]
    mean_preds_per_trial = np.array(mean_preds_per_trial)
    print(mean_preds_per_trial)

    pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
    print(pred_labels_per_trial)
    if pred_labels_per_trial == [1]: #predict EEG is abnormal

        return 1
    if pred_labels_per_trial == [0]: #predict EEG is normal
        return 0