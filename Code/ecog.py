import pyabf
import mne
import matplotlib.pyplot as plt
import numpy as np
import pathlib

def import_ecog(filename):
    abf = pyabf.ABF(filename)
    n_channels = abf.channelCount
    sampling_rate = abf.dataRate
    channel_names = ['F1', 'F2', 'STI']
    channel_types = ['eeg', 'eeg', 'stim']
    info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=channel_types)
    events = [[x, 0, x+1] for x in range(abf.sweepCount)]
    tmin = 0
    data = np.empty([np.size(abf.sweepList), 3, np.size(abf.sweepX)])
    for isweep in abf.sweepList:
        abf.setSweep(isweep, channel=1)
        data[isweep, 0, :] = abf.sweepY
        abf.setSweep(isweep, channel=2)
        data[isweep, 1, :] = abf.sweepY
        abf.setSweep(isweep, channel=0)
        data[isweep, 2, :] = abf.sweepY
    abf_epochs = mne.EpochsArray(data*1e-6, info, events, tmin)
    abf_epochs.set_montage('standard_1020')
    return abf_epochs

def calc_itpc(data):
    freqs = np.linspace(20, 90, 71)
    n_cycles = np.logspace(*np.log10([7, 30]), 71)
    power, itc = mne.time_frequency.tfr_morlet(data, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, n_jobs=1)
    return power, itc

def plot_tf(tf, savename='test.png'):
    tf.plot(
        #  mode="ratio",
        #  baseline=(0.2, 0.7),
        show=False,
        picks=("all"),
        combine="mean",
        cmap='viridis'
    )
    plt.savefig(savename)
    plt.close(plt.gcf())

if __name__ == "__main__":
   #  data_dir=pathlib.Path('../Data/ASSR_Mouse_Awake/2021_03_29/')
   data_dir=pathlib.Path('./')
   for ifile in data_dir.rglob('*abf'):
        filename = pathlib.Path.joinpath(ifile.parent, ifile.stem)
        data = import_ecog(str(ifile))
        data.average().plot();
        plt.savefig(f'{filename}-erp.png')
        plt.close(plt.gcf())
        data.copy().plot_psd(fmin=2, fmax=60);
        plt.savefig(f'{filename}-psd.png')
        plt.close(plt.gcf())
        power, itc = calc_itpc(data)
        plot_tf(itc, savename=f'{filename}-itpc.png')
        #  plot_tf(power, savename=f'{filename}-power.png')
        #  data.save(f'{filename}-epo.fif', overwrite='True')
