# Run with "ipython --matplotlib=qt receiver.py <file>.wav"
#
from __future__ import print_function
import struct
import sys
import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
from math import log10
from tones import TonesMonitor,TonesRecord,\
    note, printHeader, printFrame,\
    generateToneTemplate, DebugTonesFormat,\
    TONES

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SAMPLE_RATES = {
    # FrameRate (frames per second) need to allow all frames to be even in length to ensure the "corr" values are similar.
    #   - Originally FRAME_TIME was 0.1, gave 10 frames of 1220 with a final 11th frame of only 5... which gave 
    #     significantly smaller values for the eahc of the 11th frame processed.
    11025: {"descimate": 1, "sig_rate": 11025, "frame_rate": 9, "frame_len": 1225},
    22050: {"descimate": 2, "sig_rate": 11025, "frame_rate": 9, "frame_len": 1225},
    44100: {"descimate": 4, "sig_rate": 11025, "frame_rate": 9, "frame_len": 1225},
    48000: {"descimate": 4, "sig_rate": 12000, "frame_rate": 10, "frame_len": 1200},
}

# Shamelessly lifted from
# https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def getDecimate(sig_rate: int):
    if (sig_rate not in SAMPLE_RATES):
        print(f"Sample rate {sig_rate} not supported. Supported rates are (11025, 22050, 44100 or 48000).")
        sys.exit(-1)

    return SAMPLE_RATES[sig_rate]["descimate"]

def getNewSigRate(sig_rate: int):
    if (sig_rate not in SAMPLE_RATES):
        print(f"Sample rate {sig_rate} not supported. Supported rates are (11025, 22050, 44100 or 48000).")
        sys.exit(-1)

    return SAMPLE_RATES[sig_rate]["sig_rate"]

def getFrameLength(sig_rate: int):
    if (sig_rate not in SAMPLE_RATES):
        print(f"Sample rate {sig_rate} not supported. Supported rates are (11025, 22050, 44100 or 48000).")
        sys.exit(-1)

    return SAMPLE_RATES[sig_rate]["frame_len"]
        
def getFrameRate(sig_rate: int):
    if (sig_rate not in SAMPLE_RATES):
        print(f"Sample rate {sig_rate} not supported. Supported rates are (11025, 22050, 44100 or 48000).")
        sys.exit(-1)

    return SAMPLE_RATES[sig_rate]["frame_rate"]

# analyze wav file by chunks
def receiver(file_name):
    try:
        inp_sig_rate, sig_noise = read(file_name)
    except Exception:
        print('Error opening {}'.format(file_name))
        return

    print('file: ', file_name, ' rate: ', inp_sig_rate, ' len: ', len(sig_noise))

    decimate = getDecimate(inp_sig_rate)
    sig_rate = getNewSigRate(inp_sig_rate)
    
    if decimate > 1:
        sig_noise = signal.decimate(sig_noise, decimate)
        
    print(f"Decimated by: [{decimate}] to new SigRate: [{sig_rate}] with a length after decimation: [{len(sig_noise)}]")

    # Frames rate/length
    frame_len = getFrameLength(inp_sig_rate)
    frames = int(len(sig_noise) / frame_len)

    sig_noise = butter_bandpass_filter(sig_noise,
                                       270,
                                       1700,
                                       sig_rate,
                                       order=8)

    template = generateToneTemplate(frame_len, sig_rate)

    # See http://stackoverflow.com/questions/23507217/
    #         python-plotting-2d-data-on-to-3d-axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    y = np.arange(len(TONES))

    #pformat = DebugTonesFormat.DEBUG_TONES_MAX_ONLY
    pformat = DebugTonesFormat.DEBUG_TONES_MAX_AND_ABOVE_AVG
    printHeader(pformat)
    
    x = range(0, frames)
    X, Y = np.meshgrid(y, x)
    Z = np.zeros((len(x), len(y)))

    for frame in range(0, frames):

        beg = frame * frame_len
        end = (frame+1) * frame_len

        corr = np.zeros(len(TONES))

        for tone in range(0, len(TONES)):
            corr[tone] = log10(np.abs(signal.correlate(sig_noise[beg:end],
                                                       template[tone],
                                                       mode='same')).sum())
            Z[frame, tone] = corr[tone]

        tones_rec = TonesRecord(corr)

        printFrame(frame, tones_rec, pformat)

        print(f'', flush=True)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='w', shade=True,
                    lw=.5)
    # ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1000, lw=.5)

    ax.set_title(file_name)
    ax.set_xlabel("Tone")
    ax.set_ylabel("Frame")
    ax.set_zlabel("Log Correlation")

    ax.set_zlim(10.0, 15.0)
    ax.set_ylim(0, frames)

    ax.view_init(30, -130)

    plt.show()

if __name__ == "__main__":
    receiver(sys.argv[1])
