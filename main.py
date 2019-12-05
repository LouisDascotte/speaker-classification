import numpy as np
import scipy.io.wavfile as wf
import scipy.signal as sp
import matplotlib.pyplot as plt
import math
import random
import glob
import scikit_talkbox_lpc
import filterbanks
from scipy.signal import lfilter
from scipy import fftpack


def normalise(signal):
    return signal/np.max(np.abs(signal))


def split(signal, width, shiftingstep, samplefreq):
    shift_n = int((shiftingstep/1000)*samplefreq)
    width_n = int((width/1000)*samplefreq)
    step = int(np.size(signal)/shift_n)
    frames = []
    for i in range(step):
        if((i*shift_n)+width_n > np.size(signal)):
            diff = ((i*shift_n)+width_n) - np.size(signal)
            frame = signal[i*shift_n:(i*shift_n)+width_n-diff]
            frames.append(frame)
        else:
            frame = signal[i*shift_n:(i*shift_n)+width_n]
            frames.append(frame)
    frames = np.array(frames)
    return frames


def find_distance(x, c):
    firstpoint = None
    secondpoint = None
    for i in (range(1, len(c)-1)):
        if((c[i] > c[i+1] and c[i] > c[i-1]) or (c[i] < c[i+1] and c[i] < c[i-1])):
            if(firstpoint == None):
                firstpoint = x[i]
            elif (secondpoint == None):
                secondpoint = x[i]
                break
    if(secondpoint == None or firstpoint == None):
        return None
    else:
        distance = secondpoint - firstpoint
        return distance


def autocorrelation(signal, samplefreq, fmin=50):
    n = math.ceil(samplefreq/fmin)
    tab = plt.xcorr(signal, signal,  False, maxlags=n)
    x = tab[0]
    c = tab[1]
    period = find_distance(x, c)
    if(period == None):
        return None
    else:
        return (1/period)*samplefreq


def get_energy(signal):
    return np.sum(np.square(signal))


def is_voiced(signal, threshold):
    return get_energy(signal) > threshold


def file_picker(speaker_name, n=5):
    path = glob.glob("cmu_us_"+speaker_name+"_arctic/wav/*.wav")
    files = []
    for i in random.sample(range(len(path)-1), n):
        files.append(path[i])
    return files


def find_treshhold(frames):
    treshhold = None
    energies = []
    for frame in frames:
        energies.append(get_energy(frame))
    for i in range(1, len(energies)-1):
        if(energies[i-1] < energies[i] and energies[i] > energies[i+1]):
            if(treshhold != None):
                if(energies[i] < treshhold):
                    treshhold = energies[i]
            else:
                treshhold = energies[i]
    return treshhold


def formants(signal, width, shiftingstep, samplefreq):
    frames = split(signal, width, shiftingstep, samplefreq)
    b, a = [1, 0.67], [1, 0]  # coefficients of the high pass filter
    filtedframes = []
    roots = []  # roots of each LPC
    for frame in frames:
        # put the frame inside the high pass filter
        filteredframe = lfilter(b, a, frame)
        hamming_w = np.hamming(len(filteredframe))  # obtain the hamming window
        # put the filtered frame in the hamming window corresponding
        filteredframe = hamming_w * filteredframe
        filtedframes.append(filteredframe)
        LPC = scikit_talkbox_lpc.lpc_ref(
            filteredframe, 9)  # get the lpc coefficients
        root = np.roots(LPC)  # obtain the roots with the lpc coeffcients
        # we take either the positives complexe or the negatives complexe (positive in this case)
        for i in range(len(root)):
            imag = np.imag(root[i])
            if(imag > 0):
                roots.append(root[i])
    # angles obtained from the roots
    angles = np.arctan2(np.imag(roots), np.real(roots))
    # frequences obtained from the angles
    freqs = sorted(angles*(samplefreq/(2*math.pi)))
    return freqs


def mfcc(signal,  width, shiftingstep, samplefreq, N_tfd=257):
    b, a = [1, 0.97], [1, 0]  # coefficients of the high pass filter
    pre_emphasized_signal = lfilter(b, a, signal)
    frames = split(pre_emphasized_signal, width, shiftingstep, samplefreq)
    filtedframes = []
    P_values = []
    for frame in frames:
        hamming_w = np.hamming(len(frame))
        filteredframe = hamming_w * frame
        filtedframes.append(filteredframe)
        power = (np.square(np.absolute(
            fftpack.fft(filteredframe, N_tfd))))/(2*N_tfd)
        P_values.append(power)
    filterbank = filterbanks.filter_banks(P_values, samplefreq)
    DCT = fftpack.dct(filterbank, type=2, axis=1, norm="ortho")
    return DCT[:13]


def rule_based(width=1000, shiftingstep=1000, sf=16000, sf2=1000, N_tfd=512):
    male = file_picker("bdl", 15)
    female = file_picker("slt", 15)
    sentences = male+female
    for sentence in sentences:
        datas = wf.read(sentence)
        signal = datas[1]
        normalized_signal = normalise(signal)
        frames = split(normalized_signal, width, shiftingstep, sf2)
        treshhold = find_treshhold(frames)
        pitches = []
        for frame in frames:
            if(is_voiced(frame, treshhold)):
                pitch = autocorrelation(frame, sf2)
                if(pitch != None):
                    pitches.append(pitch)
        f1 = formants(signal, width, shiftingstep, sf)[0]
        pitch = np.mean(np.array(pitches))

        if (pitch <= 200):
            print("[Pitch] "+sentence+" is a male voice.")
        else:
            print("[Pitch] "+sentence+" is a female voice.")

        if (f1 <= 500):
            print("[Formant] "+sentence+" is a male voice.")
        else:
            print("[Formant] "+sentence+" is a female voice.")


if __name__ == "__main__":
    rule_based()
