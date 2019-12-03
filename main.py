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
        if(c[i] > c[i+1] and c[i] > c[i-1]):
            if(firstpoint == None):
                firstpoint = x[i]
            elif (secondpoint == None):
                secondpoint = x[i]
                break
    if(secondpoint == None or firstpoint ==None):
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
    dcts = []
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


def rule_based(width=1000, shiftingstep=1000, sf=1000, N_tfd=257):
    male = file_picker("bdl", 15)
    female = file_picker("slt", 15)
    sentences = []
    for element in female:
        sentences.append(element)
    for element in male:
        sentences.append(element)
    random.shuffle(sentences)
    for sentence in sentences:
        datas = wf.read(sentence)
        signal = datas[1]
        frames = split(signal, width, shiftingstep, sf)
        treshhold = find_treshhold(frames)
        pitches = []
        for frame in frames:
            if(is_voiced(frame, treshhold)):
                pitch = autocorrelation(frame, sf)
                if(pitch != None):
                    #print("pitch : " + str(pitch))
                    pitches.append(autocorrelation(frame, sf))
        #f1 = formants(signal, width, shiftingstep, sf)[0]
        #print("f1 : "+ str(f1))
        pitch = np.mean(np.array(pitches))
        print("pitch : " + str(pitch))
        if(pitch>350 ):
            print("the wav file : " + sentence+" is a female voice")
        else:
            print("the wav file : " + sentence+" is a male voice")




rule_based()
#files = file_picker("slt")
#datas = wf.read(files[0])
#test(datas[1])
#freqz = formants(datas[1], 1000, 1000, 1000)
#P_values = mfcc(datas[1], 1000, 1000, 1000)
#print(P_values)
# audio = np.array([1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76], dtype=np.float)
# normalized_signal = normalise(audio)
# tab = split(normalized_signal, 1000, 1000, 200)
# print(tab)
# f0 = autocorrelation(tab[0], 200)
# print(f0)
