import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import math, random, glob

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
        if(c[i]>c[i+1] and c[i] > c[i-1]):
            if(firstpoint == None):
                firstpoint = x[i]
            elif (secondpoint == None):
                secondpoint = x[i]
                break
    distance = secondpoint - firstpoint
    return distance  

def autocorrelation(signal, samplefreq, fmin=50):
    n = math.ceil(samplefreq/fmin)
    tab = plt.xcorr(signal, signal,  False, maxlags=n)
    x = tab[0]
    c = tab[1]
    print(c)
    period = find_distance(x, c)
    return 1/period

def get_energy(signal):
    return np.sum(np.square(signal))

def is_voiced(signal, threshold):
    return get_energy(signal) > threshold

def file_picker(speaker_name):
    path = glob.glob("cmu_us_"+speaker_name+"_arctic/wav/*.wav")
    files = []
    for i in random.sample(range(len(path)-1), 5):
        files.append(path[i])
    return files

def find_treshhold(energies):
    treshhold = None
    for i in range(1, len(energies)-1):
        if(energies[i-1] < energies[i] and energies[i]> energies[i+1]):
            if(treshhold!=None):
                if(energies[i]< treshhold):
                    treshhold = energies[i]
            else:
                treshhold = energies[i]
    return treshhold

def analyze(filename):
    datas = wf.read(filename)
    signal = datas[1]
    normalized_signal = normalise(signal)
    frames = split(normalized_signal, 300, 150, datas[0])
    energies = []
    for frame in frames:
        energies.append(get_energy(frame))
    print(find_treshhold(energies))
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(datas[1])
    axs[0].set_title("signal")
    axs[1].plot(energies)
    axs[1].set_title("energy")
    #plt.plot(datas[1])
    #plt.plot(energies)








files = file_picker("slt")
analyze(files[0])
# audio = np.array([1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76, 1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76], dtype=np.float)
# normalized_signal = normalise(audio)
# tab = split(normalized_signal, 1000, 1000, 200)
# print(tab)
# f0 = autocorrelation(tab[0], 200)
# print(f0)
