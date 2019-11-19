import numpy as np

def normalise(signal):
    return signal/np.max(np.abs(signal))

def split(signal, w, step, fs):
    step_n = step/1000 * fs
    width_n = w/1000 * fs
    

def get_signal_energy(signal):
    return np.sum(np.square(signal))




audio = np.array([1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76])

# print(normalise(audio))

print(get_signal_energy(audio))