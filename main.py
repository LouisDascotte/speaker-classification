import numpy as np

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
    

def get_signal_energy(signal):
    return np.sum(np.square(signal))




audio = np.array([1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76])

# print(normalise(audio))

print(split(audio, 1000, 1000, 5))
