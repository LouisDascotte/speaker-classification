import numpy as np

def normalise(signal):
    return signal/np.max(np.abs(signal))

def split(signal, width, shiftingstep, samplefreq):
    shifting = (shiftingstep/1000)*samplefreq # taille du shif en points
    widthsize = (width/1000)*samplefreq # taille de la fenetre en points
    step = np.size(signal)/shifting
    frames = np.array([])
    for i in range(step):
        if((i*shifting)+widthsize > np.size(signal)):
            diff = ((i*shifting)+widthsize) - np.size(signal)
            frames.append(np.array(signal[i*shifting:(i*shifting)+widthsize-diff]))
        else:    
            frames.append(np.array(signal[i*shifting:(i*shifting)+widthsize]))
    return frames
    

def get_signal_energy(signal):
    return np.sum(np.square(signal))




audio = np.array([1,3,4,65,7,8,8,9,6,4,3,32,12,31,4,45,64,75,-100,-23,-76])

# print(normalise(audio))

print(get_signal_energy(audio))
