import numpy as np
#width et shiftingstep en ms
#samplefreq en hz
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

