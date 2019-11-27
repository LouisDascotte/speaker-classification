import random
import glob

def filepicker(speaker_name):
    allPaths = glob.glob("cmu_us_"+speaker_name+"_arctic/wav/*.wav")
    rands = random.sample(range(len(allPaths)-1), 5)
    files = []
    for i in rands:
        files.append(allPaths[i].split("/wav\\")[1])
    return files

r = "bdl"
print(filepicker(r))