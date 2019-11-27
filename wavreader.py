import random
import glob
def filepicker(pathName):
    path =  pathName + "\*.wav"
    allPaths = glob.glob(path)

    for i in range(len(allPaths)):
        aux = allPaths[i].split(pathName + "\\")
        allPaths[i] = aux[1]

    randomFiles = []
    for i in range(5):
        randomChoice = random.choice(allPaths)
        while(randomChoice in randomFiles):
            randomChoice = random.choice(allPaths)
        randomFiles.append(randomChoice)
        
    print(randomFiles)

r = r"C:\Users\luisk\Documents\cmu_us_bdl_arctic\wav"
filepicker(r)