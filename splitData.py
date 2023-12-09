import os
import random
import shutil
from itertools import islice

inputFolderPath = "Dataset/all"
outputFolderPath = "Dataset/SplitData"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
    print("Directory removed!")
# ----If the directionary hasn't exist yet then create it-----#
except OSError as e:
    os.mkdir(outputFolderPath)

# ----- Directories to Create -----#
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# ----- Get the Names -----#
listNames = os.listdir(inputFolderPath)
print(listNames)

uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))

# ----- Shuffle -----#
random.shuffle(uniqueNames)

# ----- Get the number of images in each folder -----#
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio["train"])
print(splitRatio["train"])
lenVal = int(lenData * splitRatio["val"])
lenTest = int(lenData * splitRatio["test"])

# ----- Put the remaining images in training-----#
if lenData != lenVal + lenTrain + lenTest:
    remaining = lenData - (lenVal + lenTrain + lenTest)
    lenTrain += remaining

# ----- Split the list -----#
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f"Total image: {lenData}\nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}")

# ----- Copy the Files -----#
sequence = ["train", "val", "test"]
for i, out in enumerate(Output):
    for filename in out:
        shutil.copy(f"{inputFolderPath}/{filename}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{filename}.jpg")
        shutil.copy(f"{inputFolderPath}/{filename}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{filename}.txt")
print("Split Process Completed!")

# ----- Creating Data.yaml file -----#
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yml file created!")


