import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd 
import numpy
from PIL import Image
import os

model=torch.load("")
if(torch.cuda.is_available()):
	model.cuda()
print("model loaded")
dataF=torch.load("")
dataNF=torch.load("")
labelF=torch.tensor(dataF.size(1)).fill(1)
labelNF=torch.tensor(dataNF.size()).fill(2)
testData=torch.cat(dataF,dataNF,1)
testLabel=torch.cat(labelF,labelNF,1)
numSamples=testData.size(1)
print('Dataset loaded')

