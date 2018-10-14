unpack = unpack or table.unpack
import torch.nn as nn
import torch.rnn as rnn
import torch.cunn as cunn
import torch.paths as paths
import torch.cutorch as cutorch
import torch.cudnn as cudnn
import torch.image as image
import torch.optim as optim
import torch.loadcaffe as loadcaffe
import torch.ConvLSTM as ConvLSTM

cutorch.setDevice(1)
opt = {}
opt[inputSizeW] = 224
opt[inputSizeH] = 224
opt[kernelSize] = 3
opt[padding] = torch.floor(opt[kernelSize]/2)
opt[stride] = 1
opt[nSeq] = 20

protxt = 'bvlc_alexnet.prototxt'
binary = 'bvlc_alexnet.caffemodel'
alexnet = loadcaffe.load(prototxt, binary, 'cudnn')

for i in range(1,9):
    alexnet.remove()

lstm_mod = nn.Sequential()
lstm_mod.add(nn.ConvLSTM(256,256,opt[nSeq]-1,opt[kernelSize],opt[stride],opt[batchSize]))
lstm_mod = w_init(lstm_mod, 'xavier')

class_layer1 = nn.Sequential()
class_layer1.add(nn.SpatialMaxPooling(2,2))
class_layer1.add(nn.Reshape(256*3*3))
class_layer1.add(nn.Linear(256*3*3, 1000))
class_layer1.add(nn.BatchNormalization(1000))
class_layer1.add(cudnn.ReLU())
class_layer1.add(nn.Linear(10,2))
class_layer1.add(cudnn.LogSoftMax())

encoder = nn.Sequential()
encoder.add(alexnet)
encoder.add(lstm_mod)
encoder = nn.Sequencer(encoder)

model = nn.Sequential()
model.add(encoder)
model.add(nn.SelectTable(-1))
model.add(class_layer1)
