import torch.nn as nn
import torch.dpnn as dpnn
import torch.rnn as rnn
import torch.extracunn as extracunn


if(torch.cuda.is_available()):
	import torch.cuda as cuda
	backend=cuda
else:
	backend=nn
class ConvLSTM(nn.LSTM):
	def __init__(self,inputSize,outputSize,rho,kc,km,stride,batchSize):
		self.kc=kc
		self.km=km
		self.padc=torch.floor(kc/2)
		self.padm=torch.floor(km/2)
		self.stride=stride or 1
		self.batchSize=batchSize
		'''init parent if req'''

	def buildGate(self):
		hidden=nn.Sequential()
		hidden.add(nn.NarrowTable(1,2))
		input2gate=backend.SpatialConvolution(self.inputSize)
		output2gate=nn.SpatialConvolutionNoBias(self.outputSize)
		para=nn.ParallelTable()
		para.add(input2gate)
		para.add(output2gate)
		hidden.add(nn.CAddTable())
		hidden.add(backend.Sigmoid())
		return hidden
	def buildInputGate(self):
		self.inputGate=self.buildGate()
		return self.inputGate
	def buildForgetGate(self):
		self.forgetGate=self.buildGate()
		return self.forgetGate
	def buildcellGate(self):
		hidden=nn.Sequential()
		hidden.add(nn.NarrowTable(1,2))
		input2gate=backend.SpatialConvolution(self.inputSize)
		output2gate=nn.SpatialConvolutionNoBias(self.outputSize)
		para=nn.ParallelTable()
		para.add(input2gate)
		para.add(output2gate)
		hidden.add(para)
		hidden.add(nn.cAddTable())
		hidden.add(backend.Tanh())
		self.cellGate=hidden
		return hidden
	def buildCell(self):
		self.inputGate=self.buildInputGate()
		self.forgetGate=self.buildForgetGate()
		self.cellGate=self.buildcellGate()
		forget=nn.Sequential()
		concat=nn.ConcatTable()
		concat.add(self.forgetGate)
		concat.add(nn.SelectTable(3))
		forget.add(concat)
		forget.add(nn.CMulTable())
		inputg=nn.Sequential()
		concat2=nn.ConcatTable()
		concat2.add(self.inputGate)
		concat2.add(self.cellGate)
		inputg.add(concat2)
		inputg.add(nn.CMulTable())
		cell=nn.Sequential()
		concat3=nn.ConcatTable()
		concat3.add(forget)
		concat3.add(inputg)
		cell.add(concat3)
		cell.add(nn.CAddTable())
		self.cell=cell
		return cell
	def buildOutputGate(self):
		self.outputGate=self.buildGate()
		return self.outputGate
	def buildModel(self):
		self.cell=self.buildCell()
		self.outputGate=self.buildOutputGate()
		concat=nn.ConcatTable()
		concat.add(nn.NarrowTable(1,2))
		concat.add(self.cell)
		model=nn.Sequential()
		model.add(nn.FlattenTable())
		cellAct=nn.Sequential()
		cellAct.add(nn.SelectTable(3))
		cellAct.add(backend.Tanh())
		concat2=nn.ConcatTable()
		concat2.add(self.outputGate)
		concat2.add(cellAct)
		output=nn.Sequential()
		output.add(concat3)
		output.add(nn.CMulTable())
		concat3=nn.ConcatTable()
		concat3.add(output)
		concat3.add(nn.SelectTable(3))
		model.add(concat3)
		return model
	def updateOutput(self,input):
		if self.step is 1 :
			prevOutput=self.userPrevOutput or self.zeroTensor
			prevCell=self.userPrevCell or self.zeroTensor
			if self.batchSize :
				self.zeroTensor.resize(self.batchSize,self.outputSize,input.size(3),input.size(4)).zero()
			else :
				self.zeroTensor.resize(self.outputSize,input.size(2),input.size(3)).zero()
		else:
			prevOutput=self.output
			prevCell=self.cell

		if not self.train :
			self.recycle()
			recurrentModule=self.getStepModule(self.step)

			output,cell=unpack
