import torch.nn as nn
import torch.dpnn as dpnn
import torch.rnn as rnn
import torch.extracunn as extracunn


if(torch.cuda.is_available()):
	import torch.cuda as cuda
	backend=cuda
else:
	backend=nn
