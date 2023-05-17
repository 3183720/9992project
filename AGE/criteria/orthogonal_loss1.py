import torch
from torch import nn


class OrthogonalLoss(nn.Module):

	def __init__(self, opts):
		super(OrthogonalLoss, self).__init__()

	def forward(self, A):
		dim = A.shape[-1]
		return torch.sum(torch.abs(torch.matmul(A.transpose(1,2), A)))/(dim*A.shape[-1]*6)
