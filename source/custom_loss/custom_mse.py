## https://github.com/vincent-leguen/DILATE/blob/5f114b587fd7abb7b79726ed68d8e4a91049cc0e/loss/dilate_loss.py

import torch
# from custom_loss import soft_dtw
# from custom_loss import path_soft_dtw 


import torch.nn as nn

# class custom_mseLoss(nn.Module):
# 	def __init__(self, )




def dilate_loss(input, target, alpha=0.4, gamma=0.001):

	outputs, targets = input, target

	device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal