import torch

def Rx(theta: torch.Tensor) -> torch.Tensor:
	
    rotmats = torch.zeros(theta.shape[0], 3, 3)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rotmats[:,0,0] = 1
    #rotmats[:,0,1] = 0
    #rotmats[:,0,2] = 0

    #rotmats[:,1,0] = 0
    rotmats[:,1,1] = cos_t
    rotmats[:,1,2] = -sin_t

    #rotmats[:,2,0] = 0
    rotmats[:,2,1] = sin_t
    rotmats[:,2,2] = cos_t

    return rotmats

def Ry(theta: torch.Tensor) -> torch.Tensor:
	
	rotmats = torch.zeros(theta.shape[0], 3, 3)

	cos_t = torch.cos(theta)
	sin_t = torch.sin(theta)

	rotmats[:,0,0] = cos_t
	#rotmats[:,0,1] = 0
	rotmats[:,0,2] = sin_t

	#rotmats[:,1,0] = 0
	rotmats[:,1,1] = 1
	#rotmats[:,1,2] = 0

	rotmats[:,2,0] = -sin_t
	#rotmats[:,2,1] = 0
	rotmats[:,2,2] = cos_t

	return rotmats

def Rz(theta: torch.Tensor) -> torch.Tensor:

	rotmats = torch.zeros(theta.shape[0], 3, 3)

	cos_t = torch.cos(theta)
	sin_t = torch.sin(theta)

	rotmats[:,0,0] = cos_t
	rotmats[:,0,1] = -sin_t
	#rotmats[:,0,2] = 0

	rotmats[:,1,0] = sin_t
	rotmats[:,1,1] = cos_t
	#rotmats[:,1,2] = 0

	#rotmats[:,2,0] = 0
	#rotmats[:,2,1] = 0
	rotmats[:,2,2] = 1

	return rotmats

def Ra(vector: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
	rotmats = torch.empty(theta.shape[0], 3, 3)

	cos_t = torch.cos(theta)
	sin_t = torch.sin(theta)
	vnorm = torch.linalg.norm(vector, dim=0)
	v_x = vector[0,...] / vnorm
	v_y = vector[1,...] / vnorm
	v_z = vector[2,...] / vnorm

	rotmats[:,0,0] = cos_t + v_x**2 * (1 - cos_t)
	rotmats[:,0,1] = v_x * v_y * (1 - cos_t) + v_z * sin_t
	rotmats[:,0,2] = v_x * v_z * (1 - cos_t) - v_y * sin_t

	rotmats[:,1,0] = v_y * v_x * (1 - cos_t) - v_z * sin_t
	rotmats[:,1,1] = cos_t + v_y**2 * (1 - cos_t)
	rotmats[:,1,2] = v_y * v_z * (1 - cos_t) + v_x * sin_t

	rotmats[:,2,0] = v_z * v_x * (1 - cos_t) + v_y * sin_t
	rotmats[:,2,1] = v_z * v_y * (1 - cos_t) - v_x * sin_t
	rotmats[:,2,2] = cos_t + v_z**2 * (1 - cos_t)

	return rotmats
