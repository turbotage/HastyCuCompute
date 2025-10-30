"""Rotation functions in 2D & 3D spaces using PyTorch."""
import math
import torch
from torch import Tensor

import torch

def R2D(theta: torch.Tensor) -> torch.Tensor:
    """Batch 2D rotation matrices. theta: [B]"""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1)
    ], dim=-2)  # Shape: [B, 2, 2]

def Rx(theta: torch.Tensor) -> torch.Tensor:
    """Batch 3D rotation matrices around x-axis. theta: [B]"""
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, c, -s], dim=-1),
        torch.stack([zeros, s,  c], dim=-1)
    ], dim=-2)  # Shape: [B, 3, 3]

def Ry(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack([
        torch.stack([c, zeros, s], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-s, zeros, c], dim=-1)
    ], dim=-2)

def Rz(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack([
        torch.stack([c, -s, zeros], dim=-1),
        torch.stack([s,  c, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

def Rv(v1: torch.Tensor, v2: torch.Tensor, eps=1e-8, normalize=True):
    """Batch rotation matrix from v1 to v2. v1,v2: [B,3]"""
    if normalize:
        v1 = v1 / v1.norm(dim=-1, keepdim=True)
        v2 = v2 / v2.norm(dim=-1, keepdim=True)

    cross = torch.cross(v1, v2, dim=-1)
    cos_theta = (v1 * v2).sum(dim=-1, keepdim=True)
    norms = cross.norm(dim=-1, keepdim=True)
    
    # Handle colinear case
    mask = norms.squeeze(-1) < eps
    B = v1.shape[0]
    R = torch.zeros((B, 3, 3), device=v1.device, dtype=v1.dtype)
    
    # Rodrigues formula
    K = torch.zeros((B, 3, 3), device=v1.device, dtype=v1.dtype)
    K[:, 0, 1] = -cross[:, 2]
    K[:, 0, 2] =  cross[:, 1]
    K[:, 1, 0] =  cross[:, 2]
    K[:, 1, 2] = -cross[:, 0]
    K[:, 2, 0] = -cross[:, 1]
    K[:, 2, 1] =  cross[:, 0]
    
    R = torch.eye(3, device=v1.device, dtype=v1.dtype).unsqueeze(0) + K + K @ K / (1 + cos_theta.unsqueeze(-1))
    # Fix colinear cases
    R[mask] = torch.eye(3, device=v1.device, dtype=v1.dtype) * torch.sign((v1 * v2).sum(dim=-1, keepdim=False)[mask])[:, None, None]
    return R

def Ra(vector: torch.Tensor, theta: torch.Tensor):
    """Batch rotation around arbitrary axis. vector: [B,3], theta: [B]"""
    vector = vector / vector.norm(dim=-1, keepdim=True)
    vx, vy, vz = vector[:,0], vector[:,1], vector[:,2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    one_c = 1 - c

    R = torch.zeros((vector.shape[0], 3,3), device=vector.device, dtype=vector.dtype)
    R[:,0,0] = c + vx**2 * one_c
    R[:,0,1] = vx*vy*one_c + vz*s
    R[:,0,2] = vx*vz*one_c - vy*s
    R[:,1,0] = vy*vx*one_c - vz*s
    R[:,1,1] = c + vy**2 * one_c
    R[:,1,2] = vy*vz*one_c + vx*s
    R[:,2,0] = vz*vx*one_c + vy*s
    R[:,2,1] = vz*vy*one_c - vx*s
    R[:,2,2] = c + vz**2 * one_c
    return R


class SurfacePDF:
	def pdf(self, x: float) -> float:
		return math.sin(math.pi*x)
		# if x <= 0:
		# 	return 0
		# elif x >= 1:
		# 	return 0
		# else:
		# 	return 2*math.sin(math.pi*x)
		
	def dpdf(self, x: float) -> float:
		# if x <= 0:
		# 	return 0
		# elif x >= 1:
		# 	return 0
		# else:
		# 	return math.pi*math.cos(math.pi*x)
		return math.pi*math.cos(math.pi*x)
	
	def mean(self) -> float:
		return 0.5