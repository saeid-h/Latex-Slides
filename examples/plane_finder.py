#################################################################################
#																				#
# Copyright (c) 2020 Saeid Hosseinipoor <https://saeid-h.github.io/>			#
#																				#
# This file is a part of BTS.													#
# This program is free software: you can redistribute it and/or modify it under	#
# the terms of the MIT license.													#
#																				#
# This program is distributed in the hope that it will be useful, but			#
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY	#
# or FITNESS FOR A PARTICULAR PURPOSE. See the	MIT License	for more details.	#
#																				#
#################################################################################

from __future__ import absolute_import, division, print_function

import glob, os
import scipy.misc 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
# import scipy.ndimage.filters as filters


def getListOfFiles(dirName, mask='*.png'):
		listOfFile = os.listdir(dirName)
		allFiles = glob.glob(dirName+mask)
		for entry in listOfFile:
			fullPath = os.path.join(dirName, entry) + '/'
			if os.path.isdir(fullPath):
				allFiles += getListOfFiles(fullPath, mask)
		
		return allFiles


def get_occlusion_attention(depth, op_type='log'):
	def op(d1, d2, op_type=op_type):
		if op_type == 'log':
			return np.abs(np.log(d1/d2))
		if op_type == 'diff':
			return np.abs(d1-d2)
	def clear_up(d2, op_type=op_type):
		d2 [d2 == np.inf] = 0
		d2 [d2 == -np.inf] = 0
		d2 = np.nan_to_num(d2)
		w = np.ones_like(depth)
		# if op_type == 'diff': w [depth == 0] = 0
		return d2*w

	d3 = 0
	d2 = np.roll(depth, 2, axis=0)
	d2 = op(depth, d2, op_type)
	d2 = np.roll(d2, -1, axis=0)
	d2 = clear_up(d2)
	d2 [:2,:] = 0
	d2 [-2:,:] = 0

	d2 = np.roll(depth, 2, axis=0)
	d2 = op(depth, d2, op_type)
	d2 = np.roll(d2, -1, axis=0)
	d2 = clear_up(d2)
	d2 [:2,:] = 0
	d2 [-2:,:] = 0
	d3 += d2

	d2 = np.roll(depth, 2, axis=1)
	d2 = op(depth, d2, op_type)
	d2 = np.roll(d2, -1, axis=1)
	d2 = clear_up(d2)
	d2 [:,:2] = 0
	d2 [:, -2:] = 0
	d3 += d2

	d2 = np.roll(depth, 2, axis=(0,1))
	d2 = op(depth, d2, op_type)
	d2 = np.roll(d2, -1, axis=(0,1))
	d2 = clear_up(d2)
	d2 [:2,:2] = 0
	d2 [-2:, -2:] = 0
	d3 += d2

	d2 = np.roll(depth, (2,-2), axis=(0,1))
	d2 = op(depth, d2, op_type)
	d2 = np.roll(d2, (-1,1), axis=(0,1))
	d2 = clear_up(d2)
	d2 [:2,:2] = 0
	d2 [-2:, -2:] = 0
	d3 += d2

	return d3


def conv_3D_to_2D(P, H, W, f, dtype='int'):
	x = P[0]*f/P[2]+H/2
	y = P[1]*f/P[2]+W/2
	if dtype == 'int':
		x = int(x)
		y = int(y)
	return [x, y]
	
def conv_2D_to_3D(p, H, W, f):
	x = p[0] - H/2
	y = p[1] - W/2
	Z = p[2]
	X = x / f * Z
	Y = y / f * Z
	return [X, Y, Z]

def list_to_str(L, fmt='4.4f'):
	string = ''
	for l in L:
		string += format(l, fmt) + ' '
	return string	
	

def make_3D_ply(depth, divider=None, candidate_points=None, Avg_point=None, ply_name=None, color=None, f=600):
	
	if color is None:
		color = [255, 255, 255]
	if ply_name is None:
		ply_name = 'image.ply'
	H = depth.shape[0]
	W = depth.shape[1]
	max_z = 1.2*np.amax(depth)
	total_points = H*W+1
	if divider is not None:
		total_points += H*W
	if candidate_points is not None:
		total_points += len(candidate_points)
	if Avg_point is not None:
		total_points += 1
	
	plausable_points = total_points
	
	ply_file = open(ply_name, 'w')
	ply_file.write("ply\n")
	ply_file.write("format ascii 1.0\n")
	ply_file.write("element vertex {}\n".format(total_points))
	ply_file.write("property float x\n")
	ply_file.write("property float y\n")
	ply_file.write("property float z\n")
	ply_file.write("property uchar red\n")
	ply_file.write("property uchar green \n")
	ply_file.write("property uchar blue\n")
	ply_file.write("element face 0\n")
	ply_file.write("end_header\n")
	
	ply_file.write('0 0 0 0 255 0\n')
	if Avg_point is not None:
		point = Avg_point[0,0,:] 
		ply_file.write(list_to_str([point[0],point[1],-point[2]],fmt='4.0f')+"255 255 0\n")
	if candidate_points is not None:
		for point in candidate_points:
			ply_file.write(list_to_str([point[0],point[1],-point[2]],fmt='4.0f')+"255 0 0\n")
	
	
	for i in range(H):
		for j in range(W):
			point = conv_2D_to_3D([i, j, depth[i,j]], H, W, f)
			ply_file.write(list_to_str([point[0],point[1],-point[2]],fmt='4.0f')+list_to_str(color, fmt='d')+"\n")
			if divider is not None:
				point = conv_2D_to_3D([i, j, divider[i,j]], H, W, f)
				if np.abs(point[2]) < max_z:
					ply_file.write(list_to_str([point[0],point[1],-point[2]],fmt='4.0f')+list_to_str([0, 0, 255], fmt='d')+"\n")
				else:
					#ply_file.write('0 0 0 0 255 0\n')
					plausable_points -= 1
			
	ply_file.close()
	os.system("perl -p -i -e 's/"+"element vertex {}\n".format(total_points)+"/"+"element vertex {}\n".format(plausable_points)+"/g' "+ply_name)


def add_line_to_image (image, Norm, Middle, col=[0,255,0], radius=3, f=600):
	valid_points = []
	H = image.shape[0]; W = image.shape[1]
		
	x = 0
	X = x - H/2
	alpha = (Middle[0]-X/f*Middle[2]) / (X/f*Norm[2]-Norm[0])
	if np.isfinite(alpha):
		MM = Middle + alpha * Norm
		y = int(MM[1] / MM[2] * f + W/2)
		if x < H and x >= 0 and y < W and y >= 0:
			valid_points += [(x,y)]
				
	x = H - 1 
	X = x - H/2
	alpha = (Middle[0]-X/f*Middle[2]) / (X/f*Norm[2]-Norm[0])
	if np.isfinite(alpha):
		MM = Middle + alpha * Norm
		y = int(MM[1] / MM[2] * f + W/2)
		if x < H and x >= 0 and y < W and y >= 0:
			valid_points += [(x,y)]
		
	y = 0
	Y = y - W/2
	alpha = (Middle[1]-Y/f*Middle[2]) / (Y/f*Norm[2]-Norm[1])
	if np.isfinite(alpha):
		MM = Middle + alpha * Norm
		x = int(MM[0] / MM[2] * f + H/2)
		if x < H and x >= 0 and y < W and y >= 0:
			valid_points += [(x,y)]
			
	y = W - 1
	Y = y - W/2
	alpha = (Middle[1]-Y/f*Middle[2]) / (Y/f*Norm[2]-Norm[1])
	if np.isfinite(alpha):
		MM = Middle + alpha * Norm
		x = int(MM[0] / MM[2] * f + H/2)
		if x < H and x >= 0 and y < W and y >= 0:
			valid_points += [(x,y)]
	
	if len(valid_points) == 2:
		x1 = valid_points[0][0]
		y1 = valid_points[0][1]
		x2 = valid_points[1][0]
		y2 = valid_points[1][1]
		radius -= 2
		if x2 != x1:
			inc = 1 if x2>x1 else -1
			a = (y2-y1) / (x2-x1)
			b = y1 - a*x1
			x = x1 + inc
			y = int(a*x + b)
			while x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = col
				x = x + inc
				y = int(a*x + b)
		else:
			inc = 1 if y2>y1 else -1
			x = x1
			y = y1 + inc
			while x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = col
				y = y + inc
					
	x = int(Middle[0] / Middle[2] * f + H/2)
	y = int(Middle[1] / Middle[2] * f + W/2)
	radius += 4
	if x < H and x >= 0 and y < W and y >= 0:
		image [x-radius:x+radius, y-radius:y+radius, :] = 0.
		image [x-radius:x+radius, y-radius:y+radius, 2] = 255.
		
	return image


def points_on_image(depth, marked_points, radius=3, shift=1, col=[255, 0, 0], f=600, line=None):
		H = depth.shape[0]; W = depth.shape[1]
		colored_image = depth[:,:,np.newaxis] / np.max(depth) * 255.
		colored_image = np.concatenate((colored_image, colored_image, colored_image), axis=2) 
		
		for i in range(marked_points.shape[0]):
			x, y = conv_3D_to_2D(marked_points[i,:], H, W, f)
			colored_image [x-radius:x+radius, y-radius:y+radius, :] = col
			
		if line is not None:
			Norm, Middle = line
			colored_image = add_line_to_image (colored_image, Norm, Middle, f=f)
			colored_image = add_line_to_image (colored_image, np.asarray([1,0,0]), Middle, f=f)
			colored_image = add_line_to_image (colored_image, 
											np.asarray([0,Norm[2],-Norm[1]]), Middle, col=[0,0,255], f=f)
					
		return colored_image.astype(np.uint8)


def get_points (depth, seg=None, n=10, k=5, f=600, shift=1, th=250, dist='linear'):
	data = list()
	H = depth.shape[0]; W = depth.shape[1]
	if dist == 'log': th = np.log(1+th/1000 )
	
	for i in range(n):
			y = int(np.random.uniform(W*1/4, W*3/4))
			if seg is None:
				c = depth[:, y]
				cc = depth[:, y]
			else:
				c = seg[:, y]
				cc = depth[:, y]
				
			c_seg = np.zeros((np.sum(cc!=0),2))
			c_depth = np.zeros((np.sum(cc!=0),2))
				
			i = 0
			for ci in range (len(c)):
				if cc[ci] != 0:
					c_seg [i, :] = [ci, c[ci]]
					c_depth [i, :] = [ci, cc[ci]]
					i += 1
			
			if len(c_depth) > 0:
				gap_mask = (c_seg[:-shift, 1] != c_seg[shift:, 1]).astype(np.float32)
				if dist == 'log':
					diff = np.abs(np.log(c_depth[:-shift, 1] / c_depth[shift:, 1]))
				else:
					diff = np.abs(c_depth[:-shift, 1] - c_depth[shift:, 1])
				diff = diff * gap_mask
				th_len = len(diff[diff>th])
								
				X = np.argsort(diff)[-k:][::-1]
			
				for j in range(min(k, len(X), th_len)):
					p1 = [c_depth[X[j], 0], y, c_depth[X[j], 1]]
					p2 = [c_depth[X[j]+shift, 0], y, c_depth[X[j]+shift, 1]]
					X1, Y1, Z1 = conv_2D_to_3D(p1, H, W, f)
					X2, Y2, Z2 = conv_2D_to_3D(p2, H, W, f)
					data.append([0.5*(X1+X2), 0.5*(Y1+Y2), 0.5*(Z1+Z2)])
					
			x = int(np.random.uniform(H*1/4, H*3/4))
			if seg is None:
				r = depth[x, :]
				rr = depth[x, :]
			else:
				r = seg[x, :]
				rr = depth[x, :]
				
			r_seg = np.zeros((np.sum(rr!=0),2))
			r_depth = np.zeros((np.sum(rr!=0),2))
			
			j = 0
			for ri in range (len(r)):
				if rr[ri] != 0:
					r_seg[j, :] = [ri, r[ri]]
					r_depth[j, :] = [ri, rr[ri]]
					j += 1
					
			if len(r_depth) > 0:
				gap_mask = (r_seg[:-shift, 1] != r_seg[shift:, 1]).astype(np.float32)
				if dist == 'log':
					diff = np.abs(np.log(r_depth[:-shift, 1] / r_depth[shift:, 1]))
				else:
					diff = np.abs(r_depth[:-shift, 1] - r_depth[shift:, 1])
				diff = diff * gap_mask
				th_len = len(diff[diff>th])
							
				Y = np.argsort(diff)[-k:][::-1]
			
				for j in range(min(k, len(Y), th_len)):
					p1 = [x, r_depth[Y[j], 0], r_depth[Y[j], 1]]
					p2 = [x, r_depth[Y[j]+shift, 0], r_depth[Y[j]+shift, 1]]
					X1, Y1, Z1 = conv_2D_to_3D(p1, H, W, f)
					X2, Y2, Z2 = conv_2D_to_3D(p2, H, W, f)
					data.append([0.5*(X1+X2), 0.5*(Y1+Y2), 0.5*(Z1+Z2)])

	data = np.asarray(data)
	if len(data) > 0:
		data = np.unique(data, axis=0)
	return data


def max_supression(matrix, window):
	h, w = window
	H, W = matrix.shape
	for i in range(h//2, H-h//2):
		for j in range(w//2, W-w//2):
			matrix_slice = matrix[i-h//2:i+h//2, j-w//2:j+w//2]
			ind = np.unravel_index(np.argmax(matrix_slice, axis=None), (h,w))
			temp = matrix_slice[ind]
			matrix_slice[:] = 0
			matrix_slice[ind] = temp
			matrix[i-h//2:i+h//2, j-w//2:j+w//2] = matrix_slice
	return matrix


def add_line_on_floor (image, N, M, col=[0,1,0], radius=3):
	valid_points = []
	H, W, _ = image.shape

	for x in [0, H-1]:
		alpha = (x - M[0]) / N[0]
		if np.isfinite(alpha):
			y = int(M[1] + alpha * N[1])
			if x < H and x >= 0 and y < W and y >= 0:
				valid_points += [(x,y)]
				
	for y in [0, W-1]:
		alpha = (y - M[1]) / N[1]
		if np.isfinite(alpha):
			x = int(M[0] + alpha * N[0])
			if x < H and x >= 0 and y < W and y >= 0:
				valid_points += [(x,y)]
			
	if len(valid_points) == 2:
		x1, y1 = valid_points[0]
		x2, y2 = valid_points[1]
		radius -= 2
		
		if np.abs(y2-y1) < np.abs(x2-x1) and x2 != x1 and y2 != y1:
			inc = np.sign(x2-x1)
			m = (y2-y1) / (x2-x1)
			b = y1 - m*x1
			x = x1 + inc
			y = int(m*x + b)
			while x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = col
				x += inc
				y = int(m*x + b)
		
		elif x2 != x1 and y2 != y1:
			inc = np.sign(y2-y1)
			m = (y2-y1) / (x2-x1)
			b = y1 - m*x1
			y = y1 + inc
			x = int((y-b)/m)
			while x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = col
				y += inc
				x = int((y-b)/m)
		
		elif x2 == x1:
			inc = np.sign(y2-y1)
			x = x1
			y = y1 + inc
			while x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = col
				y += inc
		
		else:
			inc = np.sign(x2-x1)
			y = y1
			x = x1 + inc
			while x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = col
				x += inc

		radius += 3
		x, y = map(int, M)
		if x < H and x >= 0 and y < W and y >= 0:
				image [x-radius:x+radius, y-radius:y+radius, :] = [0, 0, 1]
		else:
			print (x, y)
					
	return image


def projection_on_floor(depth, all_points=None, points=None, line=None, default_color=[0, 0.2, 0.4],
					radius=3, col=[1,0,0], z_scale=None, f=600):
	H, W = depth.shape
	y_max = W // 2
	y_min = -y_max
	z_max = int(np.max(depth))
	z_min = int(np.min(depth[depth>0]))
	if z_scale is None:
		z_scale = (z_max-z_min)/W
		z_max = int(z_max/z_scale) + 1
		z_min = int(z_min/z_scale) - 1
	img = np.ones([W, z_max-z_min, 3]) 
	img[:,:,:] = default_color

	for j in range(W):
		for i in range(H-1, -1, -1):
			z = int(depth[i,j]/z_scale - z_min)
			y = j
			if z > 0:
				img[y-radius:y+radius, z-radius:z+radius, :] = 1 - i / H 

	if all_points is not None:
		for p in all_points:
			_, y = conv_3D_to_2D(p, H, W, f)
			z = int(p[2]/z_scale)-z_min 
			img [y-radius:y+radius, z-radius:z+radius, :] = [1,1,0]
	if points is not None:
		for p in points:
			_, y = conv_3D_to_2D(p, H, W, f)
			z = int(p[2]/z_scale)-z_min 
			img [y-radius:y+radius, z-radius:z+radius, :] = [1,0,0]
			
	if line is not None:
		Norm, Middle = line
		Norm = [Norm[1], Norm[2]]
		_, y = conv_3D_to_2D(Middle, H, W, f, 'float')
		Middle = [y, Middle[2]/z_scale-z_min]
		img = add_line_on_floor (img, Norm, Middle, col=[0,1,0], radius=radius)
		
	return np.rot90(img)


def find_lines (data, number_of_lines=1, method='ransac',
				plane_th=500, p=0.999, s=2, keep_points=0.4, verbose=False):

	line_norms = []
	Ms = []
	selected_data = []

	if method == 'ransac':	
		current_data = data
		for i in range(number_of_lines):
			N, M, D = find_line(current_data, plane_th=plane_th, p=p, s=s, verbose=verbose)
			line_norms.append(N)
			Ms.append(M)
			selected_data.append(D)
			old_data = current_data
			np.random.shuffle(old_data)
			current_data = []
			for point in old_data:
				if not point in D or len(current_data) < keep_points*len(old_data):
					current_data.append(point)
			current_data = np.asarray(current_data)
	
	elif method == 'hough':
		sin = np.sin
		cos = np.cos

		z_min = np.min(data[:,2])
		z_max = np.max(data[:,2])
		y_min = np.min(data[:,1])
		y_max = max(np.max(data[:,1]), np.abs(y_min))
		
		rho_max = np.hypot(y_max, z_max)
		rho_min = -rho_max
		n_rho = min(25, max(50, int((rho_max - rho_min) / 200) + 1))

		n_theta = 45
		Theta = np.linspace(0, np.pi, n_theta)
		funcs = zip(cos(Theta), sin(Theta))

		hough_space = np.zeros([n_rho, n_theta])
		D = dict()
		
		for point in data:
			y = point[1]
			z = point[2]
			for t, f in enumerate(funcs):
				rho = y * f[0] + z * f[1]
				r  = int((rho - rho_min) / (rho_max - rho_min) * n_rho)
				hough_space[r, t] += 1
				if not (r, t) in D.keys(): 
					D.update({(r, t): [point]}) 
				else:
					D[(r, t)] += [point]

		neighborhood_size = [2, 4]
		hough_space = max_supression(hough_space, neighborhood_size)
		top_k = []
		maxima = (hough_space != 0)
		candidates = np.argwhere(maxima)
		
		if number_of_lines is None:
			cp = np.sort([hough_space[g[0],g[1]] for g in candidates])
			number_of_lines = 1
			while cp[-number_of_lines] / cp[-1] > 0.65 and number_of_lines < 10:
				number_of_lines += 1
		np.random.shuffle(candidates)
		for k  in range(number_of_lines):
			max_k = 0
			for g in candidates:
				if not [g[0],g[1]] in top_k and hough_space[g[0],g[1]] > max_k:
					max_k = hough_space[g[0],g[1]]
					best_g = [g[0],g[1]]
			top_k.append(best_g)
				
		for k in top_k:
			rho = rho_min+(k[0])*(rho_max - rho_min)/n_rho
			theta = Theta[0]+(k[1])*(Theta[1]-Theta[0])
			z =  z_min+(z_max-z_min)/10
			y = 1
			if sin(theta) == 0:
				N = [0, 0, 1]
			else:
				N = [0, 100, -100*cos(theta)/sin(theta)]
			DD = np.asarray(D[(k[0], k[1])])
			M = DD.mean(axis=0)
			N = np.asarray(N)
			line_norms.append(N)
			Ms.append(M)
			selected_data.append(DD)
			
	else:
		print ("Error line finder: Wrong method!")

	return line_norms, Ms, selected_data


def find_line(data, plane_th=500, p=0.999, s=2, verbose=False):
	
	if len(data) < 2:
		return None, None, None
		
	total_number_of_points = data[:,0].size
	sample_count = 0
	iteration = 0
	max_iteration = np.inf
	best_dist = np.inf
	best_inliers = None
	best_sample_count = 0

	while max_iteration > iteration or best_inliers is None:
		I = np.random.uniform(0, data.shape[0], s)
		I = [int(i) for i in I]
		hyp_inliers = data[I, :]
		
		M = hyp_inliers.mean(axis=0)
		_, _, v = np.linalg.svd(hyp_inliers - M)
		N = [0, v[0][2], -v[0][1]]
		if np.linalg.norm(N) != 0:
			N = N / np.linalg.norm(N)
		else:
			N = [0, 0, 1]
		
		inliers = list()
		sample_count = 0
	
		for d in data:
			if np.linalg.norm(N) != 0:
				if np.abs(np.sum(N*(d-M))) < plane_th:
					inliers.append(d)
					sample_count += 1
			else:
				iteration -= 1
		
		if sample_count > best_sample_count:
			best_inliers = inliers
			best_sample_count = sample_count
		
		e = 1 - best_sample_count / total_number_of_points
		if e == 0:
			e += 10e-5
		max_iteration = np.log(1-p) / np.log(1-(1-e)**s)
		iteration += 1
		if iteration % 1000 == 0 and verbose:
			print (iteration, max_iteration, 1-e)
			
	inliers = np.asarray(best_inliers)
	M = inliers.mean(axis=0)
	_, _, v = np.linalg.svd(inliers - M)
	
	if verbose:
		print ("{:d}, {:5.1f}, {:5.2f}, {}, {}".format(iteration, max_iteration, 1-e, v[0], M))

	
	return v[0], M, inliers


def add_object(image, depth, new_object, position=None):
	image = image.astype(np.float)
	depth = depth.astype(np.float)
	new_object = new_object.astype(np.float)
	h, w, _ = new_object.shape

	if position is None:
		rh = int(np.random.uniform(0,image.shape[0]-h-1))
		rw = int(np.random.uniform(0,image.shape[1]-w-1))
	else:
		rh, rw = position

	d_temp = depth[rh:rh+h, rw:rw+w]
	d_max = np.max(d_temp) 
	d_min = np.min(d_temp[d_temp!=0])
	d = np.random.uniform(1.1*d_min,0.9*d_max)

	mask = new_object[:,:,-1:]
	mask[mask != 0] = 1

	depth_slice = depth[rh:rh+h, rw:rw+w, np.newaxis]
	depth_slice[depth_slice<d] = 0
	depth_slice[depth_slice>=d] = 1
	mask *= depth_slice
	
	image [rh:rh+h, rw:rw+w,:] *= (1-mask)
	new_object[:,:,:3] *= mask
	image [rh:rh+h, rw:rw+w,:] += new_object[:,:,:3]

	return image.astype(np.uint8)



if __name__ == "__main__":

	source = 'images/'
	exclusion_file_names = []
	inclusion_file_names = []

	depth_files = getListOfFiles(source, '*.png')
	depth_files.sort()

	file_names = [f.split('_')[0].split('/')[1] for f in depth_files]
	file_names = list(set(file_names))
	file_names = [f.split('/')[1] for f in depth_files \
				if 'aug_' not in f and 'mask_' not in f and 'cand_dots_' not in f 
				and 'Z_' not in f and 'instances_' not in f and 'rgb_' not in f
				and 'all_dots_' not in f and 'floor_' not in f 
				and 'occlusion_attention_' not in f and 'bob_' not in f]

	Save_image = {	'aug_': False,
					'aug_instances_': False,
					'all_dots_': True,
					'cand_dots_': True,
					'floor_': False,
					'mask_': True,
					'combined_mask_': True,
					'occlusion_attention_': True,
					'3D_ply': False,
					'add_bob': True,
					'floor_combined_': False,
				}

	f = 518.8579
	n = 50
	k = 10
	shift=1

	if Save_image['cand_dots_']:
		os.system("rm "+source+'cand_dots_*')
	if Save_image['floor_']:
		os.system("rm "+source+'floor_*')
	if Save_image['mask_']:
		os.system("rm "+source+'mask_*')
	if Save_image['3D_ply']:
		os.system("rm "+source+'*.ply')

	for file_name in file_names:
		number_of_planes = 5
		if not file_name in exclusion_file_names:
			file_seq = file_name.split('.')[0].split('_')[-1]
			d = scipy.misc.imread(source+file_name).astype(np.float32) 
			seg = scipy.misc.imread(source+'sync_instances_'+file_seq+'.png').astype(np.float32) 
			if Save_image['aug_']:
				scipy.misc.imsave(source+'aug_'+file_seq+'.png', (d/np.max(d)*255.).astype(np.uint8))
			if Save_image['aug_instances_']:
				scipy.misc.imsave(source+'aug_instances_'+file_seq+'.png', (seg/np.max(seg)*255.).astype(np.uint8))
			
			print ("Working on ", file_name, "...")
			
			data = get_points (d, seg, n, k, th=250, f=f)
			if Save_image['all_dots_']:
				scipy.misc.imsave(source+'all_dots_'+file_seq+'.png', points_on_image(d, data, f=f, col=[255,255,0]))
			
			line_norms, Ms, selected_datas = find_lines(data, number_of_lines=number_of_planes, 
														method='ransac', plane_th=50, s=2)
			
			fgbg_planes = 0
			number_of_planes = len(line_norms)
			for i in range(number_of_planes):
				line_norm=line_norms[i]
				M=Ms[i]
				if np.linalg.norm(line_norm) == 0:
					line_norm = np.asarray([0, 1, 0])
				line_norm = line_norm / np.linalg.norm(line_norm)
				selected_data=selected_datas[i]
				if Save_image['cand_dots_']:
					scipy.misc.imsave(source+'cand_dots_'+file_seq+'_'+str(i+1)+'.png', points_on_image(d, selected_data, f=f))
				
				if Save_image['floor_']:
					floor = projection_on_floor(d, all_points=data, points=selected_data, line=[line_norm, M], f=f)
					scipy.misc.imsave(source+'floor_'+file_seq+'_'+str(i+1)+'.png', floor)
				
				if Save_image['mask_'] or Save_image['combined_mask_']:
					N = np.asarray([0, line_norm[2], -line_norm[1]])

					if Save_image['floor_combined_']:
						if i == 0 and not Save_image['floor_']:
							floor = projection_on_floor(d, all_points=data, points=None, line=[line_norm, M], f=f)
						floor = np.rot90(floor)
						floor = np.rot90(floor)
						floor = np.rot90(floor)
						H, W = d.shape
						z_max = int(np.max(d))
						z_min = int(np.min(d[d>0]))
						z_scale = (z_max-z_min)/W
						Norm = [line_norm[1], line_norm[2]]
						_, y = conv_3D_to_2D(M, H, W, f, 'float')
						Middle = [y, M[2]/z_scale-z_min]
						floor = add_line_on_floor (floor, Norm, Middle, col=[0,1,0])
						floor = np.rot90(floor)
					
					if np.linalg.norm(N) == 0:
						N = np.asarray([0, 0, -1])
					else:
						N = N / np.linalg.norm(N)
					N = np.reshape(N, [1,1,3])
					M = np.reshape(M, [1,1,3])
					offset = np.sum(-M*N)
					N = N / np.linalg.norm(N) * np.sign(-offset)
					offset *= np.sign(-offset)

					x = np.linspace(0, 1, d.shape[0])
					y = np.linspace(0, 1, d.shape[1])
					xv, yv = np.meshgrid(x, y, indexing='ij')
					xv = (xv[:, :, np.newaxis] - 0.5) * d.shape[0] / f 
					yv = (yv[:, :, np.newaxis] - 0.5) * d.shape[1] / f 
					X = np.concatenate((xv, yv, np.ones_like(xv)), axis=2)  
				
					Z = np.sum(N*X, axis=2) * d + offset
	
					D =  np.zeros_like(d)
					D [Z > 0] = 0.5
					D [Z < 0] = 1.0
					D [d < 1] = 0.0

					D_bit = np.zeros_like(d, dtype=np.uint8)
					D_bit [D == 0.5] = 7
					D_bit [D == 1.0] = 2 ** (i+3)

					fgbg_planes = np.bitwise_or(fgbg_planes, D_bit)
					
					if Save_image['mask_']:
						scipy.misc.imsave(source+'mask_'+file_seq+'_'+str(i+1)+'.png', (D*255).astype(np.uint8))
					if Save_image['3D_ply']:
						FG = np.sum(M*N, axis=2) / np.sum(N*X, axis=2)
						make_3D_ply(d, FG, selected_data, M, ply_name=source+'Z_'+file_seq+'.ply', color=None, f=f)
			
			if Save_image['combined_mask_']:
				import matplotlib
				matplotlib.image.imsave(source+'combined_mask_'+file_seq+'.png', fgbg_planes)
				
			if Save_image['occlusion_attention_']:
				occlusion_attention = get_occlusion_attention(d)
				scipy.misc.imsave(source+'occlusion_attention_'+file_seq+'.png', occlusion_attention)
			
			if Save_image['add_bob']:
				new_object = scipy.misc.imread("spongebob.png")
				image = scipy.misc.imread(source+'rgb_'+file_seq+'.jpg')
				ar_image = add_object(image, d, new_object)
				scipy.misc.imsave(source+'bob_'+file_seq+'.png', ar_image)

			if Save_image['floor_combined_']:
				scipy.misc.imsave(source+'floor_combined_'+file_seq+'.png', floor)
				
