
#############################################################################
#																			#
# Copyright (c) 2020 Saeid Hosseinipoor <https://saeid-h.github.io/>		#
# All rights reserved.														#
# Licensed under the MIT License											#
#																			#
############################################################################# 

from latex_slides.classes import *
import os, sys
import cv_io
import matplotlib.pyplot as plt 
import numpy as np
import argparse


def get_gt_temp(image, args, inverse=True):
    gt_path_abs = get_gt_path(image, args.dataset)
    gt = cv_io.read(gt_path_abs)
    gt_norm = 1 / gt if inverse else gt.copy()
    gt_norm[gt<1e-3] = 0
    m = np.min(gt_norm)
    M = np.max(gt_norm)
    gt_norm = (gt_norm - m) / (M - m) * 255
    # gt_norm[gt<1e-3] = 0
    gt_path = os.path.join('tmp', 'gt_'+image).replace('.dpt', '.png')
    cv_io.save(gt_path, gt_norm.astype(np.uint8))
    return gt_path


def get_dif_map(est_path, gt_path, tol=0.2):
    est = cv_io.read(est_path)
    gt = cv_io.read(gt_path)
    m = np.min(gt)
    M = np.max(gt)
    gt_nromal = (gt - m) / (M - m) * 255 
    R = gt_nromal.copy()
    G = gt_nromal.copy()
    B = gt_nromal.copy()
    R [np.abs(est-gt)>tol] = 255
    G [np.abs(est-gt)>tol] = 0
    B [np.abs(est-gt)>tol] = 0
    R [gt < 1e-3] = 0
    G [gt < 1e-3] = 0
    B [gt < 1e-3] = 0
    error = np.stack([R,G,B],-1)
    cv_io.save('tmp/x.png', error.astype(np.uint8))
    return 'tmp/x.png'
    

def ROI_box(rgb_file, thickness=3):
    rgb = cv_io.read(rgb_file)
    th = thickness
    rgb[192-th:192+th,192-th:192+160+th,:] = [0,255,0]
    rgb[192+128-th:192+128+th,192-th:192+160+th,:] = [0,255,0]
    rgb[192-th:192+128+th,192-th:192+th,:] = [0,255,0]
    rgb[192-th:192+128+th,192+160-th:192+160+th,:] = [0,255,0]
    temp_save = os.path.join ('tmp', 'rgb_'+rgb_file.split(os.sep)[-1])
    cv_io.save(temp_save, rgb)
    return temp_save

def get_files(path, ext='png', filename_only=True):
    file_list = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.'+ext.lower()):
                if filename_only:
                    file_list.append(file)
                else:
                    file_list.append(os.path.join(root, file))
    return file_list

def get_gt_path(image_filename, dataset='replica'):
    if dataset.lower() == 'replica':
        gt_name = '_'.join(image_filename.split('_')[-4:]).replace('.png', '.dpt')
        gt_folder = '_'.join(image_filename.split('_')[:-4])
        gt_path = os.path.join(args.gt_path, gt_folder, gt_name)
    else:
        gt_name = '_'.join(image_filename.split('_')[-2:]).replace('rgb_', 'sync_depth_')
        gt_folder = '_'.join(image_filename.split('_')[:-2])
        gt_path = os.path.join(args.gt_path, gt_folder, gt_name)
    return gt_path


def score(image_filename, metric, args):
    mask_final_path = os.path.join(args.data_path, 'occ_mask_final', image_filename)
    mask_gt_path = os.path.join(args.data_path, 'occ_mask_gt', image_filename)

    gt_path = get_gt_path(image_filename, args.dataset)
    gt_depth = cv_io.read(gt_path)[192:192+128,192:192+160]
    
    final_soft = cv_io.read(mask_final_path).astype(np.float32) / 255.
    if len(final_soft.shape) > 2: final_soft = final_soft[...,0]
    final = np.zeros_like(final_soft)
    final[final_soft>0.5] = 1.0

    gt = cv_io.read(mask_gt_path).astype(np.float32) / 255.
    if len(gt.shape) > 2: gt = gt[...,0]

    final = final[gt_depth>0.1]
    gt = gt[gt_depth>0.1]
    final_DR = np.sum(final!=gt) / np.prod(gt.shape)

    return final_DR

def metric_sort(image_list, args, metric='DR', reverse=False):
    tup = [(image, score(image, metric, args)) for image in image_list]
    tup.sort(key = lambda x: x[1], reverse=reverse)
    return tup

parser = argparse.ArgumentParser(description="An example of making automates result presentaions.")
parser.add_argument('--data-path', help="The path of results.")
parser.add_argument('--gt-path', help="The path of GT images.")
parser.add_argument('--output-filename', help="The path of GT images.")
parser.add_argument('--dataset', default='replica', help="dataset name")
parser.add_argument('--title', default='Qualitative Results', help="Presentation title.")
parser.add_argument('--sub-title', default='Model 1', help="Presentation subtitle.")

if sys.argv.__len__() == 2:
    arg_list = list()
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
    for line in lines:
        arg_list += line.strip().split()
    args = parser.parse_args(arg_list)
else:
    args = parser.parse_args()
                    


if __name__ == "__main__":

    images_path = os.path.join(args.data_path, 'cmap')
    images = get_files(images_path, ext='png')
    images = metric_sort(images, args)

    tex_file_name = args.output_filename + '.tex'
    more_setup = '\\hypersetup{\ncolorlinks=true,\nlinkcolor=blue,\nfilecolor=magenta,\nurlcolor=cyan,\n}\n\\urlstyle{same}'

    presentation = LatexSlideDocument(title=args.title,
                                        subtitle=args.sub_title,
                                        institute='Stevens Institute of Technology',
                                        author='Saeid Hosseinipoor',
                                        date='\\today', 
                                        notes=None,
                                        packages=['utopia', 'hyperref'],
                                        theme = 'Boadilla',
                                        usecolortheme='default',
                                        more_setup = more_setup,
                                        TOC=False)

    os.system('mkdir -p logs')
    os.system('mkdir -p tmp')

    items = ['Examples are sorted based on the DR metric']
    items.append('Green rectangular is Region of Interest (ROI)')
    items.append('Final mask is a soft mask')
    items.append("Slide's title is combination of the image name and its DR values")
    items.append('Python generator code is available on \\href{https://github.com/saeid-h/Latex-Slides}{Github}')
    intro_slide = Items(items)
    slide = Frame('Introduction', intro_slide) 
    presentation.add_frames(slide)

    for i, image_tuple in enumerate(images):
        image, score = image_tuple
        title = image.split(os.sep)[-1].split('.')[0].replace('_', '\_') + ' ::: ' + str(round(score*100,2)) + '\%'
        scale = 0.18
        cmap_path = os.path.join(args.data_path, 'cmap', image).replace('_', '\string_')
        cmap_img = Graphics(scale, path=cmap_path)

        rgb_path = os.path.join(args.data_path, 'rgb', image)
        rgb_path = ROI_box(rgb_path).replace('_', '\string_')
        rgb_img = Graphics(scale, path=rgb_path)
        
        gt_path = get_gt_temp(image, args)
        gt_img =  Graphics(scale, path=gt_path) 

        table_data = [[rgb_img, gt_img, cmap_img]]
        table_data += [['RGB', 'GT', 'Depth']]

        scale =0.58

        mask_init_path = cmap_path.replace('cmap', 'occ_mask_init')
        mask_init_img = Graphics(scale, path=mask_init_path)

        mask_gt_path = cmap_path.replace('cmap', 'occ_mask_gt')
        mask_gt_img = Graphics(scale, path=mask_gt_path)
        
        mask_final_path = cmap_path.replace('cmap', 'occ_mask_final')
        mask_final_img = Graphics(scale, path=mask_final_path)

        table_data += [[mask_init_img, mask_gt_img, mask_final_img]]
        table_data += [['Input Mask', 'GT Mask', 'Final Mask']]
        
        table = Tables(header=None, adjustments=['c']*3, data=table_data)
        slide = Frame(title, table) 
        presentation.add_frames(slide)

    presentation.build_slides()
    with open(tex_file_name, 'w') as f:
        f.write(presentation.latex)

    os.system('pdflatex -output-directory=logs ' + tex_file_name)
    os.system('mv logs/*.pdf ./')
    os.system('rm -rf logs')
    os.system('rm -rf tmp')
