
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
import plane_finder


def get_gt_temp(image, args, inverse=True):
    # gt_path_abs = get_gt_path(image, args.dataset)
    gt_path_abs = image.replace('image_left', 'depth_left').replace('.png', '.dpt')
    gt = cv_io.read(gt_path_abs)
    gt_norm = 1 / gt if inverse else gt.copy()
    gt_norm[gt<1e-3] = 0
    m = np.min(gt_norm)
    M = np.max(gt_norm)
    gt_norm = (gt_norm - m) / (M - m) * 255
    # gt_norm[gt<1e-3] = 255 / 2
    gt_path = os.path.join('examples/tmp', 'gt_'+image.replace(os.sep,'_')).replace('.dpt', '.png')
    cv_io.save(gt_path, gt_norm.astype(np.uint8))
    return gt_path, gt


def draw_box(img, x, y, color=[255,0,0], thickness=3):
    th = thickness
    img[x-th:x+th,y-th:y+160+th,:] = color
    img[x+128-th:x+128+th,y-th:y+160+th,:] = color
    img[x-th:x+128+th,y-th:y+th,:] = color
    img[x-th:x+128+th,y+160-th:y+160+th,:] = color
    return img


def ROI_box(gt, image, h, w, M, thickness=3):
    rgb = cv_io.read(image)
    rgb = draw_box(rgb, h, w)    
    mask =  np.zeros_like(gt)
    mask [gt < M] = 0
    mask [gt > M] = 255
    mask [gt < 0.01] = 255//2
    mask_color = mask[:,:,np.newaxis]
    mask_color = np.concatenate((mask_color, np.zeros_like(mask_color), np.zeros_like(mask_color)), axis=2) 
    mask_color = 0.8 * rgb + 0.2 * mask_color
    temp_save = os.path.join('examples/tmp', 'mask_ROI_'+image.replace(os.sep,'_')).replace('.dpt', '.png')
    cv_io.save(temp_save, mask_color.astype(np.uint8))
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


def get_planes(gt, image, args, examples=4):
    gap_points = []
    th = 250
    while len(gap_points) < 8:
        gap_points = plane_finder.get_points (gt*1000, None, args.n, args.k, th=th, f=args.f)
        th = th / 2
    line_normals, Ms, selected_points = plane_finder.find_lines(gap_points, number_of_lines=args.number_of_planes, method='ransac', plane_th=50, s=2)
    img = cv_io.read(image)

    mask_paths = list()
    masks = list()
    
    for i in range(examples):
        line_normal = line_normals[i]
        M = Ms[i]
        N = np.asarray([0, line_normal[2], -line_normal[1]])
        if np.linalg.norm(N) == 0:
            N = np.asarray([0, 0, -1])
        else:
            N = N / np.linalg.norm(N)
        
        N = np.reshape(N, [1,1,3])
        M = np.reshape(M, [1,1,3])
        offset = np.sum(-M*N)
        N = N / np.linalg.norm(N) * np.sign(-offset)
        offset *= np.sign(-offset)
        
        x = np.linspace(0, 1, gt.shape[0])
        y = np.linspace(0, 1, gt.shape[1])
        xv, yv = np.meshgrid(x, y, indexing='ij')
        xv = (xv[:, :, np.newaxis] - 0.5) * gt.shape[0] / args.f 
        yv = (yv[:, :, np.newaxis] - 0.5) * gt.shape[1] / args.f 
        
        X = np.concatenate((xv, yv, np.ones_like(xv)), axis=2)  
        Z = np.sum(N*X, axis=2) * gt*1000 + offset
        mask =  np.zeros_like(gt)
        mask [Z > 0] = 0
        mask [Z < 0] = 255
        mask [gt < 0.01] = 255//2

        corner_flag = True
        i = 0
        best_score = 1.0
        while corner_flag:
            i += 1
            h_tmp = np.random.randint(Z.shape[0]-128)
            w_tmp = np.random.randint(Z.shape[1]-160) 
            ROI = Z[h_tmp:h_tmp+128, w_tmp:w_tmp+160]      
            score = np.abs(np.mean(ROI) / 255 - 0.5)
            if score < best_score:
                best_score = score
                h = h_tmp
                w = w_tmp
            if score < 0.25 or i > 100:
                corner_flag = False
                h = h_tmp
                w = w_tmp

        mask_color = 255 - mask[:,:,np.newaxis]
        mask_color = np.concatenate((np.zeros_like(mask_color), mask_color, np.zeros_like(mask_color)), axis=2) 
        mask_color = 0.8 * img + 0.2 * mask_color 
        mask_color = draw_box(mask_color, h, w, color=[0,128,0])

        mask_path = os.path.join('examples/tmp', 'mask_'+str(i)+'_'+image.replace(os.sep,'_')).replace('.dpt', '.png')
        cv_io.save(mask_path, mask_color.astype(np.uint8))

        mask_paths.append(mask_path)
        masks.append(mask)

    return mask_paths, masks


def find_ROI(depth, x=128, y=160, plane_th=500, p=0.999, s=2, verbose=False):
    H, W = depth.shape
    iteration = 0
    max_iteration = np.inf
    best_inliers = None
    best_score = 0

    while max_iteration > iteration or best_inliers is None:
        h = np.random.randint(150, H-x)
        w = np.random.randint(0, W-y)
        ROI = depth[h:h+x, w:w+y]
        ROI = ROI[ROI>0]
        M = np.median(ROI)

        diff = np.abs(np.log(ROI/M))
        # diff = diff[ROI>0]
        score = np.sum(np.logical_and(diff>np.log(1.2),diff<np.log(2)))
        if score > best_score:
            best_score = score
            best_inliers = [h, w, M]
        
        e = 1 - score / (x*y)
        if e == 0: e += 10e-5
        max_iteration = np.log(1-p) / np.log(1-(1-e)**s)
        iteration += 1
                
        if iteration % 1000 == 0 and verbose:
            print (iteration, max_iteration, 1-e)
            
    if verbose:
        print ("{:d}, {:5.1f}, {:5.2f}, ({},{}), {}".format(iteration, max_iteration, 1-e, h, w, M))

    return best_inliers
    

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description="An example of making automates result presentaions.", fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--data-path', default='/data/datasets/replica_ROI/', help="The path of dataset.")
# parser.add_argument('--gt-path', help="The path of GT images.")
parser.add_argument('--output-filename', default='ROI_search_Replica', help="The path of GT images.")
parser.add_argument('--dataset', default='replica', help="dataset name")
parser.add_argument('--title', default='Serch For ROI in Replica Dataset', help="Presentation title.")
parser.add_argument('--sub-title', default='Two Approaches', help="Presentation subtitle.")

                    
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

args.gt_path = os.path.join(args.data_path, 'depth_left')

sences = ['apartment_0', 'apartment_1', 'apartment_2', 
            'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_3', 'frl_apartment_4', 'frl_apartment_5',
            'hotel_0', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4',
            'room_0', 'room_1', 'room_2']

if __name__ == "__main__":

    images_path = os.path.join(args.data_path, 'image_left')
    images = []
    for scene in sences:
        images_temp = get_files(os.path.join(images_path, scene), ext='png')
        images += [os.path.join(images_path, scene, im) for im in images_temp]
    images.sort()
    # images = metric_sort(images, args)

    tex_file_name = os.path.join('examples', args.output_filename + '.tex')
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

    os.system('mkdir -p examples/logs')
    os.system('mkdir -p examples/tmp')

    items = ['Some random examples of two approaches to make planes.']
    items.append('Red plane is a random fronto-parallel plane.')
    items.append('Red rectangular is Region of Interest (ROI).')
    items.append('Green planes are made by ransac method.')
    items.append('Green rectangular is Region of Interest (ROI)')
    items.append('ROI boxes were randomly chosen until met criteria.')
    items.append('Python generator code is available on \\href{https://github.com/saeid-h/Latex-Slides}{Github}')
    intro_slide = Items(items)
    slide = Frame('Introduction', intro_slide) 
    presentation.add_frames(slide)

    args.f = 300; args.n = 50; args.k = 10; args.shift=1; args.number_of_planes = 5

    images = images[::150]

    for i, image in enumerate(images):
        title = '_'.join(image.split(os.sep)[-2:]).split('.')[0].replace('_', '\_') 
        print ('{}: {}'.format(i, title))
        scale = 0.15 if args.dataset == 'nyu' else 0.18 

        rgb_img = Graphics(scale, path=image)
        
        gt_path, gt = get_gt_temp(image, args)
        gt_img =  Graphics(scale, path=gt_path) 

        mask_path, mask = get_planes(gt, image, args)
        mask_img =  Graphics(scale, path=mask_path[0])

        h, w, M = find_ROI(gt)
        fronto_path = ROI_box(gt, image, h, w, M)
        fronto_parallel =  Graphics(scale, path=fronto_path)

        table_data = [[rgb_img, gt_img, fronto_parallel]]
        table_data += [['RGB', 'GT', 'Fronto Parallel Plane']]

        mask_img_0 =  Graphics(scale, path=mask_path[0])
        mask_img_1 =  Graphics(scale, path=mask_path[1])
        mask_img_2 =  Graphics(scale, path=mask_path[2])
        table_data += [[mask_img_0, mask_img_1, mask_img_2]]
        table_data += [['Old Plane 1', 'Old Plane 2', 'Old Plane 3']]

        
        
        table = Tables(header=None, adjustments=['c']*3, data=table_data)
        slide = Frame(title, table) 
        presentation.add_frames(slide)

    presentation.build_slides()
    with open(tex_file_name, 'w') as f:
        f.write(presentation.latex)

    os.system('pdflatex -output-directory=examples/logs ' + tex_file_name)
    os.system('mv examples/logs/*.pdf ./examples/')
    os.system('rm -rf examples/logs')
    os.system('rm -rf examples/tmp')
