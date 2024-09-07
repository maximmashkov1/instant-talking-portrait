import cv2
import torch
import sys
import os
class DepthModel:
    def __init__(self):
        depth_estimation_path = os.path.join(os.getcwd(), 'Depth-Anything-V2')
        sys.path.append(depth_estimation_path)
        from depth_anything_v2.dpt import DepthAnythingV2
        
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        encoder = 'vitb' 
        
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'./Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = model.to(DEVICE).eval()
        
        
    def estimate_depth(self, image):
        return self.model.infer_image(image)
import numpy as np
import matplotlib.pyplot as plt

class Skeleton:
    
    face_parts = np.array([
            (17, 18), (18, 19), (19, 20), (20, 21),  #right eyebrow
            (22, 23), (23, 24), (24, 25), (25, 26),  #left eyebrow
            (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  #right eye
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  #left eye
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # outer lips
            (60, 61), (61, 62), (62, 63), (63, 64), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60), #inner lips
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), #face outline
            (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (33, 34), (34, 35), #nose
            (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77),  (77, 78), (78, 79) #upper head
        
        ]).astype(np.int32)
        
    face_hole_tris = [
                    (60, 61, 67),
                    (61, 67, 62),
                    (62, 67, 66),
                    (62, 63, 65),
                    (62, 66, 65),
                    (63, 65, 64)]
    
    eye_hole_tris = [(36, 37, 41),  #triangles to use to cut out face holes
                               (37, 38,41), 
                               (38, 41, 40),
                               (38, 40, 39),
                               
                              (42, 43, 47),
                              (43, 47, 44),
                              (47, 44, 46),
                              (46, 44, 45),
    ]            
            
        

    def __init__(self, face_keypoints_2d):
        self.face_keypoints_2d = face_keypoints_2d.copy()

    
    def cut_out_face_holes(mask, face_keypoints_2d):

        mask_with_holes = mask.copy()
        
        for tri in Skeleton.face_hole_tris:
            pts = np.array([face_keypoints_2d[tri[0]],
                            face_keypoints_2d[tri[1]],
                            face_keypoints_2d[tri[2]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask_with_holes, [pts], 0)
        
        return mask_with_holes
    
    def cut_out_eye_holes(mask, face_keypoints_2d):

        mask_with_holes = mask.copy()
        
        for tri in Skeleton.eye_hole_tris:
            pts = np.array([face_keypoints_2d[tri[0]],
                            face_keypoints_2d[tri[1]],
                            face_keypoints_2d[tri[2]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask_with_holes, [pts], 0)
        
        return mask_with_holes

    def plot(self):
        plt.figure(figsize=(10, 10))

        for i, point in enumerate(self.face_keypoints_2d):
            x, y = point
            plt.scatter(x, y, c='g')
            plt.text(x, y, f"{i}", fontsize=8, color='purple')

            # Draw connections for face
            for start, end in self.face_parts:
                plt.plot([self.face_keypoints_2d[start][0], self.face_keypoints_2d[end][0]],
                         [self.face_keypoints_2d[start][1], self.face_keypoints_2d[end][1]], 'g-')

        plt.gca().invert_yaxis()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from poisson_disc import Bridson_sampling
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import triangle as tr

def poisson_disk_sampling(mask, radius):
    height, width = mask.shape
    dims = np.array([height, width]) 
    samples = Bridson_sampling(dims, radius)
    #samples = [(int(y), int(x)) for x, y in samples if mask[int(y), int(x)]]
    return np.array(samples)

def filter_triangles(points, mask, triangles, remove_face_if_has):
    valid_triangles = []
    for triangle in triangles:
        try:
            triangle_points = points[triangle]
    
            pixels = np.round(triangle_points).astype(int)
            
            centroid = np.mean(triangle_points, axis=0)
            centroid_idx = tuple(np.round(centroid).astype(int))
    
            if mask[centroid_idx] == 1 and not all([v in remove_face_if_has for v in triangle]) or any([v in list(range(1,17)) for v in triangle]):
                valid_triangles.append(triangle)
        except IndexError:
            print('INDEX ERROR')
    
    return np.array(valid_triangles)
    
def adjust_coordinates(points, mask_shape):
    height, width = mask_shape
    points = np.array([points[:, 1], points[:, 0]]).T
    points[:, 1] = height-points[:, 1]
    return points

def plot_verts(points):
    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(points[:,0],points[:,1])
    plt.show()

def remove_unused_points(points, valid_triangles):
    
    used_points = np.unique(valid_triangles.flatten())
    new_points = points[used_points]
    remap_indices = {old_index: new_index for new_index, old_index in enumerate(used_points)}
    new_triangles = np.vectorize(remap_indices.get)(valid_triangles)
    return new_points, new_triangles

def find_corner_and_edge_midpoint_indices(points, width, height):

    width, height = height, width
    top_left = np.array([0, 0])
    top_right = np.array([width, 0])
    bottom_left = np.array([0, height])
    bottom_right = np.array([width, height])
    
    mid_top = np.array([width / 2, 0])
    mid_bottom = np.array([width / 2, height])
    mid_left = np.array([0, height / 2])
    mid_right = np.array([width, height / 2])
    
    distances_top_left = np.linalg.norm(points - top_left, axis=1)
    distances_top_right = np.linalg.norm(points - top_right, axis=1)
    distances_bottom_left = np.linalg.norm(points - bottom_left, axis=1)
    distances_bottom_right = np.linalg.norm(points - bottom_right, axis=1)
    
    distances_mid_top = np.linalg.norm(points - mid_top, axis=1)
    distances_mid_bottom = np.linalg.norm(points - mid_bottom, axis=1)
    distances_mid_left = np.linalg.norm(points - mid_left, axis=1)
    distances_mid_right = np.linalg.norm(points - mid_right, axis=1)
    
    top_left_index = np.argmin(distances_top_left)
    top_right_index = np.argmin(distances_top_right)
    bottom_left_index = np.argmin(distances_bottom_left)
    bottom_right_index = np.argmin(distances_bottom_right)
    
    mid_top_index = np.argmin(distances_mid_top)
    mid_bottom_index = np.argmin(distances_mid_bottom)
    mid_left_index = np.argmin(distances_mid_left)
    mid_right_index = np.argmin(distances_mid_right)
    
    return [top_left_index, top_right_index, bottom_left_index, bottom_right_index,
            mid_top_index, mid_bottom_index, mid_left_index, mid_right_index]

    
def get_puppet_from_mask(mask, base_vertices=None, base_edges=None, cutout_faces=None, remove_face_if_has=None, radius=25, vertex_size=5, face=True):
    base_vertices=base_vertices.copy()

    #slice eyelids
    o_len = base_vertices.shape[0] 
    brow_support = base_vertices[list(range(17,27))] #last 10 indices
    brow_support[:,1]+=base_vertices[24,1]-base_vertices[69,1]
    added_vertices = base_vertices[[37,38,43,44]]
    added_vertices[:,1] += 0.01
    base_vertices = np.concatenate((base_vertices, added_vertices, brow_support), axis=0)
    base_edges = np.concatenate((base_edges, [(36, o_len), (o_len, o_len+1), (o_len+1, 39), 
                                                            (42,o_len+2), (o_len+2, o_len+3), (o_len+3, 45)]), axis=0)
    remove_face_if_has=[36,37,38,39,o_len,o_len+1,42,43,44,45,o_len+2,o_len+3]

    
    face = face and len(base_vertices) > 0
    
    mask_ = Skeleton.cut_out_face_holes(mask, base_vertices) if face else mask.copy()
    points = poisson_disk_sampling(mask_, radius)

    if face:
        base_vertices = base_vertices.copy()
        base_vertices = base_vertices[:, [1, 0]]
        
        points = poisson_disk_sampling(mask_, radius)
        valid_sampled_points = []
        for point in points:
            distances = [np.linalg.norm(point - orig_point) for orig_point in base_vertices]
            if min(distances) > radius * 0.5:
                valid_sampled_points.append(point)
    else:
        valid_sampled_points = points
        
    valid_sampled_points = np.array(valid_sampled_points)
    
    if face:
        all_points = np.vstack((base_vertices, valid_sampled_points))
        face_points = np.arange(len(base_vertices))
        
        triangulation_input = {
            'vertices': all_points,
            'segments': base_edges, 
        }
        tri = tr.triangulate(triangulation_input, 'pc')  
        triangles = tri['triangles']
    else:
        all_points = valid_sampled_points
        face_points = None
        delaunay=Delaunay(valid_sampled_points)
        triangles=delaunay.simplices

    
    max_height, max_width = mask_.shape
    all_points[:, 0] = np.clip(all_points[:, 0], 0, max_height-1)
    all_points[:, 1] = np.clip(all_points[:, 1], 0, max_width-1)

        
    valid_triangles = filter_triangles(all_points, mask_, triangles, remove_face_if_has)
    all_points, valid_triangles = remove_unused_points(all_points, valid_triangles)
    all_points = adjust_coordinates(all_points, mask_.shape)
    
   
    #plot_verts(all_points[np.array(remove_face_if_has)])
    
    all_points[[61,62,63]]=all_points[[67,66,65]] #close lips
    all_points[[o_len,o_len+1,o_len+2,o_len+3]][:,1] -= 1 #move eyelids back down

    return np.array(all_points, dtype=np.float32), np.array(valid_triangles, dtype=np.int32), face_points, \
    find_corner_and_edge_midpoint_indices(all_points, mask_.shape[1],  mask_.shape[0])
import importlib

from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget, QWidget, QStackedLayout, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QFileDialog, QLineEdit, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QTimer, QElapsedTimer, QPoint, QMimeData
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QColor, QBrush, QDragEnterEvent, QDropEvent


import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from numba import jit, njit, prange
from pwarp.core import ops
from pwarp.core.arap import StepOne, StepTwo
from pwarp.data.puppet import PuppetObject
#from pwarp.warp.warp import graph_warp
from time import time, sleep
from graph_warp_ import graph_warp
from collections import defaultdict

from face_inference import FaceKeypointModeling
from mouth_model.mouth_model import SpeechToLip
from face_model.diffusion import *
import threading 
import winsound
import shutil
import random
import scipy
from scipy.ndimage import gaussian_filter
import sounddevice as sd
from communication import receive_audio

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None, aspect_ratio=None, image=None, vertices=None, faces=None, depth=None, mouth_params=None):
        super(GLWidget, self).__init__(parent)

        self.mouth_params= np.array(mouth_params, dtype=np.float32)
        
        self.RENDER_GRID = 0
        self.SUBDIVISION_LEVEL = self.mouth_params[5].astype(np.int32).item()
        self.DEPTH_INFLUENCE_CLIP = 0.0
        self.DEPTH_POWER = 2
    
        
        
        self.aspect_ratio = aspect_ratio
        self.parent = parent
        self.image = image


        #teeth
        self.teeth_top = np.rot90(np.rot90(cv2.cvtColor(cv2.imread("assets/teeth_top.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), k=-1), k=-1)
        self.teeth_bottom = np.rot90(np.rot90(cv2.cvtColor(cv2.imread("assets/teeth_bottom.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), k=-1), k=-1)
        self.teeth_scale, self.teeth_t_adjustment, self.teeth_b_adjustment = self.fit_teeth(vertices.copy())

        
        self.teeth_t_position, self.teeth_b_position = None, None
        self.teeth_depth = -0.9 


        self.original_vertices = vertices.copy()
        self.original_faces = np.array(faces)
        
        self.vertices = self.original_vertices.copy()
        self.faces = self.original_faces.copy()


        
        #get depth for occlusion and displacement scaling
        self.depth = np.rot90(depth, k=-1)
        
        self.depth = (self.depth - np.min(self.depth)) / (np.max(self.depth) - np.min(self.depth))
        self.depth = np.clip(self.depth, self.DEPTH_INFLUENCE_CLIP, 1)
        self.depth = self.depth**self.DEPTH_POWER
        self.depth = self.depth * 2 - 1 
        

        original_sub_vertices, original_sub_faces = self.subdivision_2d(self.original_vertices, self.original_faces, self.SUBDIVISION_LEVEL)
        self.original_sub_vertices = np.floor(original_sub_vertices-1).clip(min=0).astype(np.int32)
                                                                         
        self.sub_vertex_depth = self.depth[self.original_sub_vertices[:, 0], self.original_sub_vertices[:, 1]][:, np.newaxis]
        self.sub_vertex_disp_scaling = self.sub_vertex_depth 
        

        self.update_set_vertices(self.vertices)


        #setup texture coords
        self.texCoords = self.vertices[:, :2] / np.array([image.shape[1], image.shape[0]])
        self.texCoords[:, 1] = 1 - self.texCoords[:, 1] 
        print(self.texCoords)
        self.indices = np.hstack(self.faces)
        self.textureID=None
        self.teeth_t_textureID=None
        self.teeth_b_textureID=None

    def fit_teeth(self, vertices):
        scale = self.mouth_params[1] * np.linalg.norm(vertices[48]-vertices[54])/440
        teeth_t =  vertices[62]-vertices[51]-self.teeth_top.shape[0]*scale*0.58 + self.mouth_params[2]
        teeth_b = vertices[66]-vertices[8]-self.teeth_bottom.shape[0]*scale*1.15 + self.mouth_params[3]
        teeth_t[0]=0
        teeth_b[0]=0
        return scale, teeth_t, teeth_b
        
        
    def update_set_vertices(self, vertices):
        t=time()
        
        #update teeth
        self.teeth_t_position  = vertices[51] + self.teeth_t_adjustment
        self.teeth_b_position = vertices[8] +  self.teeth_b_adjustment
        self.teeth_brightness = np.clip(np.linalg.norm(vertices[62]-vertices[66])/(self.teeth_scale*150), 0, 1)

        
        #update vertices        
        self.vertices, self.faces = self.subdivision_2d(vertices, self.original_faces, self.SUBDIVISION_LEVEL)

        depth_scaled_displacement = (self.vertices-self.original_sub_vertices)
        self.vertices += self.sub_vertex_disp_scaling * depth_scaled_displacement 
        self.vertices = np.hstack([self.vertices, self.sub_vertex_depth]).astype(np.float32)
        self.vertices[:, 2] = self.vertices[:, 2]*0.1 + 0.5 #squeeze

        
        
    def initializeGL(self):
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
    
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image.shape[1], self.image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, self.image)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.image.shape[1], 0, self.image.shape[0])
        glMatrixMode(GL_MODELVIEW)

        self.teeth_t_textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.teeth_t_textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.teeth_top.shape[1], self.teeth_top.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, self.teeth_top)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        self.teeth_b_textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.teeth_b_textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.teeth_bottom.shape[1], self.teeth_bottom.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, self.teeth_bottom)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #background
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glBegin(GL_QUADS)

        glTexCoord2f(0.0, 0.0)
        glVertex3f(0.0, self.image.shape[0], -0.95)  
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(self.image.shape[1], self.image.shape[0], -0.95)
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(self.image.shape[1], 0.0, -0.95) 
        
        glTexCoord2f(0.0, 1.0)
        glVertex3f(0.0, 0.0, -0.95) 
        
                
        glEnd()
     
        #top teeth
        teeth_brightness = glColor3f(self.teeth_brightness,self.teeth_brightness,self.teeth_brightness)

        
        glBindTexture(GL_TEXTURE_2D, self.teeth_t_textureID)
        glBegin(GL_QUADS)
        
        x_pos, y_pos = self.teeth_t_position
        x_size = self.teeth_top.shape[1]*self.teeth_scale
        y_size = self.teeth_top.shape[0]*self.teeth_scale
        x_pos-=x_size*0.5
        z_pos = self.teeth_depth 
        
        glTexCoord2f(0.0, 0.0)
        glVertex3f(x_pos, y_pos, z_pos) 
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(x_pos + x_size, y_pos, z_pos)
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(x_pos + x_size, y_pos + y_size, z_pos)
        
        glTexCoord2f(0.0, 1.0)
        glVertex3f(x_pos, y_pos + y_size, z_pos)
        
        glEnd()

        #bot teeth
        glBindTexture(GL_TEXTURE_2D, self.teeth_b_textureID)

        glBegin(GL_QUADS)

        x_pos, y_pos = self.teeth_b_position
        x_size = self.teeth_bottom.shape[1] * self.teeth_scale
        y_size = self.teeth_bottom.shape[0] * self.teeth_scale 
        x_pos -= x_size * 0.5
        z_pos = self.teeth_depth + 0.1
        
        glTexCoord2f(0.0, 0.0)
        glVertex3f(x_pos, y_pos, z_pos)  
        
        glTexCoord2f(1.0, 0.0)
        glVertex3f(x_pos + x_size, y_pos, z_pos)
        
        glTexCoord2f(1.0, 1.0)
        glVertex3f(x_pos + x_size, y_pos + y_size, z_pos)
        
        glTexCoord2f(0.0, 1.0)
        glVertex3f(x_pos, y_pos + y_size, z_pos)
        
            
        glEnd()

        #face mesh
        glColor3f(1,1,1)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glTexCoordPointer(2, GL_FLOAT, 0, self.texCoords)

        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)

        #grid
        if self.RENDER_GRID:
            glDisable(GL_TEXTURE_2D)  
            glColor3f(4., 4., 4.)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  
    
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, self.indices)
    
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_TEXTURE_2D)  
    
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        error = glGetError()
        if error != GL_NO_ERROR:
            print("OpenGL Error:", error)
        
        
                
        

    def cleanupGL(self):
        glDeleteTextures([self.textureID])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    
    def updateVertices(self, new_vertices):
        self.update_set_vertices(new_vertices)
        self.update()

    def subdivision_2d(self, vertices, faces, iter=1):
        vertices = np.array(vertices)
        for _ in range(iter):
            edge_midpoints = {}
            new_faces = []
            num_vertices = len(vertices)
            
            # Create new vertices at the midpoints of each edge and store them
            for face in faces:
                edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
                for edge in edges:
                    sorted_edge = tuple(sorted(edge))
                    if sorted_edge not in edge_midpoints:
                        midpoint = (vertices[sorted_edge[0]] + vertices[sorted_edge[1]]) / 2
                        vertices = np.vstack([vertices, midpoint])
                        edge_midpoints[sorted_edge] = num_vertices
                        num_vertices += 1
            
            # Replace each triangle with four smaller triangles
            for face in faces:
                new_vertices = [edge_midpoints[tuple(sorted((face[i], face[(i + 1) % 3])))] for i in range(3)]
                new_faces.extend([
                    [face[0], new_vertices[0], new_vertices[2]],
                    [new_vertices[0], face[1], new_vertices[1]],
                    [new_vertices[2], new_vertices[1], face[2]],
                    [new_vertices[0], new_vertices[1], new_vertices[2]]
                ])
            
            faces = np.array(new_faces)
        
        return vertices, faces


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDragEnterEvent, QDropEvent


class DragDropWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True) 
        self.initUI()
        self.image=None
        self.full=None
        self.parent=parent
    
    def initUI(self):
        layout = QVBoxLayout()
        self.label = QLabel("Drop an image or a saved file")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.handleFile(file_path)
            event.acceptProposedAction()

    def handleFile(self, file_path):
        
        if any([file_path.lower().endswith(ext) for ext in ['jpg', 'jpeg', 'png']]):
            self.image = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            
        elif file_path.endswith('.npz'):
            with np.load(fileName) as data:
                image = data['arr_0']
                depth = data['arr_1']
                base_vertices = data['arr_2']
                mouth_params = data['arr_3']
                self.full = [image, depth, base_vertices, mouth_params]
                self.parent.file_name = os.path.splitext(os.path.basename(fileName))[0]
        else:
            self.label.setText(f"{file_path} is not an image or an .npz file")
            return
        self.label.setText("Processing...")

def numpy_to_qimage(np_array):
    height, width = np_array.shape
    np_array_normalized = ((np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array)) * 255).astype(np.uint8)
    qimage = QImage(np_array_normalized.data, width, height, width, QImage.Format_Grayscale8)
    return qimage

def qimage_to_numpy(qimage):
    width = qimage.width()
    height = qimage.height()
    bytes_per_line = qimage.bytesPerLine()
    ptr = qimage.bits()
    ptr.setsize(height * bytes_per_line) 
    np_array = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line))
    return np_array[:, :width]

def draw_lines_on_qimage(qimage, coordinates, color=QColor(255, 255, 255), thickness=5):
    painter = QPainter(qimage)
    painter.setPen(color)
    pen = painter.pen()
    painter.setRenderHint(QPainter.Antialiasing)
    pen.setWidth(thickness)
    painter.setPen(pen)
    for start, end in coordinates:
        start_point = QPoint(*start)
        end_point = QPoint(*end)
        painter.drawLine(start_point, end_point)
    painter.end()


def make_parameter_element(cw, name, default, function, type_=float):
    layout = QHBoxLayout(cw)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    label = QLabel(name)
    textbox = QLineEdit()
    textbox.setText(str(type_(default)))
    textbox.textChanged.connect(function)
    textbox.setFixedWidth(40)
    layout.addStretch(1)
    layout.addWidget(label, 0, alignment=Qt.AlignRight)  
    layout.addWidget(textbox, 0, alignment=Qt.AlignRight)  
    return layout

def stretchy_bar():
    bar = QWidget()
    bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    bar.setMinimumHeight(3)  
    bar.setStyleSheet("background-color: darkgray;")
    return bar
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

class MainWindow(QMainWindow):
    def __init__(self, app):
        super(MainWindow, self).__init__()
        
        self.setWindowTitle("Instant talking portrait")
        
        self.app = app
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        self.centralWidget = centralWidget
        self.layout = QHBoxLayout(centralWidget)
        self.depth_model = None
        self.file_name='new character'
        self.stylesheet="""
            QMainWindow {
                background-color: #2b2b2b; 
                color: #d3d3d3; 
            }
            QLabel {
                color: #d3d3d3; 
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #d3d3d3; 
                border: 2px solid #5a5a5a; 
                padding: 5px;
            }
            QCheckBox{
                color: #d3d3d3;
            }
            QPushButton:hover {
                background-color: #4b4b4b; 
            }
        """
        
        self.setStyleSheet(self.stylesheet)
        
        self.wait_for_drop()
        
    def wait_for_drop(self):
     
        self.dragdrop = DragDropWidget(self)
        #self.dragdrop.setStyleSheet(self.stylesheet)
        
        self.layout.addWidget(self.dragdrop)
        self.image=None
        self.image_label = None
        self.selected_vertex = None
        self.glWidget = None
        self.fps = None
        self.load_button = QPushButton("Load", self)
        self.load_button.clicked.connect(self.load_from_file)
        self.load_button.setFixedWidth(100) 
        self.layout.addWidget(self.load_button)
    
        self.setMinimumSize(0,0)
        self.setGeometry(300, 100, 500, 300)
        self.show()

    
        def check_drop():
            if not (self.dragdrop.image is None and self.dragdrop.full is None and self.image is None):
                timer.stop()
                if self.dragdrop.image is not None:
                    self.setup_window(self.dragdrop.image)
                elif self.dragdrop.full is not None:
                    self.setup_window(self.dragdrop.full[0], self.dragdrop.full[1], self.dragdrop.full[2], self.dragdrop.full[3])
                else: #if image loaded with button
                    pass

        timer = QTimer(self)
        timer.timeout.connect(check_drop)
        timer.start(500)
        
    def setup_window(self, image, depth=None, base_vertices=None, mouth_params=None, load_default=False):
        self.clear_layout(self.layout)
        
        self.image=image
        if depth is None:
            self.depth_model=DepthModel() if self.depth_model is None else self.depth_model
            self.depth = self.depth_model.estimate_depth(image)
        else:
            self.depth = depth
            
        if base_vertices is None:
            self.base_vertices = np.load('assets/base_face.npy') * self.image.shape[0]
            while self.base_vertices.max(axis=0)[0] > self.image.shape[1] or self.base_vertices.max(axis=0)[1] > self.image.shape[0]:
                self.base_vertices *= 0.9
        else:
            self.base_vertices = base_vertices

        self.mouth_params = [1.0, 1.0, 0, 0, 1.0, 0, 1.0, 0] if mouth_params is None else mouth_params
        self.fps=30 if self.fps is None else self.fps
        
        #IMAGE
        height, width, channel = self.image.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qImg)
        self.glWidget=None
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)  # Allow QLabel to scale while maintaining the aspect ratio
        self.image_label.setPixmap(self.pixmap)
        self.layout.addWidget(self.image_label)

        #BUTTONS
        
        self.spacer = QSpacerItem(20, 40)
        

        
        self.button_region_width = 280
        button_width = int(self.button_region_width*0.45)
        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(5)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.save_layout = QHBoxLayout(self.centralWidget)
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_to_file)
        self.save_button.setFixedWidth(button_width) 


        self.file_name_field = QLineEdit()
        self.file_name_field.setText(self.file_name)
        self.file_name_field.textChanged.connect(self.file_name_update)

        self.save_layout.addWidget(self.file_name_field, 0)  
        self.save_layout.addWidget(self.save_button, 0)  
        
        self.load_button = QPushButton("Load", self)
        self.load_button.clicked.connect(self.load_from_file)
        self.load_button.setFixedWidth(button_width) 

        
        self.checkbox = QCheckBox("Move all")
        self.checkbox.stateChanged.connect(self.update_move_all)

        scale_vertices_label = QLabel('Scale vertices')
        self.vertex_scale_layout = QHBoxLayout(self.centralWidget)
        self.vertex_scale_layout.addStretch(1)
        plus = QPushButton("+", self)
        plus.clicked.connect(self.vertex_scale_up)
        plus.setFixedWidth(40) 
        minus = QPushButton("-", self)
        minus.clicked.connect(self.vertex_scale_down)
        minus.setFixedWidth(40) 
        self.vertex_scale_layout.addWidget(scale_vertices_label)
        self.vertex_scale_layout.addWidget(minus)
        self.vertex_scale_layout.addWidget(plus)
        
        self.load_default = QPushButton("Reset all", self)
        self.load_default.clicked.connect(self.load_default_vertices)

        
        self.fps_layout = make_parameter_element(cw=self.centralWidget, name='fps: ', default=self.fps, function=self.fps_update, type_=int)
        self.lip_scale_layout = make_parameter_element(cw=self.centralWidget, name='Lip animation scale: ', default=self.mouth_params[0], function=self.lip_scale_update)
        self.teeth_scale_layout =  make_parameter_element(cw=self.centralWidget, name='Teeth scale: ', default=self.mouth_params[1], function=self.teeth_scale_update)
        self.top_teeth_disp_layout =  make_parameter_element(cw=self.centralWidget, name='Top teeth shift: ', default=self.mouth_params[2], function=self.top_teeth_update, type_=int)
        self.bot_teeth_disp_layout =  make_parameter_element(cw=self.centralWidget, name='Bottom teeth shift: ', default=self.mouth_params[3], function=self.bottom_teeth_update, type_=int)
        self.head_movement_scale_layout =  make_parameter_element(cw=self.centralWidget, name='Head movement scale: ', default=self.mouth_params[6], function=self.head_movement_scale_update)
        self.eyefix_checkbox = QCheckBox("Fix broken eyes")
        self.eyefix_checkbox.setChecked(bool(self.mouth_params[7]))
        self.eyefix_checkbox.stateChanged.connect(self.eyefix_update)
        
        self.mesh_res_scale_layout =  make_parameter_element(cw=self.centralWidget, name='Mesh density scale: ', default=self.mouth_params[4], function=self.mesh_density_update)
        self.sub_level_layout =  make_parameter_element(cw=self.centralWidget, name='Subdivision level: ', default=self.mouth_params[5], function=self.subdivision_update, type_=int)
        
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.main_app)
        self.start_button.setFixedWidth(button_width)  

        #############################################
        self.layout.addLayout(self.button_layout)
        self.button_layout.addLayout(self.save_layout)
        self.button_layout.addWidget(self.load_button, 0, alignment=Qt.AlignTop | Qt.AlignRight)
        
        self.button_layout.addItem(self.spacer)
        self.button_layout.addWidget(stretchy_bar())
        self.button_layout.addItem(self.spacer)
        self.button_layout.addWidget(self.checkbox, 0, alignment=Qt.AlignTop | Qt.AlignRight)
        self.button_layout.addLayout(self.vertex_scale_layout)

        self.button_layout.addItem(self.spacer)
        self.button_layout.addWidget(stretchy_bar())
        self.button_layout.addItem(self.spacer)
        
        
        
        self.button_layout.addLayout(self.lip_scale_layout)
        self.button_layout.addLayout(self.teeth_scale_layout)
        self.button_layout.addLayout(self.top_teeth_disp_layout)
        self.button_layout.addLayout(self.bot_teeth_disp_layout)
        self.button_layout.addLayout(self.head_movement_scale_layout)
        self.button_layout.addWidget(self.eyefix_checkbox, 0, alignment=Qt.AlignTop | Qt.AlignRight)
        self.button_layout.addItem(self.spacer)
        self.button_layout.addLayout(self.mesh_res_scale_layout)
        self.button_layout.addLayout(self.sub_level_layout)
        self.button_layout.addLayout(self.fps_layout)
        self.button_layout.addWidget(self.start_button, 0, alignment=Qt.AlignTop | Qt.AlignRight)  
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.load_default, 0, alignment=Qt.AlignBottom | Qt.AlignRight)
        
        self.selected_vertex = None
        self.move_all_vertices = False
        aspect_ratio = self.image.shape[1] / self.image.shape[0]  # width / height
        initial_height = 1024
        initial_width = int(initial_height * aspect_ratio)

        self.setGeometry(300, 100, initial_width+self.button_region_width, initial_height)

        self.setMinimumSize(500, 500)
        self.trigger_resize_event()
    
    def eyefix_update(self, state):
        self.mouth_params[7] = int(state == Qt.Checked)
        
    def head_movement_scale_update(self,text):
        try:
            self.mouth_params[6]=float(text)
        except:
            self.mouth_params[6]=1
            
    def subdivision_update(self,text):
        try:
            self.mouth_params[5]=int(text)
        except:
            self.mouth_params[5]=0

    def mesh_density_update(self,text):
        try:
            self.mouth_params[4]=float(text)
        except:
            self.mouth_params[4]=1.
         
    def vertex_scale_down(self):
        midpoint = self.base_vertices[30]
        for i, v in enumerate(self.base_vertices):
            self.base_vertices[i] = v*0.95 + midpoint*0.05
        self.update()
        
    def vertex_scale_up(self):
        midpoint = self.base_vertices[30]
        for i, v in enumerate(self.base_vertices):
            self.base_vertices[i] = v*1.05 - midpoint*0.05
        self.update()
        
    def load_default_vertices(self):
        self.file_name='new character'
        self.clear_layout(self.layout)
        self.wait_for_drop()
        
    def fps_update(self, text):
        try:
            self.fps=int(text)
        except:
            self.fps=30
    def lip_scale_update(self,text):
        try:
            self.mouth_params[0]=float(text)
        except:
            self.mouth_params[0]=1.
    def teeth_scale_update(self,text):
        try:
            self.mouth_params[1]=float(text)
        except:
            self.mouth_params[1]=1.

    def top_teeth_update(self,text):
        try:
            self.mouth_params[2]=int(text)
        except:
            self.mouth_params[2]=0

    def bottom_teeth_update(self,text):
        try:
            self.mouth_params[3]=int(text)
        except:
            self.mouth_params[3]=0
        
    def file_name_update(self, text):
        self.file_name=text
        
    def load_from_file(self):
        fileName, filter = QFileDialog.getOpenFileName(self, 'Open file', 
            './saved_faces', '(*.npz *.png *.jpg *.bmp)')
        if fileName.endswith('.npz'):
            print(fileName)
            with np.load(fileName) as data:
                image = data['arr_0']
                depth = data['arr_1']
                base_vertices = data['arr_2']
                mouth_params = data['arr_3']
            self.file_name = os.path.splitext(os.path.basename(fileName))[0]
            self.setup_window(image, depth, base_vertices, mouth_params)
            
        elif any([fileName.endswith(ext) for ext in ['jpg', 'jpeg', 'png']]):
            image = cv2.cvtColor(cv2.imread(fileName, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.setup_window(image)
                
    def save_to_file(self):
        np.savez('./saved_faces/'+self.file_name+'.npz', self.image, self.depth, self.base_vertices, np.array(self.mouth_params))
        
            
    def update_move_all(self, state):
        self.move_all_vertices=(state == Qt.Checked)
    
   
    
    def paintEvent(self, event):
        if self.image_label is None:return
        try:
            qImg = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qImg)
        
            painter = QPainter(self.pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            vertex_size = int(self.image.shape[0]/300)
            
            pen = QPen(QColor(255, 0, 0), 2)
            pen_special = QPen(QColor(255, 255, 0), 2)
            pen_lines = QPen(QColor(0, 255, 0), vertex_size/2)
            brush = QBrush(QColor(255, 0, 0))
            painter.setBrush(brush)
            painter.setPen(pen_lines)
            for edge in Skeleton.face_parts:
                painter.drawLine(QPointF(self.base_vertices[edge[0],0], self.base_vertices[edge[0],1]), 
                                 QPointF(self.base_vertices[edge[1],0], self.base_vertices[edge[1],1]))

            
            painter.setPen(pen)
            
            
            for vertex in self.base_vertices:
                painter.drawEllipse(QPointF(vertex[0], vertex[1]), vertex_size, vertex_size)
            painter.setPen(pen_special)
            for vertex in self.base_vertices[[8,51,57,62,66,13,3,33]]:
                painter.drawEllipse(QPointF(vertex[0], vertex[1]), vertex_size, vertex_size)
            painter.end()
        
            self.image_label.setPixmap(self.pixmap)
        except:
            pass
            
    def mousePressEvent(self, event):

        if self.image_label is None: #gl -> construction
            try:
                self.close_socket=True
                self.timer.stop()
                self.setup_window(self.image, self.depth, mouth_params=self.mouth_params, base_vertices=self.base_vertices)
                return
            except Exception as e:
                print(e)
                return
                
        pos = event.pos() - self.image_label.pos()
        scale = self.image.shape[0] / self.image_label.height()
        pos = QPointF(pos.x() * scale, pos.y() * scale)
        vertex_size = int(self.image.shape[0]/300)*1.2
        L2_norm = lambda point: (point.x()**2 + point.y()**2)**0.5
        for i, vertex in enumerate(self.base_vertices):
            if L2_norm(QPointF(vertex[0], vertex[1]) - pos)< vertex_size:
                self.selected_vertex = i
                break
    
    def mouseMoveEvent(self, event):
        
        if self.image_label is None:return
            
        if self.selected_vertex is None:return
            
        if self.move_all_vertices:
            pos = event.pos() - self.image_label.pos()
            scale = self.image.shape[0] / self.image_label.height()
            displacement = [pos.x() * scale, pos.y() * scale]-self.base_vertices[self.selected_vertex]
            for i, vertex in enumerate(self.base_vertices):
                self.base_vertices[i] += displacement
            
        else:
            pos = event.pos() - self.image_label.pos()
            scale = self.image.shape[0] / self.image_label.height()
            pos = QPointF(pos.x() * scale, pos.y() * scale)
    
            self.base_vertices[self.selected_vertex] = [pos.x(), pos.y()]
            self.update()


    def mouseReleaseEvent(self, event):
        self.selected_vertex = None
        
    def clear_layout(self, layout):
        if not layout:
            return
        
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self.clear_layout(child.layout())
        
        self.image_label = None
        self.update()
        
    def trigger_resize_event(self):
        current_size = self.size()
        self.resize(current_size.width() + 1, current_size.height() + 1)
        self.resize(current_size.width(), current_size.height())

    
        
    def precompute_stuff(self):
        self.clear_layout(self.layout)
        
        self.skeleton = Skeleton(self.base_vertices)
        
        #scale depth
        chin_end=int(self.skeleton.face_keypoints_2d[8,1].astype(np.int32)+1)
        blend_to=0.4
        blend_over=100
        if chin_end+blend_over >self.depth.shape[0]:
            blend_over = self.depth.shape[0]-chin_end-1
        depth=self.depth.copy()
        depth[chin_end:chin_end+blend_over]*=np.linspace(1,blend_to,blend_over).reshape(blend_over, 1)
        depth[chin_end+blend_over:] *= blend_to

        #fix eye depth
        depth/=depth.max()
        if self.mouth_params[7]:
            eye_mask = Skeleton.cut_out_eye_holes(np.ones_like(depth), self.skeleton.face_keypoints_2d)
            eye_mask = gaussian_filter(eye_mask, 20)
            eye_mask[eye_mask>0.9] =1
            eye_mask = np.clip(eye_mask, 0.75, 1)
            depth = (depth+depth*eye_mask)/2

        mask=np.ones_like(depth)
        self.vertices, self.faces, self.face_keypoints, self.corner_vertices = get_puppet_from_mask(mask, radius=(1/self.mouth_params[4])*self.image.shape[0]*35/1024, base_vertices=self.skeleton.face_keypoints_2d, base_edges=np.array(Skeleton.face_parts), cutout_faces=self.skeleton.face_hole_tris, face=True)
        self.puppet = PuppetObject(self.vertices, self.faces, None, None)
        self.edges = ops.get_edges(len(self.faces), self.faces)
        self.gi, self.g_product = StepOne.compute_g_matrix(self.vertices, self.edges, self.faces)
        self.h = StepOne.compute_h_matrix(self.edges, self.g_product, self.gi, self.vertices)


        int_vertices = self.vertices.astype(np.int32)
        
        self.glWidget = GLWidget(self, image=self.image, vertices=self.vertices.copy(), faces=self.faces.copy(), depth=depth, mouth_params=self.mouth_params)
        self.layout.addWidget(self.glWidget)
        self.trigger_resize_event()
        
        self.face_model = FaceKeypointModeling(ddim=False)
        self.mouth_morph_targets = self.face_model.mouth_morph_targets
        self.mouth_morph_mean = self.compute_mouth_morph(self.face_model.mouth_morph_mean)
        self.mouth_morph_meshes = [self.compute_mouth_morph(morph) for morph in self.mouth_morph_targets] 
        self.morph_horizontal, self.morph_vertical = self.compute_face_morph()
        self.blink_morph = self.compute_blink_morph()
        
        #------------------------------------------------------
        #blinking
        self.blink_schedule = np.array([])
        blink_rate=0.6
        step=0.3
        length=100
        blink_array = np.append(np.linspace(0, 1, int(self.fps*step/2)), np.linspace(1, 0, int(self.fps*step/2))) *1.1
        min_interval=int(1/step)
        last_blink=0
        for i in range(int(length/step)):
            if last_blink+min_interval < i and random.uniform(0.,1.) < blink_rate*step:
                self.blink_schedule = np.append(self.blink_schedule, scipy.signal.resample(blink_array,
                                                                                           int(blink_array.shape[0]*random.uniform(0.85, 1.15))))
                last_blink=i
            else:
                self.blink_schedule = np.append(self.blink_schedule, np.zeros_like(blink_array))
        self.blink_schedule = np.append(self.blink_schedule, np.flip(self.blink_schedule))
        self.blink_schedule = gaussian_filter(self.blink_schedule,1)
        
        #------------------------------------------------------
        #idle face coordinates
        idle_length=5000
        scale=30/self.fps
        restraint=0.01*scale
        self.idle_face_coordinates = np.zeros((2,idle_length))
        is_stopped=False
        stop_rate=2
        
        for i in range(1, idle_length):
            coeff = 0.5 if is_stopped else 1
            self.idle_face_coordinates[0, i] = self.idle_face_coordinates[0, i-1]+coeff*np.random.normal(loc=-restraint* self.idle_face_coordinates[0, i-1], scale=scale)
            self.idle_face_coordinates[1, i] = self.idle_face_coordinates[1, i-1]+coeff*np.random.normal(loc=-restraint* self.idle_face_coordinates[1, i-1], scale=scale)

            if not is_stopped:
                is_stopped= random.random() < stop_rate/self.fps
            else:
                is_stopped= not random.random() < stop_rate*2/self.fps
                
        self.idle_face_coordinates= self.idle_face_coordinates.transpose(1,0)*0.003
        self.idle_face_coordinates= np.concatenate((self.idle_face_coordinates, np.flip(self.idle_face_coordinates)),axis=0)
        self.idle_face_coordinates[:,0], self.idle_face_coordinates[:,1] = gaussian_filter(self.idle_face_coordinates[:,0], 12), gaussian_filter(self.idle_face_coordinates[:,1], 12)
        

    def serve_audio(self):
        self.close_socket=False
        for audio, sr in receive_audio(parent=self):
            t=time()
            animation, head_disp = self.face_model.inference(audio, sr, self.fps, face_smoothing=2)
            print('Generation time: ', time()-t)


            #pad_len = int(self.fps*0.2)
            #zero_padding1 = np.zeros((pad_len, animation.shape[-1]))
            #zero_padding2 = np.zeros((pad_len, head_disp.shape[-1]))
            
            sd.play(audio, sr)
            #self.animation, self.head_disp = np.concatenate((zero_padding1, animation), axis=0), np.concatenate((zero_padding2, head_disp), axis=0)
            self.animation, self.head_disp = animation, head_disp
            self.animation_index = 0
            self.anim_start_index = self.global_index
            self.smoothing_arrangement = self.last_arrangement
            self.smoothing_stage = 1-1/self.smooth_over
            sd.wait()
            
    def main_app(self):
        self.precompute_stuff()
        self.animation_index = -1
        self.anim_start_index = 0
        self.global_index = 0
        self.blink_index = 0
        self.face_coord_index = 0
        
        threading.Thread(target=self.serve_audio).start()
        self.start_animation_tick()
        
    
    def start_animation_tick(self):
        self.last_arrangement = None
        self.smoothing_stage = 0
        self.smooth_over = self.fps*0.7
        
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(int(1000 / self.fps))  
        
    def update_animation(self):
        elapsed_ms = self.elapsed_timer.elapsed()
        self.global_index = int(elapsed_ms * self.fps / 1000)
        
        if self.animation_index != -1:
            
            self.animation_index = self.global_index - self.anim_start_index 
            if self.animation_index >= len(self.animation):
                self.smoothing_arrangement = self.last_arrangement
                self.animation_index = -1
                self.smoothing_stage = 1-1/self.smooth_over
                

        #base
        result = self.puppet.r.copy()

        #update blink
        if self.blink_index+1 > self.blink_schedule.shape[0]:
            self.blink_index = 0
        result += self.blink_morph*self.blink_schedule[self.blink_index]
        self.blink_index+=1


        
        if self.animation_index != -1:
            #update mouth and conditioned rotation
            result += self.mouth_morph_mean* (0.7* self.mouth_params[0])
            for j, blend in enumerate(self.animation[self.animation_index]):
                result += self.mouth_morph_meshes[j] * (blend * 0.825*self.mouth_params[0])
        
            result += self.head_disp[self.animation_index, 0]*(self.morph_horizontal*5*self.mouth_params[6])
            result += self.head_disp[self.animation_index, 1]*(self.morph_vertical*5*self.mouth_params[6])
        else:
            #update random rotation
            if self.face_coord_index+1 > self.idle_face_coordinates.shape[0]:
                self.face_coord_index=0
            result += self.idle_face_coordinates[self.face_coord_index, 0]*self.morph_horizontal*5*self.mouth_params[6]
            result += self.idle_face_coordinates[self.face_coord_index, 1]*self.morph_vertical*5*self.mouth_params[6]
            self.face_coord_index+=1
            
        
        #transition smoothing
        if self.smoothing_stage > 0:
            result = result*(1-self.smoothing_stage)+self.smoothing_arrangement*self.smoothing_stage
            self.smoothing_stage -= 1/self.smooth_over
            
        self.last_arrangement = result
        self.glWidget.updateVertices(result)

    def compute_blink_morph(self):

        target_vertices=self.puppet.r.copy()
        control_pts = list(range(0,17)) + list(range(27,45)) + list(range(68,80)) + list(range(84, 94)) + self.corner_vertices
        disp_vector = (target_vertices[[37,38,43,44]]-target_vertices[[41,40,47,46]])
        target_vertices[[41,40,47,46]] += disp_vector*0.2
        target_vertices[[37,38,43,44]] = target_vertices[[41,40,47,46]]

        target_vertices=target_vertices[control_pts]
        closed_eyes = graph_warp(
            vertices=self.puppet.r,
            faces=None,
            control_indices=control_pts,
            shifted_locations=target_vertices,
            edges=self.edges,
            precomputed=(self.gi,self.g_product,self.h)
        )
        
        closed_eyes -= self.puppet.r
        return closed_eyes
        

    def compute_face_morph(self):

        #sideways 
        target_vertices=self.puppet.r.copy()
        control_pts = list(range(27,31)) + [51,62,66,57,8] + self.corner_vertices

        influence = np.linalg.norm(target_vertices[48]-target_vertices[54])*0.5
        
        target_vertices[control_pts, 0] += influence
        target_vertices=target_vertices[control_pts]
        morph_horizontal = graph_warp(
            vertices=self.puppet.r,
            faces=None,
            control_indices=control_pts,
            shifted_locations=target_vertices,
            edges=self.edges,
            precomputed=(self.gi,self.g_product,self.h)
        )
        
        morph_horizontal -= self.puppet.r

        
        target_vertices=self.puppet.r.copy()
        control_pts =list(range(0,27)) #[2,14] +list(range(31,36))
        target_vertices[control_pts, 1] += influence
        control_pts+=self.corner_vertices
        target_vertices=target_vertices[control_pts]
        morph_vertical = graph_warp(
            vertices=self.puppet.r,
            faces=None,
            control_indices=control_pts,
            shifted_locations=target_vertices,
            edges=self.edges,
            precomputed=(self.gi,self.g_product,self.h)
        )
        
        morph_vertical -= self.puppet.r
        return morph_horizontal, morph_vertical
        
    

    def compute_mouth_morph(self, displacement):
        #displacement = np.flip(displacement)
        target_vertices=self.puppet.r.copy()
        
        displacement*=np.linalg.norm(target_vertices[48]-target_vertices[54])*7
        
        jaw = displacement[-1]
        lips = displacement[:-1].reshape(12, 2)
        control_pts = [0,1,2,4,7,8,9,11,12,14,15,16]+list(range(48, 68)) + list(range(17, 26)) + list(range(27, 31)) + list(range(36, 48)) +list(range(70,80)) + [68,69, 51] +self.corner_vertices
        

        jaw_scale = 2

        
        target_vertices[8, 1]+=jaw*jaw_scale
        target_vertices[[7,9], 1]+=jaw*jaw_scale*0.9
        
        right = lips.copy()
        right[:,0]*=-1
        
        target_vertices[48]+=lips[0]
        target_vertices[49]+=lips[1]
        target_vertices[50]+=lips[2]
        target_vertices[51]+=lips[3]
        
        target_vertices[52]+=right[2]
        target_vertices[53]+=right[1]
        target_vertices[54]+=right[0]
        target_vertices[55]+=right[6]
        target_vertices[56]+=right[5]
        
        target_vertices[57]+=lips[4]
        target_vertices[58]+=lips[5]
        target_vertices[59]+=lips[6]
        target_vertices[60]+=lips[7]
        target_vertices[67]+=lips[8]
        target_vertices[66]+=lips[9]
        
        target_vertices[65]+=right[8]
        target_vertices[64]+=right[7]
        target_vertices[63]+=right[11]
        
        target_vertices[62]+=lips[10]
        target_vertices[61]+=lips[11]
        
        target_vertices = target_vertices[control_pts]
        
        morph = graph_warp(
            vertices=self.puppet.r,
            faces=None,
            control_indices=control_pts,
            shifted_locations=target_vertices,
            edges=self.edges,
            precomputed=(self.gi,self.g_product,self.h)
        )
        
        morph -= self.puppet.r
        
        return morph
        
    
    def resizeEvent(self, event):
        
        if self.image is None: return
            
        new_width = self.width()
        new_height = self.height()

        aspect_ratio = self.image.shape[1] / self.image.shape[0]

        if new_width / new_height > aspect_ratio:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)
        if self.image_label is not None:
            self.image_label.setFixedSize(new_width, new_height)
        if self.glWidget is not None:
            self.glWidget.setFixedSize(new_width, new_height)
        self.update()
    
            

        super(MainWindow, self).resizeEvent(event)
    

    def closeEvent(self, event):
            self.app.quit() 

if __name__ == '__main__':
    app = QApplication.instance()  
    if app: 
        app.quit()

    app = QApplication([])  
    window = MainWindow(app)
    window.show()
    
    sys.exit(app.exec_()) 