import argparse
import numpy as np
import cv2
import scipy.io as scio
from models import *
from models.TransformGenerator import transform_generator
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense,Flatten, Dropout, Lambda
from keras.layers import Activation, GlobalAveragePooling2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD,RMSprop
import keras.backend as K
import keras
import tensorflow as tf
import os
import gc
import sys
from PIL import Image
from keras.applications import vgg16
from layers import BilinearInterpolation
import random
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--weight_path', default='weights/gan_weight.h5', type=str)
parser.add_argument('--log_path', default='gan_log.txt', type=str)

class crop_GAN():

    def __init__(self,dataset_path,weight_path,log_path):
    
        self.img_height = 224
        self.img_width = 224
        self.num_channels = 3
        self.dataset_path = dataset_path
        self.log_path = log_path
        self.input_shape = (self.img_height, self.img_width, self.num_channels)
        input = Input(self.input_shape)
        self.generator = self.build_generator()
        self.generator.load_weights(weight_path)
        self.generator.summary()


    def read_file(self, file):
        if not os.path.isfile(file):
            raise ValueError("Train File Not Exist.")
        else:
            with open(file) as f:
                lines = f.readlines()
            return np.array(lines)
            
    def get_img_input(self, file_list):
        r=[]
        s = []
        for file in file_list:
            img_path = file.strip()
            image = cv2.imread(img_path, 1)
            h,w,c = image.shape   
            s.append([h,w])
            image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
            image = np.array(image) / 127.5 - 1.0
            r.append(image)
        r = np.array(r)
        s = np.array(s,dtype='float32')
        return r,s
        
    def build_generator(self):
        img_size = Input(shape=(2,),name='size_input')
        model = transform_generator()
        model = Model(model.input,model.output,name='model_2')
        image = model.input
        crop_data = model(image)
        return Model([image,img_size],crop_data)
    
    #convert affine to normalized image coordinates
    def convert_affine_to_loc(self,res):
        x1 = (max(0.0,(1.0+res[2]-res[0])/2))
        x2 = (min(1.0,(1.0+res[2]+res[0])/2))
        y1 = (max(0.0,(1.0+res[5]-res[4])/2))
        y2 = (min(1.0,(1.0+res[5]+res[4])/2))
        return x1,x2,y1,y2
    
    def union(self,au, bu, area_intersection):
        area_a = (au[1] - au[0]) * (au[3] - au[2])
        area_b = (bu[3] - bu[2]) * (bu[1] - bu[0])
        area_union = area_a + area_b - area_intersection
        return area_union

    def intersection(self,ai, bi):
        x = max(ai[0], bi[2])
        y = max(ai[2], bi[0])
        w = min(ai[1], bi[3]) - x
        h = min(ai[3], bi[1]) - y
        if w < 0 or h < 0:
            return 0
        return w*h

    def caculate_iou(self,a, b, dataset):
        if dataset == 'flms':
            tmp = b[2]
            b[2] = b[1]
            b[1] = tmp
            
        a = np.float32(a)
        b = np.float32(b)
        area_i = self.intersection(a, b)
        area_u = self.union(a, b, area_i)
        iou= (area_i) / (area_u + 1e-6)
        return iou
    

    def Eval(self):
        # ---------------------
        #  Test fcd
        # ---------------------
        fcd_label = open(os.path.join(self.dataset_path,'FCD/label/test.txt'),'r').readlines()
        label_map = {}
        test_crop = []
        test_path = []
        for line in fcd_label:
            line = line.strip()
            tokens = line.split(' ')
            test_path.append(os.path.join(self.dataset_path,'FCD/test',tokens[0]))
            test_crop.append([float(tokens[1]),float(tokens[2]),float(tokens[3]),float(tokens[4])])
            
        test_path = np.array(test_path)
        test_crop = np.array(test_crop,dtype='float')
        test_img,test_img_size = self.get_img_input(test_path)
        result = self.generator.predict([test_img,test_img_size])
            
        test_size = []
        for path in test_path:
            img = Image.open(path)
            test_size.append(img.size)
        test_size = np.array(test_size)
            
        crop = []
        mm=0
        for res in result:
            x1,x2,y1,y2 = self.convert_affine_to_loc(res)
            crop.append([x1,x2,y1,y2])
            mm+=1
        fcd_pred_bx = np.array(crop,dtype=float)
            
        iou = 0.0
        bde = 0.0
        for i in range(fcd_pred_bx.shape[0]):
            iou += self.caculate_iou(fcd_pred_bx[i],test_crop[i], 'fcd')
            bde += np.sum(abs([fcd_pred_bx[i][2],fcd_pred_bx[i][3],fcd_pred_bx[i][0],fcd_pred_bx[i][1]]-test_crop[i])) / 4.0
        fcd_iou = float(iou)/fcd_pred_bx.shape[0]
        fcd_bde = float(bde)/fcd_pred_bx.shape[0]
        
        # ---------------------
        #  Test flms
        # ---------------------
        
        flms_path = os.path.join(self.dataset_path,'FLMS/500_image_dataset.mat')
        flms_data = scio.loadmat(flms_path)
        test_path = []
        for i in range(flms_data['img_gt'].shape[0]):
            test_path.append(os.path.join(self.dataset_path,'FLMS/image/',flms_data['img_gt'][i][0][0][0]))
        test_path = np.array(test_path)
            
        test_size = []
        for path in test_path:
            img = Image.open(path)
            test_size.append(img.size)
        test_size = np.array(test_size)
    
        test_img,test_img_size = self.get_img_input(test_path)
        result = self.generator.predict([test_img,test_img_size])
        
            
        crop = []
        mm=0
        for res in result:
            x1,x2,y1,y2 = self.convert_affine_to_loc(res)
            crop.append([x1,x2,y1,y2])
            mm+=1
        flms_pred_bx = np.array(crop,dtype=float)
        iou = 0.0
        pred_num = 0.0
        bde = 0.0
        for i in range(flms_pred_bx.shape[0]):
            res_iou = []
            res_bde = []
            w,h = test_size[i]
            for j in range(flms_data['img_gt'][i][0][1].shape[0]):
                pred_num_iou = self.caculate_iou(flms_pred_bx[i],flms_data['img_gt'][i][0][1][j]/[h,w,h,w],'flms')
                pred_num_bde = np.sum(abs([flms_pred_bx[i][2],flms_pred_bx[i][0],flms_pred_bx[i][3],flms_pred_bx[i][1]]-flms_data['img_gt'][i][0][1][j]/[h,w,h,w])) / 4.0
                if pred_num_iou <= 1.0:
                    res_iou.append(pred_num_iou)
                if pred_num_bde <= 1.0 and pred_num_bde > 0.0:
                    res_bde.append(pred_num_bde)
            bde += min(res_bde)
            iou += max(res_iou)
            
        flms_iou = float(iou)/flms_pred_bx.shape[0]
        flms_bde = float(bde)/flms_pred_bx.shape[0]
        
            
        # ---------------------
        #  Test cuhk-icd
        # ---------------------
        cuhk_path = os.path.join(self.dataset_path,'CUHK-ICD/cropping.txt')
        raw = open(cuhk_path,'r')
        lines = raw.readlines()
        raw.close()
        num = len(lines)
        test_path = []
        crops=[]
        for i in range(int(num/4)):
            test_path.append(os.path.join(self.dataset_path,'CUHK-ICD/images',lines[i*4][:-1]))
            crop = np.zeros((3,4))
            for j in range(1,4):
                tokens = lines[i*4+j][:-1].split(' ')
                crop[j-1][0] = float(tokens[0])
                crop[j-1][1] = float(tokens[1])
                crop[j-1][2] = float(tokens[2])
                crop[j-1][3] = float(tokens[3])
            crops.append(crop)
        test_path = np.array(test_path)
        crops=np.array(crops,dtype=float)
            
        test_size = []
        for path in test_path:
            img = Image.open(path)
            test_size.append(img.size)
        test_size = np.array(test_size)
    
        test_img,test_img_size = self.get_img_input(test_path)
        result = self.generator.predict([test_img,test_img_size])
            
        crop = []
        mm=0
        for res in result:
            x1,x2,y1,y2 = self.convert_affine_to_loc(res)
            crop.append([x1,x2,y1,y2])
            mm+=1
        cuhk_pred_bx = np.array(crop,dtype=float)
        
            
        iou = np.zeros(3)
        bde = np.zeros(3)
        for i in range(cuhk_pred_bx.shape[0]):
            res_iou = np.zeros(3)
            res_bde = np.zeros(3)
            w,h = test_size[i]
            for j in range(3):
                res_iou[j] = self.caculate_iou(cuhk_pred_bx[i],crops[i][j]/[h,h,w,w],'cuhk')
                res_bde[j] = np.sum(abs([cuhk_pred_bx[i][2],cuhk_pred_bx[i][3],cuhk_pred_bx[i][0],cuhk_pred_bx[i][1]]-(crops[i][j]/[h,h,w,w]))) / 4.0
            iou += res_iou
            bde += res_bde
            
        cuhk_iou = iou/cuhk_pred_bx.shape[0]
        cuhk_bde = bde/cuhk_pred_bx.shape[0]
           
        print ("[fcd iou : %.3f , flms iou : %.3f , cuhk-icd iou : %.3f  %.3f  %.3f]" % (fcd_iou,flms_iou,cuhk_iou[0],cuhk_iou[1],cuhk_iou[2]))
        print ("[fcd bde : %.3f , flms bde : %.3f , cuhk-icd bde : %.3f  %.3f  %.3f]" % (fcd_bde,flms_bde,cuhk_bde[0],cuhk_bde[1],cuhk_bde[2]))
        log_file = open(self.log_path,'w')
        log_file.write("[fcd iou : %.3f , flms iou : %.3f , cuhk-icd iou : %.3f  %.3f  %.3f]\n" % (fcd_iou,flms_iou,cuhk_iou[0],cuhk_iou[1],cuhk_iou[2]))
        log_file.write("[fcd bde : %.3f , flms bde : %.3f , cuhk-icd bde : %.3f  %.3f  %.3f]\n" % (fcd_bde,flms_bde,cuhk_bde[0],cuhk_bde[1],cuhk_bde[2]))
        log_file.close()
        
if __name__ == '__main__':
    args = parser.parse_args()
    s = crop_GAN(args.dataset_path,args.weight_path,args.log_path)
    s.Eval()