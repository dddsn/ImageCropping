import argparse
import numpy as np
import cv2
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
parser.add_argument('--weight_path', default='weights/gan_weight.h5', type=str)
parser.add_argument('--img_path', default='images', type=str)
parser.add_argument("--save_path", default='result', type=str)
parser.add_argument('--log', default=True, type=bool)
class crop_GAN():

    def __init__(self,weight_path,img_path,save_path,log):
    
        self.img_height = 224
        self.img_width = 224
        self.num_channels = 3
        self.input_shape = (self.img_height, self.img_width, self.num_channels)
        input = Input(self.input_shape)
        self.draw = True
        self.log = log
        self.base = img_path
        self.root = save_path
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
            img_path = file.strip()#os.path.join(self.base,file.strip())
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
    
    def turn_affine_to_loc(self,res,img_size):#
        x1 = (max(0.0,(1.0+res[2]-res[0])/2))*img_size[1]
        x2 = (min(1.0,(1.0+res[2]+res[0])/2))*img_size[1]
        y1 = (max(0.0,(1.0+res[5]-res[4])/2))*img_size[0]
        y2 = (min(1.0,(1.0+res[5]+res[4])/2))*img_size[0]
        return x1,x2,y1,y2

    def Run(self):
        
        img_path  = [os.path.join(self.base,path) for path in os.listdir(self.base)]
        imgs,imgs_size = self.get_img_input(img_path)
        affine = self.generator.predict([imgs,imgs_size])
        crop = []
        i = 0
        for e in affine:
            x1,x2,y1,y2 = self.turn_affine_to_loc(e,imgs_size[i])
            crop.append([x1,x2,y1,y2])
            i += 1
        
        if self.draw:
            for idx in range(len(crop)):
                img = cv2.imread(img_path[idx])
                img_name = img_path[idx].split('/')[-1]
                x1,x2,y1,y2 = crop[idx]
                cv2.rectangle(img, (int(x1+3),int(y1+3)),(int(x2-3),int(y2-3)), (0,180,255), 6)
                cv2.imwrite(os.path.join(self.root,img_name),img)
            
        if self.log:
            log_file = open(os.path.join(self.root,'log.txt'),'w')
            for idx in range(len(crop)):
                x1,x2,y1,y2 = crop[idx]
                log_file.write(img_path[idx]+'   '+str(x1)+'   '+str(x2)+'   '+str(y1)+'   '+str(y2)+'\n')
            log_file.close()
        return crop          
        
if __name__ == '__main__':
    args = parser.parse_args()
    s = crop_GAN(args.weight_path,args.img_path,args.save_path,args.log)
    s.Run()