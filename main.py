# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:31:04 2023

@author: Pranav
"""

import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from utils import eval_real,load_video
import tempfile

st.title('Action Recognition video spliter')
st.header('Please Upload a Video')

file = st.file_uploader('',type=['mp4'])
isresnet = st.button("Load Resnet Model")
iscnnlstm = st.button("Load CNN-LSTM Model")

label_data = pd.read_csv("classes.txt", sep = ' ', header = None)
label_data.columns = ['index','labels']
classes = label_data['labels']

if file is not None :
    video_bytes = file.read()
    st.video(video_bytes) 
    
    with tempfile.NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())

        model = load_model('activity_recognition.h5',compile = False)

        if iscnnlstm == True :
            model = load_model('activity_recognition2.h5',compile = False)

        images = load_video(f.name)
        
        edited_img = []
        count = 0
        for count in range(len(images[0]) - 15):
            imgs = images[0][count:count+15]
            edited_img.append(imgs)
            count += 1
        edited_imgs = np.array(edited_img)
        images_arr = np.array(images)

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        pred = []
        with st.spinner('Wait for it...'):
            for img in edited_imgs:
                img = img.reshape(1,15,64,64,3)
                prediction = eval_real(img,model)
                pred.append(prediction)
        st.success('Done!')

        names = []
        start = []
        end = []
        start.append(0)
        names.append(classes[pred[0]])
        for i in range(len(pred)-1):
            if pred[i] != pred[i+1] :
                names.append(classes[pred[i]])
                if len(end) != 0:
                    end.pop()
                end.append(i)
                start.append(i+1)
                end.append(i+1)
            else :
                if len(end) != 0:
                    end.pop()
                end.append(i)

        for j in range(len(end)):
            if j < len(names) :
                name = names[j]
            else: name = names[j-1]
            if start[j] - end[j] != 0:
                st.write('Video ',j,' Start: ',start[j],' End: ',end[j],name)
            else:
                st.write('Video',j,'Insignificant Change')