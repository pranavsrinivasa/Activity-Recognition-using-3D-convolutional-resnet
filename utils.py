import cv2
import numpy as np


def video_Frames(clip_path,img_size = 64):
    video = cv2.VideoCapture(clip_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for count in range(frame_count):
        flag, frame = video.read()
        if not flag:
            break
        frame = cv2.resize(frame,(img_size,img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        #normalizing the pixels between 0 and 1
        frame = frame/255.0
        yield frame 
    video.release()


def load_video(folder_path):
    imgs = []
    frames_generator = video_Frames(folder_path)
    frames_array = np.array(list(frames_generator))
    imgs.append(frames_array)
    real_imgs = np.array(imgs)

    return imgs


def eval_real(real_imgs, model):    
    pred1 = model.predict(real_imgs)
    pred1_max = pred1.argmax()

    return pred1_max