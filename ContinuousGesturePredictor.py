from pynput.keyboard import Key, Controller
import time

keyboard=Controller()


import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
from collections import Counter 
from populate import data as words
import ngrams
# global variables
# words=['HELLO','I','AM','A','BOY','ABHIRAM', 'HOW','ARE','WHO','ARE','YOU']
word_map=[]
bg = None
result='    '
r=''
flag=0
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    _, cnts, _ = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # initialize weight for running average
    aWeight = 0.5
    global result,words,word_map
    for word in words:
        word_map.append(Counter(word))
    # import pdb; pdb.set_trace()
    # get the reference to the webcam
    # camera = cv2.VideoCapture(0)

    #Get from IPcamera
    #hardcoded URL - 
    url='http://192.168.1.2:8080/shot.jpg'
    import requests

    # region of interest (ROI) coordinates
    top, right, bottom, left = 70, 390, 285, 610

    # initialize num of frames
    num_frames = 0
    start_recording = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        # (grabbed, frame) = camera.read()

        #getting image from IP
        img_resp=requests.get(url)
        img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
        frame=cv2.imdecode(img_arr,-1)

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 0, 0),thickness=2)
                if start_recording:
                    # import pdb; pdb.set_trace()
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thresholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            start_recording = True
        if keypress==ord('z'):
            result='    '

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(97, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]+prediction[0][2]+prediction[0][3]+prediction[0][4]+prediction[0][5]+prediction[0][6]+prediction[0][7]+prediction[0][8]+prediction[0][9]+prediction[0][10]+prediction[0][11]+prediction[0][12]+prediction[0][13]+prediction[0][14]+prediction[0][15]+prediction[0][16]+prediction[0][17]+prediction[0][18]+prediction[0][19]+prediction[0][20]+prediction[0][21]+prediction[0][22]+prediction[0][23]+prediction[0][24]+prediction[0][25]))

def showStatistics(predictedClass, confidence):
    global result,r,flag,word_map
    textImage = np.zeros((300,1500,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "A"
        keyboard.press(Key.left)
    elif predictedClass == 1:
        className = "B"
    elif predictedClass == 2:
        className = "C"
    elif predictedClass==3:
        className="D"
    elif predictedClass==4:
        className="E"
    elif predictedClass==5:
        className="F"
    elif predictedClass==6:
        className="G"
    elif predictedClass==7:
        className="H"
    elif predictedClass==8:
        className="I"
    elif predictedClass==9:
        className="J"
    elif predictedClass==10:
        className="K"
    elif predictedClass==11:
        className="L"
    elif predictedClass==12:
        className="M"
        keyboard.press(Key.right)
    elif predictedClass==13:
        className="N"
    elif predictedClass==14:
        className="O"
        keyboard.press(Key.down)
    elif predictedClass==15:
        className="P"
        keyboard.press(Key.up)
    elif predictedClass==16:
        className="Q"
    elif predictedClass==17:
        className="R"
    elif predictedClass==18:
        className="S"
    elif predictedClass==19:
        className="T"
    elif predictedClass==20:
        className="U"
    elif predictedClass==21:
        className="V"
    elif predictedClass==22:
        className="W"
    elif predictedClass==23:
        className="X"
    elif predictedClass==24:
        className="Y"
    elif predictedClass==25:
        className="Z"
    elif predictedClass==26:
        className=" "
        keyboard.press(Key.space)
        # keyboard.press(Key.space)
        # print("SPACE being presssed")
    if result=='    ':
        result+=className
        # print(result)
    else:
        if r=='':
            r=result[:-4]
        if className == r[len(r)-1] and className == r[len(r)-2] and className == r[len(r)-3] and className == r[len(r)-4]:
            # print(className,"    ",result[len(result)-1],confidence)
            if className!=result[len(result)-1] and confidence>0.95:
                result+=className
                flag=1
                if className == ' ':
                    t=result.split("_")
                    map2=Counter(t[len(t)-1])
                    print(map2)
                    dict2=dict(map2)
                    count=0
                    for map1 in word_map:
                        dict1=dict(map1)
                        if dict1.items()<=dict2.items():
                            x=result.split("_")
                            x[len(x)-1]=words[count]
                            sep=' '
                            result="    "+sep.join(x)+"_"
                            print(result)
                            break
                        count+=1
                    # import pdb; pdb.set_trace()

        else:
            r+=className
            # print("same being generated",confidence)
    if len(result)>50:
        result='    '
    
    cv2.putText(textImage, result, 
                (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (255, 255, 255),
                2)
    if flag==1:
        # for i in range(100000000):
        #     x=5
        flag=0
    # cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    # (30, 100), 
    # cv2.FONT_HERSHEY_SIMPLEX, 
    # 1,
    # (255, 255, 255),
    # 2)
    cv2.imshow("Statistics", textImage)




# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,97,100,1],name='input')
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=conv_2d(convnet,512,3,activation='relu')
convnet=conv_2d(convnet,512,3,activation='relu')
convnet=conv_2d(convnet,512,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=fully_connected(convnet,128,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,64,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,27,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("TrainedModel/newASL_VG_space.h5")

main()
