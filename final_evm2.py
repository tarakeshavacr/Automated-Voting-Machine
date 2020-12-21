from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import time
import numpy as np
import facenet
import serial
import detect_face
import os
import time
import pickle
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0

def video1():
    cap = cv2.VideoCapture(0)
    while True:                                                        
        ret, frame = cap.read()                            #Video capturing
        cv2.imwrite('check1.jpg', frame)
        cv2.imshow('window1',frame)
        if cv2.waitKey(1) == ord('q'):
            break

def finger(result_names):
    #print(result_names)
    global count0
    global count1
    global count2
    global count3
    global count4
    global count5
    global count6
    original = cv2.imread("finger.bmp")
    image_to_compare = cv2.imread(str(result_names)+".bmp")
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)                #fingerprint comparison
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("This fingerprint image is not clear")
        else:
            print("This fingerprint image is clear")
    
     
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
         
        matches = flann.knnMatch(desc_1, desc_2, k=2)
        good_points = []
        ratio = 0.6                                  #comparing fingerprints
        #serial_port.write(b'3')
        for m, n in matches:
            if m.distance < ratio*n.distance:
                good_points.append(m)
        if(len(good_points)>1):
            print("fingerprint matched")
            if result_names=='Rohit saw':
                if(count0<1):
                    
                    print("you are valid for voting")
                    
                    serial_port.write(b'1')
                    time.sleep(0.5)
                    serial_port.write(b'3')
                    
                    
                    count0=count0+1
                else:
                    print("Voter blocked")
            if result_names=='Tarakeshava':
                if(count1<1):
                    
                    print("you are valid for voting")
                    
                    serial_port.write(b'1')
                    time.sleep(0.5)
                    serial_port.write(b'3')
                    
                    
                    count1=count1+1
                else:
                    print("Voter Blocked")
            if result_names=='Rohan':
                if(count2<1):
                    
                    print("you are valid for voting")
                    
                    serial_port.write(b'2')
                    time.sleep(0.5)
                    serial_port.write(b'3')
                    
                    
                    count2=count2+1
                else:
                    print("Voter Blocked")
            if result_names=='Sandeep':
                if(count3<1):
                    
                    print("you are valid for voting")
                    
                    serial_port.write(b'2')
                    time.sleep(0.5)
                    serial_port.write(b'3')
                    
                    
                    count3=count3+1
                else:
                    print("you are not valid for voting")        
        else:
            
            print("Voter Blocked")
         
        result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
        cv2.imshow("result", result)
        cv2.imshow("Original", original)
        cv2.imshow("Duplicate", image_to_compare)
        cv2.waitKey(0)
        
img_path='check1.jpg'
modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
serial_port='com14'# serial port 
serial_port = serial.Serial(serial_port, 9600, timeout=1)
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading feature extraction model')
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        # video_capture = cv2.VideoCapture("akshay_mov.mp4")
        c = 0
        while True:
            print('Start Recognition!')
            os.system('fm.exe')
            video1()
            prevTime = 0
            # ret, frame = video_capture.read()
            frame = cv2.imread(img_path,0)

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Face Detected: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        print(predictions)                               # CNN
                        best_class_indices = np.argmax(predictions, axis=1)
                        # print(best_class_indices)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print(best_class_probabilities)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                        #plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        print('Result Indices: ', best_class_indices[0])
                        print(HumanNames)
                        for H_i in HumanNames:
                            # print(H_i)
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                                print(result_names)
                                finger(result_names)
                else:
                    print('Unable to align')
                    print('image is not clear')
                    
            cv2.imshow('Image', frame)
            if cv2.waitKey(1)& 0xFF==ord('q'):
                break
            cv2.destroyAllWindows()


