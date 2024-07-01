# Program to detect face boxes and display them on the video stream

# import libraries
import sys
import face_recognition
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2, Preview
import libcamera
import pygame
from bluedot import BlueDot
from gpiozero import Robot
from time import sleep
import RPi.GPIO as GPIO


def move(pos):
    if pos.top:
        robot.forward()
        sleep(0.1)
    elif pos.bottom:
        robot.backward()
        sleep(0.1)
    elif pos.right:
        robot.right()
        sleep(0.1)
    elif pos.left:
        robot.left()
        sleep(0.1)
             
        
def stop():
    robot.stop()

# Function to get cropped face image
def face_image_crop(frame, boxes):
    cropped_image_numpy_list = []
    if boxes:
        for box in boxes:
            top, right, bottom, left = box
            ver_distance = abs(top-bottom)
            latt_distance = abs(left-right)
            train_top = top-int(0.9*ver_distance)
            if train_top < 0:
                train_top = 0
            train_bottom = bottom+int(0.3*ver_distance)
            if train_bottom > 718:
                train_bottom = 718
            train_left = left-int(0.3*latt_distance)
            if train_left < 0:
                train_left = 0            
            train_right = right+int(0.3*latt_distance)
            if train_right > 1079:
                train_right = 1079

            # Crop the image
            cropped_image_numpy = frame[train_top:train_bottom, train_left:train_right]

            # Resize the image
            new_size = (224, 224)
            cropped_image_numpy = cv2.resize(cropped_image_numpy, new_size, interpolation=cv2.INTER_AREA)
            
            # Expand dimensions to match the 4D Tensor shape.
            cropped_image_numpy = np.expand_dims(cropped_image_numpy, axis=0)

            # Save or display the cropped image
            cropped_image_numpy_list.append(cropped_image_numpy)

    return cropped_image_numpy_list    



# Load tensorflow lite CNN model

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#class_labels = ['adult', 'infant', 'kid']
class_labels = ['adult','kid']
# initialize the video stream and allow the camera sensor to warm up
# Set the src to the following:
# src = 0: for the built-in single webcam, could be your laptop webcam
# src = 2: I had to set it to 2 in order to use the USB webcam attached to my laptop

# use picamera instead from picamera2 library
picam2 = Picamera2()
#picam2.resolution= (1920, 1080)
camera_config = picam2.create_still_configuration(main={'size': (1080, 720)}, lores={'size': (640, 480)}, display='lores')
camera_config['transform'] = libcamera.Transform(hflip=1, vflip=1)
picam2.configure(camera_config)
picam2.start()

time.sleep(1)

pygame.mixer.init()
sound1= pygame.mixer.Sound('sound_files/sound1.wav')
sound2 = pygame.mixer.Sound('sound_files/sound2.wav')
sound3 = pygame.mixer.Sound('sound_files/sound3.wav')
sound4 = pygame.mixer.Sound('sound_files/sound4.wav')
sound5 = pygame.mixer.Sound('sound_files/sound5.wav')


playing = sound1.play()
while playing.get_busy():
    pygame.time.delay(100)


print("Please move into the camera's range and look straight at the camera for 10 seconds.")
print("")

no_face_counter = 0
multi_face_counter = 0
names_for_ave = []

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    # Capture a frame
    frame_origin = picam2.capture_array()
    # convert picamera image format to cv2 image format
    frame = cv2.cvtColor(frame_origin, cv2.COLOR_RGB2BGR)

    # Detect the face boxes
    boxes = face_recognition.face_locations(frame)

    if boxes:
        
        if len(boxes) > 1:
            multi_face_counter += 1
            if multi_face_counter > 5:
                print("Only one person at a time.")
                playing = sound2.play()
                while playing.get_busy():
                    pygame.time.delay(100)
                multi_face_counter = 0

        else:
            cropped_image_numpy_list = face_image_crop(frame, boxes)

            # Use ML model to predict age group
            names = []
            for cropped_image_numpy in cropped_image_numpy_list:

                # convert image array from UNIT8 to float32
                cropped_image_numpy_32 = cropped_image_numpy.astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], cropped_image_numpy_32)
                # Run inference
                interpreter.invoke()
                # Get the output tensor
                name_prediction = interpreter.get_tensor(output_details[0]['index'])

                # Get the top predicted class index
                predicted_class_index = np.argmax(name_prediction, axis=1)[0]
                # Get the predicted class label
                predicted_class_label = class_labels[predicted_class_index]
                names.append(predicted_class_label)
                names_for_ave.append(predicted_class_label)


            # loop over the recognized faces
            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left,top), (right, bottom), (0, 255, 255), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    else:
        no_face_counter += 1
        if no_face_counter > 7:
            print("No face recognized. Please move into the camera's range and look straight at the camera for 10 seconds.")
            playing = sound3.play()
            while playing.get_busy():
                pygame.time.delay(100)
            no_face_counter = 0
    
    if len(names_for_ave) == 7:
        adult_count = 0
        kid_count = 0
        for each in names_for_ave:
            if each == 'adult':
                adult_count += 1
            else:
                kid_count += 1
        if adult_count > kid_count:
            print("Adult.")
            print("You are authorized to drive the rover! Please use the tablet for rover control.")
            playing = sound4.play()
            while playing.get_busy():
                pygame.time.delay(100)
                print("")
            
            bd = BlueDot()
            robot = Robot(left=(17,18), right=(27,22))
            bd.when_pressed = move
            bd.when_moved = move
            bd.when_released = stop

            sleep(20)
            bd.stop()
            sys.exit()

            
        else:
            print("Child")
            print("Rover control is not permitted for minors. You must have permission from your parents or guardians. I'm excited to drive with you, but I'll need to recognize one of their faces first.")
            print("")
            robot = Robot(left=(17,18), right=(27,22))
            robot.backward()
            sleep(0.5)
            robot.stop()
            playing = sound5.play()
            while playing.get_busy():
                pygame.time.delay(100)
            sys.exit()

        names_for_ave = []


    # display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break
    

cv2.destroyAllWindows()
#vs.stop()
picam2.stop_preview()
picam2.stop()