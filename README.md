# Rover with Facial Recognition Capabilities
This is a rover with facial recognition capabilities that I developed for my capstone project in my Master’s degree in Computer Science. 
The rover detects a user's face and classifies them as either a child or an adult. For adults, the rover enables control through Blue Dot, 
an Android app installed on my tablet. If classified as a child, the rover moves backward and advises that permission from a parent or guardian is required.

![capstone1](https://github.com/morgan-park/rover/assets/94096127/26f4fca5-4a6f-4640-ac1d-cc90eeb259fc)


## Hardware Design
The hardware design involves building the physical robot using Raspberry Pi. 
I integrated a camera, a speaker, wheels, motors, and motor drivers, etc., for user interaction and motion.
<br>
<br>
## Software Design
The software design focuses on training machine learning models. I conducted experiments using transfer learning with Convolutional Neural Networks, 
specifically MobileNet V.2, developed by Google and implemented using TensorFlow.
All code was written in Python. The main libraries used include TensorFlow, OpenCV, PiCamera2, NumPy, Matplotlib, libcamera, gpiozero, pygame, and BlueDot.
<br>
<br>
## Code Details
__1. transfer-learning-experiment.ipynb:__ 
This Python code uses TensorFlow to train three different models, plot their results, and compare them:
  * Transfer learning from MobileNet V.2 with only the output layer retrained.
  * Transfer learning from MobileNet V.2 with the last 30 layers retrained.
  * Custom-built CNN model.

For detailed model architecture, visit my website: [Model architecture and performance of different models](https://morgan-park.github.io/projects.html#project0)

__2. rover-motion.py:__ Written in Python, this code runs on a Raspberry Pi. Using OpenCV and the face-recognition Python library, it enables the rover to 
recognize user faces, preprocess them for image recognition, and classify them using TensorFlow Lite. I integrated AI voice generated through narakeet.com, 
played using pygame, for the rover's communication with users. Gpiozero controls the rover's motion, and BlueDot allows adult users to control the rover via 
an Android tablet.

This code is designed for experimentation and modification for your specific purposes. The live camera window will pop up once you run the program, 
allowing you to view the result of facial recognition and interact with the code. The program will continue running; to quit, press 'q.'


__3. run_program.py:__ 
For a smoother user experience, I created this program to operate the rover without displaying all the detailed and complex code. Users can simply execute 
this code, and the rover will run until interrupted by pressing Ctrl + C.
