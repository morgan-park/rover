import subprocess
from time import sleep
import pygame
import sys

program_path = "rover-motion.py"

pygame.mixer.init()
sound0 = pygame.mixer.Sound('sound_files/sound0.wav')

playing = sound0.play()
sleep(2)   
    
while True:
    subprocess.run(["python", program_path])
