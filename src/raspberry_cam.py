# src/raspberry_cam.py

import cv2
import RPi.GPIO as GPIO
from picamera import PiCamera
from time import sleep
from detector import PlateDetector
from ocr import recognize_plate
from utils import crop_bbox
from config import SERVO_PIN, TRUSTED_PLATES

def setup_servo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(2.5)
    return pwm

def open_barrier(pwm):
    pwm.ChangeDutyCycle(7.5)
    sleep(2)
    pwm.ChangeDutyCycle(2.5)

def camera_loop():
    camera = PiCamera()
    pwm = setup_servo()
    detector = PlateDetector()
    try:
        while True:
            camera.capture('frame.jpg')
            frame = cv2.imread('frame.jpg')
            dets = detector.detect(frame)
            if len(dets) == 0:
                continue
            best = dets[0][:4]
            plate_img = crop_bbox(frame, best)
            plate_txt = recognize_plate(plate_img, 'cam')
            print(f"Odczytano: {plate_txt}")
            if plate_txt in TRUSTED_PLATES:
                print("Tablica zaufana, otwieram szlaban!")
                open_barrier(pwm)
            else:
                print("Tablica NIEzaufana!")
            sleep(1)
    finally:
        pwm.stop()
        GPIO.cleanup()
