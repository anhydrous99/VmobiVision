"""
run.py
========================
The core module of Vmobi. See usage documentation for more information.
"""

import cv2
import time
import argparse
import keyboard
from pathlib import Path
from threading import Lock

from utils import file_check, Mode
from speech_tools import Speaker
from inferencing_engines import ObjectDetectionEngine, TextDetectionEngine


def main():
    """
    The core function of the program
    """
    parser = argparse.ArgumentParser(
        description='Object Detection & Optical Text Recognition software',
        epilog='''This program uses a Single Shot MultiBox Detector (SSD) model for object detection and 
         the model Efficient and Accurate Scene Text Detector for text detection. Text recognition is done
         through google's tesseract.'''
    )
    parser.add_argument('-o', '--object_detection_model',
                        default='models/ssd_mobilenet_v2_quantized_300x300_2019_01_03.tflite',
                        help='The model file for object detection')
    parser.add_argument('-t', '--text_detection_model',
                        default='models/east_mobilenet_v2_cocotext.tflite',
                        help='The model file for text detection')
    parser.add_argument('-l', '--object_detection_label',
                        default='models/labels.txt',
                        help='Label file for object detection')
    parser.add_argument('-s', '--object_detection_threshold',
                        default=0.8)
    parser.add_argument('--no_text_to_speech', action='store_true',
                        help='Use this to disable text to speech')
    args = parser.parse_args()

    odm_path = Path(args.object_detection_model)
    tdm_path = Path(args.text_detection_model)
    odl_path = Path(args.object_detection_label)
    od_threshold = args.object_detection_threshold

    # Check if files exist
    file_check(odm_path)
    file_check(tdm_path)
    file_check(odl_path, '.txt')

    # Create text-to-speech instance
    spk = Speaker(rate=160, dontspeak=args.no_text_to_speech)

    # Say intro
    spk.say('Hello, welcome to Vmobi. Press o to switch to object detection mode. Press t to switch to text detection'
            ' mode. Press q to exit', True)

    # Create mutex locks for safe multi-threading
    mode_mutex = Lock()
    stop_mutex = Lock()

    # Create some global vars for signaling
    mode = Mode.NONE
    stop = False

    # Callback functions on key presses
    def o_callback():
        global mode
        with mode_mutex:
            if mode != Mode.ODM:
                mode = Mode.ODM
            else:
                mode = Mode.NONE

    def t_callback():
        global mode
        with mode_mutex:
            if mode != Mode.TDM:
                mode = Mode.TDM
            else:
                mode = Mode.NONE

    def q_callback():
        global stop
        with stop_mutex:
            stop = True

    # Assign callback keys to callback functions
    keyboard.add_hotkey('q', q_callback)
    keyboard.add_hotkey('o', o_callback)
    keyboard.add_hotkey('t', t_callback)

    # Start Engines
    od_engine = ObjectDetectionEngine(str(odm_path), str(odl_path), od_threshold)
    td_engine = TextDetectionEngine(str(tdm_path))

    # Connect camera
    capture = cv2.VideoCapture(0)

    # Wait for camera to warm up
    time.sleep(0.4)

    # Check if connection with camera is successful
    if not capture.isOpened():
        spk.say("Error: couldn't connect with camera.", True)
        exit(1)

    # Loop through frames
    while capture.isOpened():

        # Read current frame
        _, frame = capture.read()

        with mode_mutex:
            if mode == Mode.ODM:
                classes = od_engine.run_inference(frame)
                for cls in classes:
                    spk.asyncsay(cls)
                spk.runAndWait()
            elif mode == Mode.TDM:
                found_text = td_engine.run_inference(frame)
                for text in found_text:
                    spk.asyncsay(text)
                spk.runAndWait()

        with stop_mutex:
            if stop:
                break

    capture.release()


if __name__ == "__main__":
    main()
