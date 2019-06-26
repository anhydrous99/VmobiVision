import argparse
from pathlib import Path

from utils import file_check
from speech_tools import Speaker

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
                    default='models/east_mobilenet_v2_quantized_512x512_11_06_2019.tflite',
                    help='The model file for text detection')
parser.add_argument('-l', '--object_detection_label',
                    default='models/labels.txt',
                    help='Label file for object detection')
parser.add_argument('--no_text_to_speech', action='store_true',
                    help='Use this to disable text to speech')
args = parser.parse_args()

odm_path = Path(args.object_detection_model)
tdm_path = Path(args.text_detection_model)
odl_path = Path(args.object_detection_label)

file_check(odm_path)
file_check(tdm_path)
file_check(odl_path, '.txt')

spk = Speaker(rate=160, dontspeak=args.no_text_to_speech)

spk.say('Hello, welcome to i Mob e. Press o to switch to object detection mode. Press t to switch to text detection'
        ' mode. Press s to start or stop speaking.')
