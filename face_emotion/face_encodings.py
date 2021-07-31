#Note this is a simple implementation from this source: 
# https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

import face_recognition
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from emotion_models import *


# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("../data/face/images.m1/barack_obama.jpg")
obama_face_encoding =  face_recognition.face_encodings(obama_image)[0]
# Load another sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("../data/face/images.m1/joe_biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
# Load another sample picture and learn how to recognize it.
jerico_image = face_recognition.load_image_file("../data/face/images.m1/jerico_johns.jpg")
jerico_face_encoding = face_recognition.face_encodings(jerico_image)[0]
# Load another sample picture and learn how to recognize it.
sudhrity_image = face_recognition.load_image_file("../data/face/images.m1/sudhrity_mondal.jpg")
sudhrity_face_encoding = face_recognition.face_encodings(sudhrity_image)[0]
# Load another sample picture and learn how to recognize it.
diana_image = face_recognition.load_image_file("../data/face/images.m1/diana_chacon.jpg")
diana_face_encoding = face_recognition.face_encodings(diana_image)[0]
# Load another sample picture and learn how to recognize it.
josh_image = face_recognition.load_image_file("../data/face/images.m1/josh_jonte.jpg")
josh_face_encoding = face_recognition.face_encodings(josh_image)[0]
# Load another sample picture and learn how to recognize it.
piotr_image = face_recognition.load_image_file("../data/face/images.m1/piotr_parkitny.jpg")
piotr_face_encoding = face_recognition.face_encodings(piotr_image)[0]
# Load another sample picture and learn how to recognize it.
kevin_image = face_recognition.load_image_file("../data/face/images.m1/kevin_martin.jpg")
kevin_face_encoding = face_recognition.face_encodings(kevin_image)[0]
# Load another sample picture and learn how to recognize it.
eric_image = face_recognition.load_image_file("../data/face/images.m1/eric_lundy.jpg")
eric_face_encoding = face_recognition.face_encodings(eric_image)[0]
# Load another sample picture and learn how to recognize it.
divesh_image = face_recognition.load_image_file("../data/face/images.m1/divesh_kumar.jpg")
divesh_face_encoding = face_recognition.face_encodings(divesh_image)[0]
# Load another sample picture and learn how to recognize it.
catherine_image = face_recognition.load_image_file("../data/face/images.m1/catherine_mou.jpg")
catherine_face_encoding = face_recognition.face_encodings(catherine_image)[0]
# Load another sample picture and learn how to recognize it.
brad_image = face_recognition.load_image_file("../data/face/images.m1/brad_desaulniers.jpg")
brad_face_encoding = face_recognition.face_encodings(brad_image)[0]
# Load another sample picture and learn how to recognize it.
alice_image = face_recognition.load_image_file("../data/face/images.m1/alice_hua.jpg")
alice_face_encoding = face_recognition.face_encodings(alice_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    jerico_face_encoding,
    sudhrity_face_encoding,
    diana_face_encoding,
    josh_face_encoding,
    piotr_face_encoding,
    kevin_face_encoding,
    eric_face_encoding,
    divesh_face_encoding,
    catherine_face_encoding,
    brad_face_encoding,
    alice_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Jerico Johns",
    "Sudhrity Mondal",
    "Diana Chacon",
    "Josh Jonte",
    "Piotr Parkkitny",
    "Kevin Martin",
    "Eric Lundy",
    "Divesh Kumar",
    "Catherine Mou",
    "Brad Desaulniers",
    "Alice Hua"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

