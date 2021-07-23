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
from model import *


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("images/barack_obama.jpg")
obama_face_encoding =  face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("images/joe_biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
jerico_image = face_recognition.load_image_file("images/jerico_johns.jpg")
jerico_face_encoding = face_recognition.face_encodings(jerico_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    jerico_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Jerico Johns" 
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


#Emotion recognition
def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model


model = load_trained_model('./models/FER_trained_model.pt')
emotion_dict = {0: 'Neutral', 1: 'Happy', 2: 'Surprise', 3: 'Sad',
                    4: 'Angry', 5: 'Disgust', 6: 'Fear'}


val_transform = transforms.Compose([
        transforms.ToTensor()])



while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        #resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        resize_frame = cv2.resize(gray[top:bottom, left:right], (48, 48))

        X = resize_frame/256
        X = Image.fromarray((X))
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = emotion_dict[int(top_class.numpy())] 

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 100, 0), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left+2, bottom - 50), (right-2, bottom-26), (152, 251, 152), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        font = cv2.FONT_HERSHEY_DUPLEX
        x, y, w, h = left+2, bottom-50, right-2, bottom-26
        sub_img = frame[y:h, x:w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.8, white_rect, 0.1, 1.0)
        frame[y:h, x:w] = res

        scale = 0.05 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
        fontScale = min(w,h)/(25/scale)

        #cv2.putText(frame, name, (left + 6, bottom - 29), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, name, (left + 6, bottom - 29), font, fontScale, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left+2, bottom - 25), (right-2, bottom-2), (47, 79, 79), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame, pred, (left + 6, bottom - 4), font, 1.0, (34, 139, 34), 2, cv2.LINE_AA)


        # First we crop the sub-rect from the image
        x, y, w, h = left+2, bottom-25, right-2, bottom-2
        #sub_img = frame[y:y+h, x:x+w]
        sub_img = frame[y:h, x:w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.2, 1.0)

        # Putting the image back to its position
        #frame[y:y+h, x:x+w] = res
        frame[y:h, x:w] = res
        #cv2.putText(frame, pred, (left + 6, bottom - 4), font, fontScale, (34, 139, 34), 2, cv2.LINE_AA)
        cv2.putText(frame, pred, (left + 6, bottom - 4), font, fontScale, (255, 0, 255), 1, cv2.LINE_AA)

        ##  Draw a label with a name below the face
        #cv2.rectangle(frame, (left+2, bottom - 35), (right-2, byttom-2), (152, 251, 152), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (34, 139, 34), 2, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
