import os
import sys
import cv2
import dlib
import numpy as np
import time
import threading

from skimage import io

face_recognition = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

enrolled_faces = {}

# Semaphor for identifying thread
identifying = False

def face_to_vector(image, face):
  return (
    np
    .array(face_recognition.compute_face_descriptor(image, face))
    .astype(float)
  )

def faces_from_image(image):
  UPSAMPLING_FACTOR = 0
  faces = [
    (face.height() * face.width(), shape_predictor(image, face))
    for face in face_detector(image, UPSAMPLING_FACTOR)
  ]
  return [face for _, face in sorted(faces, reverse=True)]

def image_from_file(path):
  return io.imread(path)

def identify(image):
  # Get all faces
  faces = faces_from_image(image)
  # Pick largest face
  face = faces[0] if faces else None
  
  # Calculate face descriptor
  descriptor = face_recognition.compute_face_descriptor(image, face)
  face_vector = np.array(descriptor).astype(float)

  # THIS is probably hazardous as ordering may not be always the same?
  enroll_identifiers = np.array(list(enrolled_faces.keys()))
  enroll_matrix = np.array(list(enrolled_faces.values()))

  # Calculate differences between the face and all enrolled faces
  differences = np.subtract(np.array(enroll_matrix), face_vector)
  distances = np.linalg.norm(differences, axis=1)
  # and pick the closest one
  closest_index = np.argmin(distances)

  return enroll_identifiers[closest_index], distances[closest_index]


def handle_frame(origFrame, cb):
  global identifying
  try:
    frame = cv2.resize(origFrame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    start = time.time()
    identifier, distance = identify(frame)
    if (distance < 0.6):
       cb(identifier, distance, time.time() - start)
    else:
       cb('-', distance, time.time() - start)
    sys.stdout.flush()

  except Exception as e:
    exc = e
    cb(None, 0, time.time() - start)
    # print(e)

  identifying = False

def webcam(cb):
  global identifying
  video_capture = cv2.VideoCapture(0)

  while True:
    video_capture.grab()
    if (not identifying):
      ret, frame = video_capture.retrieve()

      if (ret == False):
        print('No frame')
        break
      identifying = True

      thread = threading.Thread(target=handle_frame, args=(frame, cb))
      thread.daemon=True
      thread.start()

  # When everything is done, release the capture
  video_capture.release()

def load_enrolled_faces():
  global enrolled_faces
  if (not os.path.isfile('faces.npy')):
    print('No enrolled faces found')
  else:
    print('Enrolled faces loaded from faces.npy')
    enrolled_faces = np.load('faces.npy').item()

def logger(identifier, distance, duration):
  if (identifier == '-'):
    print('Unknown person', duration)
  elif (identifier == None):
    print('No face', duration)
  else:
    print(identifier, distance, duration)

load_enrolled_faces()
webcam(logger)
