import os
import sys
import cv2
import dlib
import numpy as np
import time
import threading

from skimage import io

DOWNSAMPLE_RATIO = 2

faceRecognition = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

enrolledFaces = {}

def face_to_vector(image, face):
  return (
    np
    .array(faceRecognition.compute_face_descriptor(image, face))
    .astype(float)
  )

def upscaleFaceRect(rect, factor):
  return dlib.rectangle(rect.left() * factor,
                            rect.top() * factor,
                            rect.right() * factor,
                            rect.bottom() * factor)


def faces_from_image(image):
  UPSAMPLING_FACTOR = 0
  faces = [
    (face.height() * face.width(), shapePredictor(image, face))
    for face in faceDetector(image, UPSAMPLING_FACTOR)
  ]
  return [face for _, face in sorted(faces, reverse=True)]

def largest_face_from_image(image):
  faces = faces_from_image(image)
  return faces[0] if faces else None

def image_from_file(path):
  return io.imread(path)

def identify(image):
#  faceDetect = time.time()
  face = largest_face_from_image(image)
#  print("face detection took " + str(time.time() - faceDetect))
#  faceVec = time.time()
  faceVector = face_to_vector(image, face)
#  print("Face to vector took " + str(time.time() - faceVec))

#  faceCompare = time.time()

  # THIS is probably hazardous as ordering may not be always the same?
  enrollIdentifiers = np.array(list(enrolledFaces.keys()))
  enrollMatrix = np.array(list(enrolledFaces.values()))
  distances = compute_distances(enrollMatrix, faceVector)
  closestIndex = np.argmin(distances)
#  print("Face compare took " + str(time.time() - faceCompare))

  return enrollIdentifiers[closestIndex], distances[closestIndex]

def compute_distances(matrix, vector):
    differences = np.subtract(np.array(matrix), vector)
    return np.linalg.norm(differences, axis=1)

# Semaphor for identifying thread
identifying = False

def handleFrame(origFrame, cb):
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

      thread = threading.Thread(target=handleFrame, args=(frame, cb))
      thread.daemon=True
      thread.start()

  # When everything is done, release the capture
  video_capture.release()

def loadEnrolledFaces():
  global enrolledFaces
  if (not os.path.isfile('faces.npy')):
    print('No enrolled faces found')
  else:
    print('Enrolled faces loaded from faces.npy')
    enrolledFaces = np.load('faces.npy').item()

def logger(identifier, distance, duration):
  if (identifier == '-'):
    print('Unknown person', duration)
  elif (identifier == None):
    print('No face', duration)
  else:
    print(identifier, distance, duration)

loadEnrolledFaces()
webcam(logger)
