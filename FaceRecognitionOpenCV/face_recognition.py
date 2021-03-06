# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades (Acts like Filters with Equations, not a Neural Network)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the cascade for the face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # Load the cascade for the eyes
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # Load the cascade for the smile

# Defining a function that will do the detections
def detect(gray, frame): # Function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles
    faces = face_cascade.detectMultiScale(gray, 1.5, 2) # Apply the detectMultiScale method from the face cascade to locate one or several faces in the image
    for (x, y, w, h) in faces: # For each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Paint a rectangle around the face
        
        roi_gray = gray[y:y+h, x:x+w] # Get the region of interest in the black and white image
        roi_color = frame[y:y+h, x:x+w] # Get the region of interest in the colored image
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5) # Apply the detectMultiScale method to locate one or several eyes in the image
        for (ex, ey, ew, eh) in eyes: # For each detected eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) # Paint a rectangle around the eyes, but inside the referential of the face
    
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 23) # Apply the detectMultiScale method to locate one or several smiles in the image
        for (sx, sy, sw, sh) in smile: # For each detected smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2) # Paint a rectangle around the smiles, but inside the refertial of the face
    
    return frame; # Return the image with the detector rectangles

# Face recognition with the webcam
video_capture = cv2.VideoCapture(1) # Turning the webcam on, parameter 0 if internal webcam, 1 if external webcam

while True:
    _, frame = video_capture.read() # Getting the last frame. Underscore means returned value ignored
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Color transformation to gray. BGR2GRAY takes average of RGB values for correct shades of grayscale
    canvas = detect(gray, frame) # Get the output from the detect function
    cv2.imshow('Video', canvas) # Display the outputs
    if cv2.waitKey(1) & 0xFF == ord('q'): # If 'q' is pressed on keyboard, break out of while True loop
        break
    
video_capture.release() # Turning the webcam off
cv2.destroyAllWindows() # Destroy all the windows inside which the images were displayed
