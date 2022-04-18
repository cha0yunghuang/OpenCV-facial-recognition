import cv2

cap = cv2.VideoCapture(0)

# OpenCV haarcascade_frontalface model
face_Cascade = cv2.CascadeClassifier('Cam-detection/opencv/haarcascade_frontalface_alt2.xml')


while(True):
    ret, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detection
    faceRec = face_Cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
    
    # Draw rectangles on the face
    for (x, y, w, h) in faceRec:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    # Display
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break