import cv2

image = cv2.imread('Image-detection/source/gossip-girl.webp')
# image = cv2.imread('Image-detection/source/ff.webp')
# image = cv2.imread('Image-detection/source/wwe.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# OpenCV haarcascade_frontalface model
face_Cascade = cv2.CascadeClassifier('Image-detection/opencv/haarcascade_frontalface_default.xml')

# Detection
faceRec = face_Cascade.detectMultiScale(gray, 1.1, 10)
print(len(faceRec))

# Create and locate rectangles on the face
for (x, y, w, h) in faceRec:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('Detection', image)
cv2.waitKey(10000)

