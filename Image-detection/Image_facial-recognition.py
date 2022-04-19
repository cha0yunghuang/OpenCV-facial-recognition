import cv2

image = cv2.imread('image-detection/source/gossip-girl.webp')
# image = cv2.imread('Image-detection/source/ff.webp')
# image = cv2.imread('Image-detection/source/wwe.jpeg')

resized_image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)

# Convert to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# OpenCV haarcascade_frontalface model
face_Cascade = cv2.CascadeClassifier('image-detection/opencv/haarcascade_frontalface_default.xml')

# Detection
faceRec = face_Cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10)
print(len(faceRec))

# Draw rectangles on the resized_image
for (x, y, w, h) in faceRec:
    cv2.rectangle(resized_image, (x,y), (x+w, y+h), (0,255,0), 3)

cv2.imshow('Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

