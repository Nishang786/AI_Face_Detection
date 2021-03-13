import cv2

face_trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('rdjce.png')
webcam = cv2.VideoCapture(0)

while True:
     
     
    frame_read, frame  = webcam.read()

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_trained_data.detectMultiScale(grayscale_img)

#print(face_coordinates)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('face detect', frame)

    cv2.waitKey(1)
    
     

print('code completed')