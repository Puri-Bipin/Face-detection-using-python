import cv2

#face detector=classifier
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#imread reads the images
#img=cv2.imread('group.jpg')

#captures video from webcam
webcam=cv2.VideoCapture(0)

while True:
    #reads current frame
    successful_frame_read, frame = webcam.read()


    #converts/must convert into graysacle
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect images MultiScale(dosenot depend whether images are large or small) only complexions and 
    #returns the coordinates of rectangle surrounding the image
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

    #draws rectangle on the frame
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,250,0),2)

    #shows image
    cv2.imshow('Puri face detector',frame)
    #each frame one milisecond
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()



