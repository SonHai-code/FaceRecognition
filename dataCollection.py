from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import cv2
from time import time

####################################
classID = 0 # which 0 is fake and 1 is real
outputFolderPath = 'Dataset/DataCollect'
blurThreshold = 35 # Greater is more focus
save = True
debug = False

confidence = 0.8
offsetPercentageW = 10
offsetPercentageH = 20
camWidth = 640
camHeight = 480
floatingPoint = 6
####################################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4, camHeight)

detector = FaceDetector()
while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = [] # Contain True-False values indicating the faces are blur or not
    listInfo = [] # The normalized values and the class name for the label .txt file

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            print(x ,y, w, h)

            #-----Checking the score.When we are out of the frame, not dectect another object------#
            if score > confidence:
                #-----Adding an offset to the face detected------#
                offsetW = (offsetPercentageW / 100) * w
                #Put it a little bit backward.
                x = int(x - offsetW)
                #Open in both side: left and right.
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                #-----Avoid values under 0 error. When we out of frame not giving an error------#
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                #----Find the Blurriness-----#
                imgFace = img[y:y+h, x:x+w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)


                #----Normalize values-----#
                ih, iw, _ = img.shape
                # Give center point of img
                xc, yc = x + w/2, y + h/2
                xcn, ycn = round(xc/iw, floatingPoint), round(yc/ih, floatingPoint)
                wn, hn = round(w/iw, floatingPoint), round(h/ih, floatingPoint)
                print(xcn, ycn, wn, hn)

                #-----Avoid values greater than 1 error. When we are too close with camera------#
                if xcn > 1: x = 1
                if ycn > 1: y = 1
                if wn > 1: w = 1
                if hn > 1: h = 1

                #----Creating the image info----#
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                #----Drawing----#
                cv2.rectangle(imgOut, (x ,y, w, h ), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut,f'Score = {int(score*100)}% Blur = {blurValue}', (x, y - 20), scale=2, thickness=3)

                if debug: # Debug Mode
                    cv2.rectangle(img, (x ,y, w, h ), (255, 0, 0), 3)
                    cvzone.putTextRect(img,f'Score = {int(score*100)}% Blur = {blurValue}', (x, y - 20), scale=2, thickness=3)
            #----Conditioning to save the images or not----#
            if save:
                if all(listBlur) and listBlur!=[]:
                    #----Save the image----#
                    timeNow = str(time()).split(".")
                    timeNow = timeNow[0] + timeNow[1]
                    print(timeNow);
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                    #----Save the label text file----#
                    for info in listInfo:
                        f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                        f.write(info)
                        f.close()

    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
