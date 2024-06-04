import cv2
import cvzone
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector

detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()
    img, faces = detector.findFaceMesh(img, draw=True)
     # Check if any faces are detected
    if faces:
        # print(faces[3][0:2])
        image = cv2.imread('glasses/glass3.png',cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img,image,(1,5))
        # Loop through each detected face
        for face in faces:
            # print (face)
            # Get specific points for the eye
            # leftEyeUpPoint: Point above the left eye
            # leftEyeDownPoint: Point below the left eye
            leftEyeUpPoint = face[159]
            leftEyeDownPoint = face[23]
            # Calculate the vertical distance between the eye points
            # leftEyeVerticalDistance: Distance between points above and below the left eye
            # info: Additional information (like coordinates)
            leftEyeVerticalDistance, info = detector.findDistance(leftEyeUpPoint, leftEyeDownPoint)
            # Print the vertical distance for debugging or information
            print(leftEyeVerticalDistance)
    cv2.imshow("Person",img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break



webcam.release()
cv2.destroyAllWindows()
