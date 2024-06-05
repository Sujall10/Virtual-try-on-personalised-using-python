import cv2
import mediapipe as mp
import time

# Initialize video capture from the default webcam
cap = cv2.VideoCapture(0)
pTime = 0

# Initialize MediaPipe Face Detection and Drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # Extract bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                    int(bboxC.width * iw), int(bboxC.height * ih))
            
            # Draw the bounding box and confidence score on the image
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    # Calculate and display the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    # Display the image with face detections
    cv2.imshow("Image", img)
    

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
