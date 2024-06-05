import cv2
import numpy as np
from cvzone import *
import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(min_detection_confidence=self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        ih, iw, ic = img.shape
        bboxs = []
        landmark_coords = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # Get coordinates for a specific landmark, e.g., landmark 23
                landmark_33 = faceLms.landmark[33]
                x_33 = int(landmark_33.x * iw)
                y_33 = int(landmark_33.y * ih)
                landmark_coords.append((x_33, y_33))

                # Collecting coordinates for both eyes
                eye_coords = []
                for id in [33, 133, 362, 263]:  # Right eye (33, 133) and left eye (362, 263)
                    x = int(faceLms.landmark[id].x * iw)
                    y = int(faceLms.landmark[id].y * ih)
                    eye_coords.append((x, y))

                # Calculate bounding box around the eyes
                x_min = min(eye_coords, key=lambda p: p[0])[0]
                y_min = min(eye_coords, key=lambda p: p[1])[1]
                x_max = max(eye_coords, key=lambda p: p[0])[0]
                y_max = max(eye_coords, key=lambda p: p[1])[1]
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                bboxs.append(bbox)

                if draw:
                    img = self.fancyDraw(img, bbox)

                # Optionally draw the specific landmark point on the image
                if draw:
                    cv2.circle(img, (x_33, y_33), 5, (0, 255, 0), cv2.FILLED)
        return img, bboxs, landmark_coords

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right  x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left  x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img




def overlay_image(main_img, overlay_img, x, y):
    """
    Overlays `overlay_img` onto `main_img` at position `(x, y)`.
    """
    # Get the height and width of the overlay image
    overlay_height, overlay_width, _ = overlay_img.shape

    # Define the ROI (Region of Interest) on the main image to place the overlay
    roi = main_img[y:y+overlay_height, x:x+overlay_width]

    # Create a mask of the overlay image
    overlay_img_gray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(overlay_img_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Create masked regions
    main_img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    overlay_img_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    # Combine the two images
    dst = cv2.add(main_img_bg, overlay_img_fg)
    main_img[y:y+overlay_height, x:x+overlay_width] = dst

    return main_img

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img, bboxs, landmark_coords = detector.findFaces(img)
        print("Bounding Boxes:", bboxs)
        print("Landmark 23 Coordinates:", landmark_coords)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        imgg = cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    # Load main image and overlay image
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img, bboxs, landmark_coords = detector.findFaces(img)
        print("Bounding Boxes:", bboxs)
        print("Landmark 23 Coordinates:", landmark_coords)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        imgg = cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    main_img = img
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    # glasses_img = cv2.imread('glasses/glass3.png', cv2.IMREAD_UNCHANGED) # Make sure the image has an alpha channel (RGBA)
    while True:
        success, img1 = cap.read()
        if not success:
            break
    overlay_img = cv2.imread('glasses/glass11.png', -1)  # -1 loads with alpha channel if present

    # Example coordinates (you would replace these with your actual landmarks or annotations)
    x = 308
    y = 307

    # Overlay the image
    result_img = overlay_image(img, overlay_img, x, y)

    # Display the result
    cv2.imshow('Overlay Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
