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
        eyebrow_coords = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # Get coordinates for specific landmarks
                landmark_33 = faceLms.landmark[33]
                x_33 = int(landmark_33.x * iw)
                y_33 = int(landmark_33.y * ih)
                landmark_coords.append((x_33, y_33))

                landmark_27 = faceLms.landmark[27]
                x_27 = int(landmark_27.x * iw)
                y_27 = int(landmark_27.y * ih)
                landmark_coords.append((x_27, y_27))

                landmark_133 = faceLms.landmark[133]
                x_133 = int(landmark_133.x * iw)
                y_133 = int(landmark_133.y * ih)
                landmark_coords.append((x_133, y_133))

                landmark_362 = faceLms.landmark[362]
                x_362 = int(landmark_362.x * iw)
                y_362 = int(landmark_362.y * ih)
                landmark_coords.append((x_362, y_362))

                landmark_263 = faceLms.landmark[263]
                x_263 = int(landmark_263.x * iw)
                y_263 = int(landmark_263.y * ih)
                landmark_coords.append((x_263, y_263))

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

                # Optionally draw the specific landmark points on the image
                if draw:
                    cv2.circle(img, (x_33, y_33), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_27, y_27), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_133, y_133), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_263, y_263), 5, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x_362, y_362), 5, (0, 255, 0), cv2.FILLED)

                eyebrow_points = [70, 105, 110, 336, 334]
                eyebrow_landmarks = []
                for id in eyebrow_points:
                    x = int(faceLms.landmark[id].x * iw)
                    y = int(faceLms.landmark[id].y * ih)
                    eyebrow_landmarks.append((x, y))
                
                eyebrow_coords.append(eyebrow_landmarks)

                # Optionally draw circles on these eyebrow landmark points
                if draw:
                    for (x, y) in eyebrow_landmarks:
                        cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
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
        print("Landmark Coordinates:", landmark_coords)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
