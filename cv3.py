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
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # Draw every landmark point with numbering
                for id, lm in enumerate(faceLms.landmark):
                    x = int(lm.x * iw)
                    y = int(lm.y * ih)
                    if draw:
                        cv2.circle(img, (x, y), 2, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)

                # Optionally, draw bounding box around face mesh points
                x_min = min([int(lm.x * iw) for lm in faceLms.landmark])
                y_min = min([int(lm.y * ih) for lm in faceLms.landmark])
                x_max = max([int(lm.x * iw) for lm in faceLms.landmark])
                y_max = max([int(lm.y * ih) for lm in faceLms.landmark])
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                bboxs.append(bbox)

                if draw:
                    img = self.fancyDraw(img, bbox)
        return img, bboxs

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
        img, bboxs = detector.findFaces(img)
        print("Bounding Boxes:", bboxs)

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
