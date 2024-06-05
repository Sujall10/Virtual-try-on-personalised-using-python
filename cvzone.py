import cv2
import mediapipe as mp
import time
import numpy as np

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
        landmarks = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    x = int(lm.x * iw)
                    y = int(lm.y * ih)
                    landmarks.append((x, y))
                    if draw:
                        cv2.circle(img, (x, y), 2, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)

                x_min = min([int(lm.x * iw) for lm in faceLms.landmark])
                y_min = min([int(lm.y * ih) for lm in faceLms.landmark])
                x_max = max([int(lm.x * iw) for lm in faceLms.landmark])
                y_max = max([int(lm.y * ih) for lm in faceLms.landmark])
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                bboxs.append(bbox)

                if draw:
                    img = self.fancyDraw(img, bbox)
        return img, bboxs, landmarks

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

def overlay_image(background, overlay, x, y, overlay_size=None):
    bg_h, bg_w, bg_channels = background.shape
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    h, w, _ = overlay.shape
    rows, cols = h, w

    if x + w > bg_w or y + h > bg_h:
        return background

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:]

    background_part = background[y:y+rows, x:x+cols]

    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(background_part, background_part, mask=mask_inv)
    img_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)

    dst = cv2.add(img_bg, img_fg)
    background[y:y+rows, x:x+cols] = dst

    return background

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    glasses_img = cv2.imread('glasses/glass4.png', cv2.IMREAD_UNCHANGED) # Make sure the image has an alpha channel (RGBA)

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs, landmarks = detector.findFaces(img)

        if landmarks:
            left_eye_point = landmarks[63]
            right_eye_point = landmarks[298]
            eye_center = ((left_eye_point[0] + right_eye_point[0]) // 2, (left_eye_point[1] + right_eye_point[1]) // 2)
            glasses_width = int(1.5 *abs(right_eye_point[0] - left_eye_point[0]))
            glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
            img = overlay_image(img, glasses_img, eye_center[0] - glasses_width // 2, eye_center[1] - glasses_height // 2, (glasses_width, glasses_height))

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
