import mediapipe as mp
import numpy as np
import cv2
face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


webcam = cv2.VideoCapture(0)
while webcam.isOpened():
    success, img=webcam.read()

    #applying facemesh model using mediapipe
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_mesh.FaceMesh(refine_landmarks = True).process(img)

    #drawing annotations
    cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            iris_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
            iris_connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            (results.multi_face_landmarks[0].landmark[205].x * img.shape[1], results.multi_face_landmarks[0].landmark[205].y * img.shape[0])

            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections= face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections= face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections= face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    cv2.imshow("Person",img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break



webcam.release()
cv2.destroyAllWindows()