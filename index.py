import mediapipe as mp
import numpy as np
import cv2
face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]
 
    height, width = img.shape[:-1]
 
    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:
 
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0
 
        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))
 
            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y
 
            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')
 
            relevant_keypnts = []
 
            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0

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
    img= getLandmarks(img)
    cv2.imshow("Person",img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break



webcam.release()
cv2.destroyAllWindows()