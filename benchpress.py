import cv2
import mediapipe as mp
import numpy as np # Importar numpy para cálculos vectoriales

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    a = np.array(a) # Primer punto (ej. hombro/cadera)
    b = np.array(b) # Punto medio (ej. codo/hombro)
    c = np.array(c) # Tercer punto (ej. muñeca/codo)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Inicializar el módulo de pose de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, # model_complexity=1 suele ser un buen balance
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

video_path = r'C:\Users\user\Desktop\Carpetas\Training_Materal\BenchPress2.mp4' # Ajusta la ruta a tu video
cap = cv2.VideoCapture(video_path)

print('Listo')

# Define el nuevo ancho y alto para la visualización
new_width = 800
new_height = 600

# Parámetro para la rotación (ajústalo según tu video)
rotation_code = cv2.ROTATE_90_CLOCKWISE # O cv2.ROTATE_90_COUNTERCLOCKWISE, o None si no es necesario

if not cap.isOpened():
    print(f"Error: No se pudo abrir el archivo de video en la ruta: {video_path}")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotar el frame ANTES de procesar con MediaPipe si es necesario
        if rotation_code is not None:
            rotated_frame = cv2.rotate(frame, rotation_code)
        else:
            rotated_frame = frame

        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame para detectar la pose
        results = pose.process(rgb_frame)

        # Ángulos por defecto si no se detecta la pose
        left_elbow_angle = 0
        right_elbow_angle = 0
        right_shoulder_angle = 0 # Nuevo: Ángulo del hombro derecho

        # Dibujar los landmarks y calcular ángulos si se detecta una pose
        if results.pose_landmarks:
            # Dibuja todos los landmarks del cuerpo
            mp_drawing.draw_landmarks(
                rotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Extraer coordenadas de los landmarks necesarios
            h, w, c = rotated_frame.shape # Alto, Ancho, Canales del frame rotado

            try:
                # Codo Izquierdo
                shoulder_l = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
                elbow_l = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * w,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
                wrist_l = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * h]

                left_elbow_angle = calculate_angle(shoulder_l, elbow_l, wrist_l)
                cv2.putText(rotated_frame, f"Codo Izq: {int(left_elbow_angle)} deg",
                            tuple(np.array(elbow_l).astype(int) + [10, 0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # Codo Derecho
                shoulder_r = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                              results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
                elbow_r = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h]
                wrist_r = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * w,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * h]

                right_elbow_angle = calculate_angle(shoulder_r, elbow_r, wrist_r)
                cv2.putText(rotated_frame, f"Codo Der: {int(right_elbow_angle)} deg",
                            tuple(np.array(elbow_r).astype(int) + [10, 30]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                # --- NUEVO: Hombro Derecho ---
                # Landmarks: Cadera derecha (24), Hombro derecho (12), Codo derecho (14)
                hip_r = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * h]
                
                # Ya tenemos shoulder_r y elbow_r de los cálculos del codo derecho
                
                right_shoulder_angle = calculate_angle(hip_r, shoulder_r, elbow_r)
                cv2.putText(rotated_frame, f"Hombro Der: {int(right_shoulder_angle)} deg",
                            tuple(np.array(shoulder_r).astype(int) + [10, -20]), # Ajustar posición para que no se superponga
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


            except Exception as e:
                # Esto captura errores si algún landmark no se detecta
                # print(f"Error al calcular ángulo: {e}")
                pass # Ignoramos el error y los ángulos permanecen en 0

        # Redimensionar el frame rotado para la visualización
        if rotation_code == cv2.ROTATE_90_CLOCKWISE or rotation_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
             resized_frame = cv2.resize(rotated_frame, (new_height, new_width))
        else:
             resized_frame = cv2.resize(rotated_frame, (new_width, new_height))

        cv2.imshow("Deteccion de Pose y Angulos", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pose.close()