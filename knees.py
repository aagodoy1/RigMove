import cv2
import mediapipe as mp

# Inicializar el módulo de pose de MediaPipe
mp_pose = mp.solutions.pose
# Puedes probar con diferentes modelos:
# - mp_pose.Pose(static_image_mode=False, model_complexity=1) para un balance velocidad/precisión
# - mp_pose.Pose(static_image_mode=False, model_complexity=2) para más precisión (más lento)
# - mp_pose.Pose(static_image_mode=False, model_complexity=0) para más velocidad (menos preciso)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5, # Umbral de confianza para la detección inicial
    min_tracking_confidence=0.5   # Umbral de confianza para el seguimiento
)

# Para dibujar los landmarks y conexiones
mp_drawing = mp.solutions.drawing_utils

#video_path = r'C:\Users\user\Desktop\Carpetas\Training_Materal\Streching.mp4'  # Cambia esto por la ruta a tu video
video_path = r'C:\Users\user\Desktop\Carpetas\Training_Materal\Streching.mp4'  # Cambia esto por la ruta a tu video

cap = cv2.VideoCapture(video_path)

print('Listo')

# Define el nuevo ancho y alto para la visualización (ajústalo según necesites)
new_width = 800
new_height = 600

if not cap.isOpened():
    print(f"Error: No se pudo abrir el archivo de video en la ruta: {video_path}")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen de BGR a RGB (MediaPipe trabaja con RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame para detectar la pose
        # Nota: Puedes usar .process() directamente aquí sin crear otra variable
        results = pose.process(rgb_frame)

        # Dibujar los landmarks si se detecta una pose
        if results.pose_landmarks:
            # Dibuja todos los landmarks del cuerpo
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Estilo para los puntos
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Estilo para las conexiones
            )
            
            # --- Opcional: Para enfocarse solo en las rodillas ---
            # Puedes acceder a landmarks específicos. Los índices de los landmarks de MediaPipe Pose son:
            # Izquierda: 25 (Rodilla izquierda), 23 (Cadera izquierda)
            # Derecha: 26 (Rodilla derecha), 24 (Cadera derecha)
            
            # Ejemplo: Extraer y dibujar solo las rodillas
            landmark_id_left_knee = mp_pose.PoseLandmark.LEFT_KNEE.value
            landmark_id_right_knee = mp_pose.PoseLandmark.RIGHT_KNEE.value
            
            # Si quieres dibujar solo las rodillas, podrías hacer algo más complejo
            # que requiere crear una lista de puntos y conexiones para cada rodilla.
            # Sin embargo, la forma más sencilla es dibujar la pose completa y luego
            # enfocarte en los valores de las rodillas si necesitas sus coordenadas.
            
            #Acceder a las coordenadas de la rodilla izquierda (ejemplo):
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]:
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
                x_knee = int(left_knee.x * frame.shape[1])
                y_knee = int(left_knee.y * frame.shape[0])
                cv2.circle(frame, (x_knee, y_knee), 5, (0, 255, 0), -1) # Dibuja un círculo verde en la rodilla izquierda
                cv2.putText(frame, f"LIz: ({x_knee},{y_knee})", (x_knee + 10, y_knee), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Redimensionar el frame para la visualización
        resized_frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow("Deteccion de Pose (Rodillas)", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pose.close() # Es buena práctica cerrar el objeto MediaPipe Pose