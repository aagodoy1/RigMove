import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

video_path = r'C:\Users\user\Desktop\Carpetas\Training_Materal\Streching.mp4'
cap = cv2.VideoCapture(video_path)

print('Listo')

# Define el nuevo ancho y alto para la visualización
new_width = 800
new_height = 600 # Puedes ajustar estos valores según lo que te parezca mejor

if not cap.isOpened():
    print(f"Error: No se pudo abrir el archivo de video en la ruta: {video_path}")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame antes de procesar y mostrar
        # Esto es importante para que el procesamiento de MediaPipe no se afecte por el cambio de tamaño para mostrar
        # aunque MediaPipe puede escalar internamente si el frame es muy grande.
        # Si quieres que MediaPipe trabaje con la resolución original y solo achicar para mostrar, hazlo después de procesar.
        # Para tu caso, si quieres achicar la ventana, es mejor redimensionar antes de mostrar.

        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Redimensionar el frame para la visualización después de dibujar los landmarks
        # (Así los landmarks se dibujan en la resolución original y luego se escalan con el frame)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow("Detección de Manos", resized_frame) # Mostrar el frame redimensionado

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()