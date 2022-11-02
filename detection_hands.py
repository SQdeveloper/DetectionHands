from pickle import FALSE
import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5) as hands:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks is not None:
            for hand_landmarks in result.multi_hand_landmarks:
                #dibujamos la plantilla de las manos
                mp_drawing.draw_landmarks(
                     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                #obtenemos la posicion(par ordenado) del dedo indice
                x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

                #movemos el mause en la posicion de las coordenadas del dedo
                #pyautogui.moveTo(x1,y1)
                cv2.circle(frame, (x1,y1), 5, (0,0,255), 3)
        #frame = cv2.flip(frame, 1)
        cv2.imshow("image", frame)
        k = cv2.waitKey(1)
        if k == ord('s'):
            break
cap.release()
cv2.destroyAllWindows()