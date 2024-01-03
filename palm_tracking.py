import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

model_params = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

hands = mp_hands.Hands(**model_params)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
video.release()
cv2.destroyAllWindows()
