from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import base64

app = Flask(__name__)
socketio = SocketIO(app)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

model_params = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

hands = mp_hands.Hands(**model_params)

def is_palm_open(landmarks):
    # Define finger tip landmarks
    tip_landmarks = [4, 8, 12, 16, 20]

    # Check if all finger tips are below the corresponding MCP landmarks
    return all(landmarks[tip_landmark].y < landmarks[tip_landmark - 1].y for tip_landmark in tip_landmarks)

@app.route('/')
def index():
    return render_template('index_video.html')

def generate_frames():
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if palm is open
            if is_palm_open(hand_landmarks.landmark):
                palm_status = "Palm is open"
            else:
                palm_status = "Palm is closed"

            cv2.putText(frame, palm_status, (20, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('video_frame', frame_encoded, namespace='/test')
        socketio.emit('palm_status', palm_status, namespace='/test')

        cv2.waitKey(1)

@socketio.on('connect', namespace='/test')
def handle_connect():
    print("Client connected")
    socketio.start_background_task(target=generate_frames)

if __name__ == '__main__':
    socketio.run(app, debug=True)
