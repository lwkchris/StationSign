# demo.py
from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from Prediction import StationSignRecognition
import time

class StationSignRecognitionApp:
    def __init__(self, model_path, data_path):
        self.app = Flask(__name__)
        self.model_path = model_path
        self.data_path = data_path
        self.recognition = StationSignRecognition(model_path=self.model_path, data_path=self.data_path)

        self.video_capture = None
        self.video_thread = None
        self.frame = None
        self.predictions = []  # Store predictions
        self.lock = threading.Lock()

        self.last_prediction_time = 0  # Timestamp of the last valid prediction
        self.waiting_message_sent = False  # Flag to avoid redundant "Waiting" messages

        self.register_routes()

    def process_video(self):
        """Background thread to process video frames."""
        while True:
            if self.video_capture:
                ret, current_frame = self.video_capture.read()
                if not ret:
                    break

                with self.lock:
                    self.frame = current_frame.copy()

                # Generate predictions
                new_predictions = self.recognition.process_frame(current_frame)

                with self.lock:
                    if new_predictions:  # Update predictions and timestamp if new predictions are found
                        self.predictions = new_predictions
                        self.last_prediction_time = time.time()
                        self.waiting_message_sent = False  # Reset waiting message flag
                    else:
                        # Check if 3 seconds have passed since the last prediction
                        if time.time() - self.last_prediction_time >= 3:
                            # Send "Waiting for detecting" message if it hasn't already been sent
                            if not self.waiting_message_sent:
                                self.predictions = [["Waiting for detecting"]]  # Send only the message
                                self.waiting_message_sent = True

    def register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/start', methods=['POST'])
        def start_video():
            if self.video_capture is None:
                self.video_capture = cv2.VideoCapture(0)  # Use webcam
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.start()
            return jsonify({'status': 'Video streaming started'})

        @self.app.route('/stop', methods=['POST'])
        def stop_video():
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
                self.video_thread = None
            return jsonify({'status': 'Video streaming stopped'})

        @self.app.route('/predictions', methods=['GET'])
        def get_predictions():
            """Return the latest predictions."""
            with self.lock:
                return jsonify(self.predictions)

        @self.app.route('/video_feed')
        def video_feed():
            def generate():
                while True:
                    if self.frame is not None:
                        _, buffer = cv2.imencode('.jpg', self.frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self, debug=True):
        self.app.run(debug=debug)


if __name__ == '__main__':
    MODEL_PATH = "model.pth"
    DATA_PATH = "data"
    app = StationSignRecognitionApp(model_path=MODEL_PATH, data_path=DATA_PATH)
    app.run()