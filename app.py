from flask import Flask, Response, jsonify, request, render_template
from main import FitnessTrainer
import threading
import time
import os

app = Flask(__name__)

# Global trainer and lock
trainer = None
trainer_lock = threading.Lock()


def get_trainer():
    global trainer
    with trainer_lock:
        if trainer is None:
            trainer = FitnessTrainer(camera_index=int(os.environ.get('CAMERA_INDEX', '1')))
        return trainer


@app.route('/')
def index():
    return render_template('index.html')


def mjpeg_generator():
    t = get_trainer()
    if not getattr(t, 'running', False):
        t.start()
        time.sleep(0.5)

    while True:
        frame = t.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    t = get_trainer()
    return jsonify(t.get_status())


@app.route('/start', methods=['POST'])
def start():
    t = get_trainer()
    t.start()
    return jsonify({'status': 'started'})


@app.route('/stop', methods=['POST'])
def stop():
    t = get_trainer()
    t.stop()
    return jsonify({'status': 'stopped'})


@app.route('/reset', methods=['POST'])
def reset():
    t = get_trainer()
    t.correct = 0
    t.incorrect = 0
    t.stage = None
    t.feedback = ''
    return jsonify({'status': 'reset'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
