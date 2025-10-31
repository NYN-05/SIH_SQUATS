from flask import Flask, Response, jsonify, request, render_template
from main import FitnessTrainer, JumpTrainer
import threading
import time
import os
import atexit

# Load .env file before Flask tries to do it (prevents Windows console error)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

app = Flask(__name__)
# Disable Flask's own dotenv loading to prevent Windows console errors
app.config['ENV'] = 'development'

# Global trainers and locks
squat_trainer = None
jump_trainer = None
trainer_lock = threading.Lock()
current_mode = 'squat'  # 'squat' or 'jump'
active_generators = {}  # Track active MJPEG generators


def validate_mode(mode):
    """Validate and normalize mode parameter."""
    if mode not in ['squat', 'jump']:
        return 'squat'
    return mode


def cleanup_resources():
    """Cleanup resources on shutdown."""
    global squat_trainer, jump_trainer
    print("🧹 Cleaning up resources...")
    if squat_trainer:
        try:
            squat_trainer.release()
        except Exception as e:
            print(f"Error releasing squat trainer: {e}")
    if jump_trainer:
        try:
            jump_trainer.release()
        except Exception as e:
            print(f"Error releasing jump trainer: {e}")


# Register cleanup function
atexit.register(cleanup_resources)


def get_trainer(mode='squat'):
    global squat_trainer, jump_trainer, current_mode
    mode = validate_mode(mode)
    
    with trainer_lock:
        if mode == 'squat':
            if squat_trainer is None:
                try:
                    camera_idx = int(os.environ.get('CAMERA_INDEX', '0'))
                    print(f"🎥 Initializing squat trainer with camera index {camera_idx}")
                    squat_trainer = FitnessTrainer(camera_index=camera_idx)
                    if not squat_trainer.camera_initialized:
                        print(f"⚠️  Camera not initialized for squat trainer")
                        return None
                    print(f"✅ Squat trainer created successfully")
                except Exception as e:
                    print(f"❌ Error creating squat trainer: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            return squat_trainer
        else:  # jump mode
            if jump_trainer is None:
                try:
                    camera_idx = int(os.environ.get('CAMERA_INDEX', '0'))
                    print(f"🎥 Initializing jump trainer with camera index {camera_idx}")
                    jump_trainer = JumpTrainer(camera_index=camera_idx)
                    if not jump_trainer.camera_initialized:
                        print(f"⚠️  Camera not initialized for jump trainer")
                        return None
                    print(f"✅ Jump trainer created successfully")
                except Exception as e:
                    print(f"❌ Error creating jump trainer: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            return jump_trainer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """Return a simple SVG favicon to prevent 404 errors."""
    svg_data = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
        <path fill="#4F46E5" d="M16 4L4 10V22L16 28L28 22V10L16 4Z"/>
        <path fill="#fff" d="M16 16L4 10M16 16L28 10M16 16V28" stroke="#fff" stroke-width="2"/>
    </svg>'''
    return Response(svg_data, mimetype='image/svg+xml')


def mjpeg_generator(mode='squat', generator_id=None):
    """Generate MJPEG stream with proper error handling and cleanup."""
    mode = validate_mode(mode)
    t = get_trainer(mode)
    
    if t is None:
        # Return error frame
        import cv2
        import numpy as np
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Error: Cannot initialize camera", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        error_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_bytes + b'\r\n')
            time.sleep(1)
        return
    
    if not getattr(t, 'running', False):
        try:
            t.start()
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error starting trainer: {e}")
            return
    
    try:
        while True:
            # Check if generator should stop (mode switched)
            if generator_id and generator_id not in active_generators:
                print(f"🛑 Generator {generator_id} stopped (mode switched)")
                break
                
            try:
                frame = t.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"❌ Frame generation error: {e}")
                time.sleep(0.1)
    except GeneratorExit:
        print(f"🛑 Generator {generator_id} exited")
        if generator_id and generator_id in active_generators:
            del active_generators[generator_id]
    except Exception as e:
        print(f"❌ Generator error: {e}")


@app.route('/video_feed')
def video_feed():
    mode = request.args.get('mode', 'squat')
    mode = validate_mode(mode)
    
    # Create unique generator ID
    generator_id = f"{mode}_{time.time()}"
    active_generators[generator_id] = mode
    
    return Response(mjpeg_generator(mode, generator_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    try:
        mode = request.args.get('mode', current_mode)
        mode = validate_mode(mode)
        t = get_trainer(mode)
        
        if t is None:
            return jsonify({
                'error': 'Trainer not initialized',
                'mode': mode,
                'camera_ready': False
            }), 500
        
        result = t.get_status()
        result['mode'] = mode
        return jsonify(result)
    except Exception as e:
        print(f"❌ Status error: {e}")
        return jsonify({'error': str(e), 'mode': mode, 'camera_ready': False}), 500


@app.route('/start', methods=['POST'])
def start():
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'squat')
        mode = validate_mode(mode)
        t = get_trainer(mode)
        
        if t is None:
            return jsonify({'status': 'error', 'message': 'Cannot initialize trainer', 'mode': mode}), 500
        
        t.start()
        return jsonify({'status': 'started', 'mode': mode})
    except Exception as e:
        print(f"❌ Start error: {e}")
        return jsonify({'status': 'error', 'message': str(e), 'mode': mode}), 500


@app.route('/stop', methods=['POST'])
def stop():
    try:
        data = request.get_json() or {}
        mode = data.get('mode', current_mode)
        mode = validate_mode(mode)
        t = get_trainer(mode)
        
        if t is None:
            return jsonify({'status': 'error', 'message': 'Trainer not found', 'mode': mode}), 500
        
        t.stop()
        return jsonify({'status': 'stopped', 'mode': mode})
    except Exception as e:
        print(f"❌ Stop error: {e}")
        return jsonify({'status': 'error', 'message': str(e), 'mode': mode}), 500


@app.route('/reset', methods=['POST'])
def reset():
    try:
        data = request.get_json() or {}
        mode = data.get('mode', current_mode)
        mode = validate_mode(mode)
        t = get_trainer(mode)
        
        if t is None:
            return jsonify({'status': 'error', 'message': 'Trainer not found', 'mode': mode}), 500
        
        with trainer_lock:
            if mode == 'squat':
                t.correct = 0
                t.incorrect = 0
                t.stage = None
                t.feedback = ''
            else:  # jump mode
                t.state = 'idle'
                t.baseline_nose_y = None
                t.peak_delta_pixels = 0.0
                t.last_jump_inches = 0.0
                t.last_jump_valid = False
                t.last_jump_reason = ''
                t.nose_y_buffer = []
                t.airborne_frames = 0
                t.feedback = 'Touch right hand to nose to ARM'
        
        return jsonify({'status': 'reset', 'mode': mode})
    except Exception as e:
        print(f"❌ Reset error: {e}")
        return jsonify({'status': 'error', 'message': str(e), 'mode': mode}), 500


@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    global current_mode, active_generators
    
    try:
        data = request.get_json() or {}
        new_mode = data.get('mode', 'squat')
        new_mode = validate_mode(new_mode)
        
        if new_mode == current_mode:
            return jsonify({'status': 'already_active', 'mode': new_mode})
        
        with trainer_lock:
            # Stop current trainer
            try:
                if current_mode == 'squat' and squat_trainer:
                    squat_trainer.stop()
                    print(f"⏹️  Stopped squat trainer")
                elif current_mode == 'jump' and jump_trainer:
                    jump_trainer.stop()
                    print(f"⏹️  Stopped jump trainer")
            except Exception as e:
                print(f"⚠️  Error stopping trainer: {e}")
            
            # Clear active generators to stop old streams
            old_generators = list(active_generators.keys())
            for gen_id in old_generators:
                if current_mode in gen_id:
                    del active_generators[gen_id]
            
            # Switch mode
            old_mode = current_mode
            current_mode = new_mode
            
            print(f"🔄 Switched from {old_mode} to {new_mode}")
        
        # Give time for resources to release
        time.sleep(0.3)
        
        return jsonify({'status': 'switched', 'mode': new_mode, 'previous_mode': old_mode})
    except Exception as e:
        print(f"❌ Switch mode error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Disable Flask's auto-reload to avoid Windows console errors
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
