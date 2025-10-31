# AI Fitness Trainer Pro

An AI-powered fitness trainer application that uses computer vision and MediaPipe Pose to track and analyze exercises in real-time, providing intelligent feedback on form and technique. Built for SIH 2025.

## 🎯 Overview

This application combines computer vision, pose estimation, and real-time video processing to create an intelligent fitness coaching system. It supports multiple exercise modes (squats and vertical jumps) with detailed form analysis, validation, and performance tracking through an intuitive web interface.

## ✨ Features

### Squat Counter Mode
- Real-time squat detection and counting using MediaPipe Pose
- Form analysis with feedback on posture
- Tracks correct and incorrect squats
- Text-to-speech feedback (optional)

### Vertical Jump Mode
- Measures vertical jump height in inches
- Detects cheating (landing in squat position)
- Hand gesture controls (touch nose to arm/reset)
- Real-time jump validation
- Airborne detection and landing analysis

### General Features
- Dual-mode operation (switch between squats and jumps)
- Web-based interface with live video feed
- Session logging capabilities
- REST API for integration with other systems
- Clean single-overlay display

## Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Windows/Linux/macOS

## Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd "c:\Users\JHASHANK\Desktop\SIH 2025\SIH_SQUATS"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Application (Recommended)

1. **Start the Flask web server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **Use the web interface:**
   - **Select Mode:** Choose "Squat Counter" or "Vertical Jump" from dropdown
   - Click "Start" to begin tracking
   - Click "Stop" to pause
   - Click "Reset" to reset counters

### Squat Mode Usage
- Stand in front of camera with full body visible
- Perform squats with proper form
- App counts correct and incorrect squats
- Feedback appears at bottom of screen

### Jump Mode Usage
1. **ARM the system:** Touch your RIGHT hand to your nose
2. **Jump:** Perform your vertical jump
3. **See Results:** Jump height displayed with validation status
4. **Reset:** Touch your LEFT hand to your nose to reset for next jump

**Jump Tips:**
- Make sure baseline is set (hand to nose) before jumping
- Land with legs straight to avoid "cheat" detection
- Minimum jump height: 2 inches
- Camera calibration: 100 pixels = 12 inches (adjustable in code)

### Standalone Application

Run the trainer directly with OpenCV window:

```bash
python main.py
```

**Command-line options:**
```bash
python main.py --help

Options:
  --index INT           Camera index (default: 0)
  --backend {dshow,msmf} Camera backend
  --width INT           Frame width (default: 640)
  --height INT          Frame height (default: 480)
  --fullscreen          Open in fullscreen mode
  --leniency {A,B,custom} Counting leniency preset
  --dwell-ms INT        Hold time in S3 before counting (ms)
  --s3-cutoff INT       Knee angle cutoff for squat bottom
  --min-knee INT        Min knee angle for valid squat
  --back-min INT        Minimum back angle
  --back-max INT        Maximum back angle
```

**Example:**
```bash
python main.py --index 1 --leniency B --fullscreen
```

## Configuration

### Camera Settings

- By default, the app uses camera index 0
- To use a different camera, set the `CAMERA_INDEX` environment variable:
  ```bash
  $env:CAMERA_INDEX = "1"  # PowerShell
  set CAMERA_INDEX=1       # Command Prompt
  ```

### Backend Integration

- Set `BACKEND_URL` environment variable to enable remote session tracking:
  ```bash
  $env:BACKEND_URL = "http://127.0.0.1:5000"
  ```

## API Endpoints

When running the Flask app:

- `GET /` - Web interface
- `GET /video_feed?mode=squat|jump` - MJPEG video stream
- `GET /status?mode=squat|jump` - Get current stats (JSON)
- `POST /start` - Start tracking (body: `{"mode": "squat|jump"}`)
- `POST /stop` - Stop tracking (body: `{"mode": "squat|jump"}`)
- `POST /reset` - Reset counters (body: `{"mode": "squat|jump"}`)
- `POST /switch_mode` - Switch between modes (body: `{"mode": "squat|jump"}`)

### Response Examples

**Squat Mode Status:**
```json
{
  "correct": 5,
  "incorrect": 2,
  "stage": "S1",
  "feedback": "Good form!",
  "camera_ready": true,
  "mode": "squat"
}
```

**Jump Mode Status:**
```json
{
  "state": "landed",
  "last_jump_inches": 24.5,
  "last_jump_valid": true,
  "last_jump_reason": "ok",
  "feedback": "VALID JUMP: 24.5 inches!",
  "camera_ready": true,
  "mode": "jump"
}
```

## Troubleshooting

### Import Errors
If you see import errors like "Import 'cv2' could not be resolved":
- Ensure you've activated the virtual environment
- Run `pip install -r requirements.txt`
- Restart VS Code/your IDE

### Camera Not Found
- Check if your camera is connected and working
- Try different camera indices with `--index` parameter
- On Windows, try `--backend dshow` or `--backend msmf`

### TTS Not Working
- The app will continue to work without TTS
- On Windows, ensure SAPI5 is available
- On Linux, install `espeak`: `sudo apt-get install espeak`

## Project Structure

```
SIH_SQUATS/
├── app.py              # Flask web application
├── main.py             # Core fitness trainer logic
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css      # Stylesheet
└── templates/
    └── index.html     # Web interface
```

## Dependencies

- Flask - Web framework
- OpenCV - Computer vision
- MediaPipe - Pose detection
- NumPy - Numerical operations
- pyttsx3 - Text-to-speech
- requests - HTTP client (optional)

## License

This project is part of SIH 2025.

## Authors

- Team SIH_SQUATS
