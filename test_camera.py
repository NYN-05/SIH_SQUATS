"""Quick camera test script to verify camera functionality."""
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

camera_index = int(os.environ.get('CAMERA_INDEX', '0'))
print(f"🎥 Testing camera at index {camera_index}...")

# Try to open camera
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"❌ Cannot open camera at index {camera_index}")
    print("Trying other indices...")
    
    for idx in range(0, 3):
        print(f"   Trying index {idx}...")
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                print(f"   ✅ Camera {idx} works! Frame shape: {frame.shape}")
            else:
                print(f"   ⚠️  Camera {idx} opened but cannot read frames")
            test_cap.release()
        else:
            print(f"   ❌ Camera {idx} not available")
else:
    print(f"✅ Camera opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame read successfully! Shape: {frame.shape}")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   Channels: {frame.shape[2]}")
    else:
        print("❌ Cannot read frame from camera")
    
    cap.release()

print("\n🎥 Camera test complete!")
