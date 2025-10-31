"""Restart Flask server script."""
import os
import signal
import sys
import subprocess
import time

print("🔄 Restarting Flask server...")

# Find and kill existing Flask process on port 5000
try:
    # Windows command to find process on port 5000
    result = subprocess.run(
        ['powershell', '-Command', 
         'Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess'],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        pid = int(result.stdout.strip())
        print(f"🛑 Stopping existing server (PID: {pid})...")
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        print("✅ Server stopped")
    else:
        print("ℹ️  No server running on port 5000")
except Exception as e:
    print(f"⚠️  Could not stop existing server: {e}")

print("\n🚀 Starting new server...")
print("=" * 50)

# Start new server
subprocess.run([sys.executable, 'app.py'])
