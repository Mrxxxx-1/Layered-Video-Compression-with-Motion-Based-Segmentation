import numpy as np
import cv2
import os

# Parameters: set these based on your file
# file_path = "../video/rgbs/SAL.rgb"  # Path to the .rgb file
file_path = "E:/24fall/multimedia/video/rgbs/SAL.rgb"
frame_width = 960  # Frame width in pixels
frame_height = 540  # Frame height in pixels
frame_rate = 30  # Frames per second

# Calculate the size of a single frame in bytes (width * height * 3 for RGB)
frame_size = frame_width * frame_height * 3
# Verify total frames and expected duration
total_bytes = os.path.getsize(file_path)
total_frames = total_bytes // frame_size
playback_duration = total_frames / frame_rate
print(f"Total frames: {total_frames}")
print(f"Expected playback duration: {playback_duration} seconds")

# Open the .rgb file
with open(file_path, "rb") as f:
    while True:
        # Read a single frame
        frame_data = f.read(frame_size)
        if not frame_data:
            break
        
        # Convert raw data to a NumPy array and reshape
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((frame_height, frame_width, 3))
        
        # Fix color if necessary (e.g., RGB to BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Display the frame
        cv2.imshow("RGB Video", frame)
        
        # Exit on 'q'
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
