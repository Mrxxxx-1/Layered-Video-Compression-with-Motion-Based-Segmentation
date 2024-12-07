import numpy as np
import cv2

# Parameters: set these based on your file
# file_path = "../video/rgbs/SAL.rgb"  # Path to the .rgb file
file_path = "E:/24fall/multimedia/video/rgbs/SAL.rgb"
frame_width = 960  # Frame width in pixels
frame_height = 540  # Frame height in pixels
frame_rate = 30  # Frames per second

# Calculate the size of a single frame in bytes (width * height * 3 for RGB)
frame_size = frame_width * frame_height * 3

# Open the .rgb file
with open(file_path, "rb") as f:
    while True:
        # Read a single frame
        frame_data = f.read(frame_size)
        if not frame_data:
            break  # Exit the loop if no more frames

        # Convert the raw data to a NumPy array
        raw_frame = np.frombuffer(frame_data, dtype=np.uint8)
        
        # Extract red, green, and blue channels
        red_channel = raw_frame[0 : frame_width * frame_height].reshape((frame_height, frame_width))
        green_channel = raw_frame[frame_width * frame_height : 2 * frame_width * frame_height].reshape((frame_height, frame_width))
        blue_channel = raw_frame[2 * frame_width * frame_height :].reshape((frame_height, frame_width))

        # Stack the channels along the third dimension to form an RGB image
        frame = np.stack((red_channel, green_channel, blue_channel), axis=-1)
        
        # Display the frame using OpenCV
        cv2.imshow("Planar RGB Video", frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
            break

# Release resources
cv2.destroyAllWindows()
