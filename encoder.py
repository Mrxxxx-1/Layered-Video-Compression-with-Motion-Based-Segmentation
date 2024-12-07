import numpy as np
import imageio
import struct
from scipy.fftpack import dct, idct

# Constants
BLOCK_SIZE = 8
MACROBLOCK_SIZE = 16
FRAME_WIDTH = 512   # Adjust according to input
FRAME_HEIGHT = 512  # Adjust according to input

def read_rgb_video(file_path, num_frames, width, height):
    frames = []
    with open(file_path, 'rb') as f:
        for _ in range(num_frames):
            r = np.frombuffer(f.read(width * height), dtype=np.uint8).reshape(height, width)
            g = np.frombuffer(f.read(width * height), dtype=np.uint8).reshape(height, width)
            b = np.frombuffer(f.read(width * height), dtype=np.uint8).reshape(height, width)
            frames.append(np.stack((r, g, b), axis=-1))
    return frames

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def quantize_block(block, q_step):
    return np.round(block / (2 ** q_step))

def process_macroblock(macroblock, quantization_step):
    dct_blocks = []
    for i in range(0, MACROBLOCK_SIZE, BLOCK_SIZE):
        for j in range(0, MACROBLOCK_SIZE, BLOCK_SIZE):
            block = macroblock[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            block_dct = dct2(block)
            quantized_block = quantize_block(block_dct, quantization_step)
            dct_blocks.append(quantized_block)
    return dct_blocks

def encode_video(input_file, output_file, n1, n2, num_frames):
    frames = read_rgb_video(input_file, num_frames, FRAME_WIDTH, FRAME_HEIGHT)
    
    with open(output_file, 'wb') as out:
        out.write(struct.pack('ii', n1, n2))  # Write quantization values
        
        for frame_idx, frame in enumerate(frames):
            print(f'Processing frame {frame_idx + 1}/{num_frames}')
            for i in range(0, FRAME_HEIGHT, MACROBLOCK_SIZE):
                for j in range(0, FRAME_WIDTH, MACROBLOCK_SIZE):
                    # Extract macroblock
                    macroblock = frame[i:i+MACROBLOCK_SIZE, j:j+MACROBLOCK_SIZE, :]
                    
                    # For simplicity, assume foreground for demonstration (motion detection required)
                    block_type = 1  # 1 for foreground, 0 for background
                    
                    # Process each color channel separately
                    for c in range(3):
                        blocks = process_macroblock(macroblock[:, :, c], n1 if block_type else n2)
                        out.write(struct.pack('B', block_type))  # Write block type
                        for block in blocks:
                            out.write(block.astype(np.int16).tobytes())  # Write quantized coefficients

if __name__ == "__main__":
    input_file = "../video/rgbs/SAL.rgb"  # Replace with your file path
    output_file = "output_video.cmp"
    n1 = 2  # Foreground quantization step
    n2 = 4  # Background quantization step
    num_frames = 30  # Replace with the actual number of frames in your video
    
    encode_video(input_file, output_file, n1, n2, num_frames)
