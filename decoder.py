import numpy as np
import struct
from scipy.fftpack import idct
import imageio

# Constants
BLOCK_SIZE = 8
MACROBLOCK_SIZE = 16
FRAME_WIDTH = 512   # Adjust according to input
FRAME_HEIGHT = 512  # Adjust according to input

def read_compressed_file(file_path):
    with open(file_path, 'rb') as f:
        # Read quantization steps
        n1, n2 = struct.unpack('ii', f.read(8))
        compressed_data = f.read()  # Read the rest of the data
    return n1, n2, compressed_data

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def dequantize_block(block, q_step):
    return block * (2 ** q_step)

def process_macroblock(block_data, quantization_step):
    blocks = []
    offset = 0
    for _ in range((MACROBLOCK_SIZE // BLOCK_SIZE) ** 2):
        block = np.frombuffer(block_data[offset:offset + BLOCK_SIZE * BLOCK_SIZE * 2], dtype=np.int16).reshape((BLOCK_SIZE, BLOCK_SIZE))
        dequantized_block = dequantize_block(block, quantization_step)
        idct_block = idct2(dequantized_block)
        blocks.append(idct_block)
        offset += BLOCK_SIZE * BLOCK_SIZE * 2
    return blocks

def reconstruct_frame(compressed_data, n1, n2, frame_count):
    offset = 0
    frames = []

    for _ in range(frame_count):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        for i in range(0, FRAME_HEIGHT, MACROBLOCK_SIZE):
            for j in range(0, FRAME_WIDTH, MACROBLOCK_SIZE):
                for c in range(3):  # For each color channel
                    block_type = struct.unpack('B', compressed_data[offset:offset + 1])[0]
                    offset += 1
                    
                    quant_step = n1 if block_type == 1 else n2
                    block_data = compressed_data[offset:offset + ((MACROBLOCK_SIZE // BLOCK_SIZE) ** 2) * (BLOCK_SIZE ** 2 * 2)]
                    offset += ((MACROBLOCK_SIZE // BLOCK_SIZE) ** 2) * (BLOCK_SIZE ** 2 * 2)
                    
                    blocks = process_macroblock(block_data, quant_step)
                    
                    # Reconstruct macroblock
                    mb = np.zeros((MACROBLOCK_SIZE, MACROBLOCK_SIZE))
                    idx = 0
                    for bi in range(0, MACROBLOCK_SIZE, BLOCK_SIZE):
                        for bj in range(0, MACROBLOCK_SIZE, BLOCK_SIZE):
                            mb[bi:bi+BLOCK_SIZE, bj:bj+BLOCK_SIZE] = blocks[idx]
                            idx += 1
                    frame[i:i+MACROBLOCK_SIZE, j:j+MACROBLOCK_SIZE, c] = np.clip(mb, 0, 255)

        frames.append(frame)
    return frames

def save_video(frames, output_file):
    # imageio.mimsave(output_file, frames, fps=30)
    imageio.mimsave(output_file, frames)


if __name__ == "__main__":
    input_file = "output_video.cmp"  # Compressed file path
    output_file = "output_video.mp4"  # Output video file
    num_frames = 30  # Adjust based on your input video
    
    n1, n2, compressed_data = read_compressed_file(input_file)
    frames = reconstruct_frame(compressed_data, n1, n2, num_frames)
    save_video(frames, output_file)
    print(f"Decoded video saved as {output_file}")
