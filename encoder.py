import numpy as np
from scipy.fftpack import dct
import cv2

def read_video(filename, width, height):
    with open(filename, 'rb') as f:
        data = f.read()
    frame_size = width * height * 3  # For RGB format
    num_frames = len(data) // frame_size
    frames = [np.frombuffer(data[i * frame_size:(i + 1) * frame_size], dtype=np.uint8).reshape((height, width, 3)) for i in range(num_frames)]
    return frames

def compute_motion_vectors(curr_frame, prev_frame, block_size=16, search_range=8):
    h, w = curr_frame.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            best_mad = float('inf')
            best_vector = (0, 0)
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_x, ref_y = j + dx, i + dy
                    if 0 <= ref_x <= w - block_size and 0 <= ref_y <= h - block_size:
                        diff = curr_frame[i:i + block_size, j:j + block_size] - prev_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        mad = np.mean(np.abs(diff))
                        if mad < best_mad:
                            best_mad = mad
                            best_vector = (dx, dy)
            motion_vectors[i // block_size, j // block_size] = best_vector
    return motion_vectors

def segment_blocks(motion_vectors, threshold=1):
    magnitudes = np.linalg.norm(motion_vectors, axis=2)
    background = magnitudes < threshold
    foreground = ~background
    return background, foreground

def perform_dct_optimized(frame, block_size, quant_fg, quant_bg, foreground):
    h, w, _ = frame.shape
    quant_table_fg = np.full((block_size, block_size), 2**quant_fg, dtype=np.float32)
    quant_table_bg = np.full((block_size, block_size), 2**quant_bg, dtype=np.float32)
    compressed = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            is_fg = foreground[i // block_size, j // block_size]
            quant_table = quant_table_fg if is_fg else quant_table_bg
            block_type = 1 if is_fg else 0
            dct_coeffs = []
            for c in range(3):  # R, G, B channels
                block = frame[i:i + block_size, j:j + block_size, c]
                dct_block = cv2.dct(block.astype(np.float32))
                quantized = np.round(dct_block / quant_table).astype(np.int32)
                dct_coeffs.append(quantized)
            compressed.append((block_type, np.array(dct_coeffs)))
    return compressed

def pad_frame(frame, block_size):
    """
    Pads the frame to ensure dimensions are divisible by block size.
    """
    h, w, c = frame.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

def main(input_file, output_file, width, height, quant_fg, quant_bg):
    n = 0
    frames = read_video(input_file, width, height)
    block_size = 16  # Macroblock size
    padded_frames = [pad_frame(frame, block_size) for frame in frames]
    h, w, _ = padded_frames[0].shape

    compressed_frames = []
    prev_frame = padded_frames[0]
    for idx, frame in enumerate(padded_frames):
        if idx > 0:
            
            # Compute motion vectors and segment blocks on the padded frame
            motion_vectors = compute_motion_vectors(
                frame[:, :, 0], prev_frame[:, :, 0], block_size=block_size
            )
            # Use dimensions of the motion vector grid for segmentation
            num_blocks_h, num_blocks_w = motion_vectors.shape[:2]
            background, foreground = segment_blocks(motion_vectors)

            # Adjust segmentation arrays to padded frame dimensions
            foreground = np.repeat(
                np.repeat(foreground, block_size, axis=0), block_size, axis=1
            )
            foreground = foreground[:h, :w]  # Match the padded frame size

            # Perform DCT and compression
            compressed_frame = perform_dct_optimized(
                frame, block_size=8, quant_fg=quant_fg, quant_bg=quant_bg, foreground=foreground
            )
            compressed_frames.append(compressed_frame)
            print("Foreground shape:", foreground.shape)
            print("Padded frame shape:", frame.shape)
            # print(compressed_frame)
            n = n + 1
            print(n)
        prev_frame = frame

    with open(output_file, 'w') as f:
        f.write(f"{quant_fg} {quant_bg}\n")
        for frame_data in compressed_frames:
            for block_type, coeffs in frame_data:
                coeff_str = ' '.join(map(str, coeffs.flatten()))
                f.write(f"{block_type} {coeff_str}\n")



# Example usage
main('E:/24fall/multimedia/video/rgbs/SAL.rgb', 'output_video.cmp', width=960, height=540, quant_fg=2, quant_bg=4)
