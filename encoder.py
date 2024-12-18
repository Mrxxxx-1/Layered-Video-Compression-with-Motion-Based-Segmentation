import numpy as np
# from scipy.fftpack import dct
import cv2
from numba import njit, prange
import sys

def read_video(filename, width, height):
    with open(filename, 'rb') as f:
        data = f.read()
    frame_size = width * height * 3  # For RGB format
    num_frames = len(data) // frame_size
    frames = [np.frombuffer(data[i * frame_size:(i + 1) * frame_size], dtype=np.uint8).reshape((height, width, 3)) for i in range(num_frames)]
    return frames

# @njit(parallel=True)
# def compute_motion_vectors(curr_frame_r, prev_frame_r, search_range=16):
#     block_size = 16
#     h, w = curr_frame_r.shape
#     motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)
#     diff = np.abs(curr_frame_r - prev_frame_r)
#     print("Max difference:", np.max(diff))
#     print("Mean difference:", np.mean(diff))
#     for i in prange(h):
#         if i % block_size != 0 or i > h - block_size:
#             continue  # Skip until we reach a block boundary
#         for j in range(w):
#             if j % block_size != 0 or j > w - block_size:
#                 continue  # Skip until we reach a block boundary
                
#             best_mad = float('inf')
#             best_vector = (0, 0)
            
#             for dy in range(-search_range, search_range + 1):
#                 for dx in range(-search_range, search_range + 1):
#                     ref_x, ref_y = j + dx, i + dy
#                     if 0 <= ref_x <= w - block_size and 0 <= ref_y <= h - block_size:
#                         diff = curr_frame_r[i:i + block_size, j:j + block_size] - prev_frame_r[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
#                         mad = np.mean(np.abs(diff))
#                         if mad < best_mad:
#                             best_mad = mad
#                             best_vector = (dx, dy)
                            
#             motion_vectors[i // block_size, j // block_size] = best_vector
    
#     return motion_vectors
@njit
def compute_motion_vectors(curr_frame_r, prev_frame_r, search_range=16, block_size=16):
    h, w = curr_frame_r.shape
    motion_vectors = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)

    for block_y in range(0, h - block_size + 1, block_size):
        for block_x in range(0, w - block_size + 1, block_size):
            best_mad = float('inf')
            best_vector = (0, 0)

            curr_block = curr_frame_r[block_y:block_y + block_size, block_x:block_x + block_size]

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ref_x = block_x + dx
                    ref_y = block_y + dy

                    # Ensure the reference block is within bounds
                    if 0 <= ref_x <= w - block_size and 0 <= ref_y <= h - block_size:
                        ref_block = prev_frame_r[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

                        # Compute MAD (Mean Absolute Difference)
                        mad = np.mean(np.abs(curr_block - ref_block))
                        if mad < best_mad and mad > 1e-3:  # Exclude near-zero MAD
                            best_mad = mad
                            best_vector = (dx, dy)

            # Save the best motion vector
            motion_vectors[block_y // block_size, block_x // block_size] = best_vector
            # print(motion_vectors)
    return motion_vectors



def segment_blocks(motion_vectors, threshold=8):
    magnitudes = np.linalg.norm(motion_vectors, axis=2)
    # print(magnitudes)
    background = magnitudes < threshold
    foreground = ~background
    return background, foreground


def perform_dct_optimized(frame, block_size, quant_fg, quant_bg, foreground, background):
    h, w, _ = frame.shape
    quant_table_fg = np.full((block_size, block_size), 2**quant_fg, dtype=np.float32)
    quant_table_bg = np.full((block_size, block_size), 2**quant_bg, dtype=np.float32)
    compressed = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if foreground[i // block_size, j // block_size]:
                quant_table = quant_table_fg
                block_type = 1  # Foreground block
            elif background[i // block_size, j // block_size]:
                quant_table = quant_table_bg
                block_type = 0  # Background block
            else:
                continue  # Skip if not classified as foreground or background

            dct_coeffs = []
            for c in range(3):  # R, G, B channels
                block = frame[i:i + block_size, j:j + block_size, c]
                dct_block = cv2.dct(block.astype(np.float32))
                quantized = np.round(dct_block / quant_table).astype(np.int32)
                dct_coeffs.append(quantized)

            compressed.append((block_type, np.array(dct_coeffs)))

    return compressed
# def perform_dct_optimized(frame, block_size, quant_fg, quant_bg, foreground):
#     h, w, _ = frame.shape
#     quant_table_fg = np.full((block_size, block_size), 2**quant_fg, dtype=np.float32)
#     quant_table_bg = np.full((block_size, block_size), 2**quant_bg, dtype=np.float32)
#     compressed = []
#     for i in range(0, h, block_size):
#         for j in range(0, w, block_size):
#             is_fg = foreground[i // block_size, j // block_size]
#             quant_table = quant_table_fg if is_fg else quant_table_bg
#             block_type = 1 if is_fg else 0
#             dct_coeffs = []
#             for c in range(3):  # R, G, B channels
#                 block = frame[i:i + block_size, j:j + block_size, c]
#                 dct_block = cv2.dct(block.astype(np.float32))
#                 quantized = np.round(dct_block / quant_table).astype(np.int32)
#                 dct_coeffs.append(quantized)
#             compressed.append((block_type, np.array(dct_coeffs)))
#     return compressed

def pad_frame(frame, block_size):
    """
    Pads the frame to ensure dimensions are divisible by block size.
    """
    h, w, c = frame.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

def main():
    if len(sys.argv) != 4:
        print("Usage: myencoder.exe input_video.rgb n1 n2")
        sys.exit(1)

    input_file = sys.argv[1]
    quant_fg = int(sys.argv[2])
    quant_bg = int(sys.argv[3])
    output_file = 'output_video.cmp'

    width, height = 960, 540  # Assuming fixed resolution; adjust if needed

    frames = read_video(input_file, width, height)
    block_size = 16  # Macroblock size
    padded_frames = [pad_frame(frame, block_size) for frame in frames]
    h, w, _ = padded_frames[0].shape

    compressed_frames = []
    prev_frame = padded_frames[0]
    for idx, frame in enumerate(padded_frames):
        if idx > 0:
            # Compute motion vectors and segment blocks on the padded frame
            motion_vectors = compute_motion_vectors(frame[:, :, 0], prev_frame[:, :, 0])

            # print(motion_vectors)
            background, foreground = segment_blocks(motion_vectors)

            # Adjust segmentation arrays to padded frame dimensions
            foreground = np.repeat(np.repeat(foreground, block_size, axis=0), block_size, axis=1)
            foreground = foreground[:h, :w]  # Match the padded frame size
            background = np.repeat(np.repeat(background, block_size, axis=0), block_size, axis=1)[:h, :w]

            # Perform DCT and compression
            compressed_frame = perform_dct_optimized(frame, block_size=8, quant_fg=quant_fg, quant_bg=quant_bg, foreground=foreground, background=background)
            compressed_frames.append(compressed_frame)

            print(f"Processed frame {idx}")

        prev_frame = frame

    with open(output_file, 'w') as f:
        f.write(f"{quant_fg} {quant_bg}\n")
        for frame_data in compressed_frames:
            for block_type, coeffs in frame_data:
                coeff_str = ' '.join(map(str, coeffs.flatten()))
                f.write(f"{block_type} {coeff_str}\n")

if __name__ == "__main__":
    main()


# Example usage
# main('E:/24fall/multimedia/video/rgbs/SAL.rgb', 'output_video.cmp', width=960, height=540, quant_fg=2, quant_bg=4)
