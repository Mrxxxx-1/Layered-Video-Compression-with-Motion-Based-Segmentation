import numpy as np
import cv2
from numba import njit, prange
import sys

BLOCK_SIZE = 16  # Define as a constant for Numba parallelization


def read_video(filename, width, height):
    with open(filename, "rb") as f:
        data = f.read()
    frame_size = width * height * 3  # For RGB format
    num_frames = len(data) // frame_size
    frames = [
        np.frombuffer(
            data[i * frame_size : (i + 1) * frame_size], dtype=np.uint8
        ).reshape((height, width, 3))
        for i in range(num_frames)
    ]
    return frames


@njit(parallel=True, fastmath=True)
def compute_motion_vectors(curr_frame_r, prev_frame_r, search_range=16):
    BLOCK_SIZE = 16
    h, w = curr_frame_r.shape
    vertical_blocks = h // BLOCK_SIZE
    horizontal_blocks = w // BLOCK_SIZE

    motion_vectors = np.zeros((vertical_blocks, horizontal_blocks, 2), dtype=np.int32)

    # prange over the block indices, not over pixels with a step
    for by in prange(vertical_blocks):  # step of 1, which is allowed
        block_y = by * BLOCK_SIZE
        for bx in range(horizontal_blocks):
            block_x = bx * BLOCK_SIZE
            best_mad = 1e9
            best_dx, best_dy = 0, 0
            curr_block = curr_frame_r[
                block_y : block_y + BLOCK_SIZE, block_x : block_x + BLOCK_SIZE
            ]

            for dy in range(-search_range, search_range + 1):
                ref_y = block_y + dy
                if ref_y < 0 or ref_y > h - BLOCK_SIZE:
                    continue
                for dx in range(-search_range, search_range + 1):
                    ref_x = block_x + dx
                    if ref_x < 0 or ref_x > w - BLOCK_SIZE:
                        continue

                    ref_block = prev_frame_r[
                        ref_y : ref_y + BLOCK_SIZE, ref_x : ref_x + BLOCK_SIZE
                    ]
                    mad = np.mean(np.abs(curr_block - ref_block))
                    if mad < best_mad:
                        best_mad = mad
                        best_dx = dx
                        best_dy = dy

            motion_vectors[by, bx, 0] = best_dx
            motion_vectors[by, bx, 1] = best_dy

    return motion_vectors


def segment_blocks(motion_vectors, threshold=8):
    magnitudes = np.linalg.norm(motion_vectors, axis=2)
    background = magnitudes < threshold
    foreground = ~background
    return background, foreground


def perform_dct_optimized(
    frame, block_size, quant_fg, quant_bg, foreground, background
):
    h, w, _ = frame.shape
    quant_table_fg = np.full((block_size, block_size), 2**quant_fg, dtype=np.float32)
    quant_table_bg = np.full((block_size, block_size), 2**quant_bg, dtype=np.float32)
    compressed = []

    # Process blocks
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            fg_block = foreground[i // block_size, j // block_size]
            bg_block = background[i // block_size, j // block_size]

            # Determine which quantization table to use
            if fg_block:
                quant_table = quant_table_fg
                block_type = 1
            elif bg_block:
                quant_table = quant_table_bg
                block_type = 0
            else:
                # If it's neither foreground nor background, skip (though this case shouldn't happen)
                continue

            dct_coeffs = []
            # Process each color channel
            for c in range(3):
                block = frame[i : i + block_size, j : j + block_size, c]
                dct_block = cv2.dct(block.astype(np.float32))
                quantized = np.round(dct_block / quant_table).astype(np.int32)
                dct_coeffs.append(quantized)
            compressed.append((block_type, np.array(dct_coeffs)))
    return compressed


def pad_frame(frame, block_size):
    """
    Pads the frame so that its dimensions are divisible by block_size.
    """
    h, w, c = frame.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(
        frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
    )


def main():
    if len(sys.argv) != 4:
        print("Usage: myencoder.exe input_video.rgb n1 n2")
        sys.exit(1)

    input_file = sys.argv[1]
    quant_fg = int(sys.argv[2])
    quant_bg = int(sys.argv[3])
    output_file = "output_video_optimized_6_0.cmp"

    # Set the resolution of the input video
    width, height = 960, 540  # Adjust as needed
    frames = read_video(input_file, width, height)

    # Pad frames for block processing
    padded_frames = [pad_frame(frame, BLOCK_SIZE) for frame in frames]
    h, w, _ = padded_frames[0].shape

    compressed_frames = []
    prev_frame = padded_frames[0]

    for idx, frame in enumerate(padded_frames):
        if idx > 0:
            # Compute motion vectors on the luminance component (R channel)
            motion_vectors = compute_motion_vectors(frame[:, :, 0], prev_frame[:, :, 0])

            # Segment into background/foreground blocks
            background, foreground = segment_blocks(motion_vectors)

            # Expand block-level mask to pixel-level mask
            # If DCT block_size differs from motion block_size (16), adjust accordingly
            dct_block_size = 8
            fg_expanded = np.repeat(
                np.repeat(foreground, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1
            )[:h, :w]
            bg_expanded = np.repeat(
                np.repeat(background, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1
            )[:h, :w]

            # Perform DCT and compression with smaller DCT blocks if desired
            compressed_frame = perform_dct_optimized(
                frame,
                block_size=dct_block_size,
                quant_fg=quant_fg,
                quant_bg=quant_bg,
                foreground=fg_expanded,
                background=bg_expanded,
            )
            compressed_frames.append(compressed_frame)

            print(f"Processed frame {idx}")
        prev_frame = frame

    # Write output file
    with open(output_file, "w") as f:
        f.write(f"{quant_fg} {quant_bg}\n")
        for frame_data in compressed_frames:
            for block_type, coeffs in frame_data:
                coeff_str = " ".join(map(str, coeffs.flatten()))
                f.write(f"{block_type} {coeff_str}\n")


if __name__ == "__main__":
    main()
