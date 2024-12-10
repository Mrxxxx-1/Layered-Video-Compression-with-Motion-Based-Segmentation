import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import QApplication

from comparisonUI import AVPlayer

# from testUI import AVPlayer


def read_compressed_file(input_file):
    """
    Reads the compressed file and extracts quantization values and compressed frames.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    # The first line contains the quantization levels for foreground and background
    quant_fg, quant_bg = map(int, lines[0].split())

    # Each subsequent line represents a block: block_type followed by coefficients
    blocks = []
    for line in lines[1:]:
        parts = line.split()
        block_type = int(parts[0])
        coeffs = np.array(list(map(int, parts[1:])))
        blocks.append((block_type, coeffs))

    return quant_fg, quant_bg, blocks


def reconstruct_frame(
    blocks, frame_height, frame_width, quant_fg, quant_bg, block_size=8
):
    """
    Reconstructs a single frame from blocks using IDCT and dequantization.
    """
    # Create an empty frame
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Define quantization tables based on quantization levels
    quant_table_fg = np.full((block_size, block_size), 2**quant_fg, dtype=np.float32)
    quant_table_bg = np.full((block_size, block_size), 2**quant_bg, dtype=np.float32)

    block_idx = 0
    for i in range(0, frame_height, block_size):
        for j in range(0, frame_width, block_size):
            if block_idx >= len(blocks):
                break

            block_type, coeffs = blocks[block_idx]
            quant_table = quant_table_fg if block_type == 1 else quant_table_bg

            # Reshape the coefficients into 3 channels of 8x8 blocks
            dct_coeffs = coeffs.reshape((3, block_size, block_size))
            for c in range(3):
                # Dequantize and perform inverse DCT
                dequantized = dct_coeffs[c] * quant_table
                idct_block = cv2.idct(dequantized.astype(np.float32))
                idct_block = np.clip(idct_block, 0, 255).astype(np.uint8)
                frame[i : i + block_size, j : j + block_size, c] = idct_block

            block_idx += 1

    return frame


def main():
    if len(sys.argv) != 3:
        print("Usage: mydecoder.exe input_video.cmp input_audio.wav")
        sys.exit(1)

    output_cmp_file = sys.argv[1]
    audio_file = sys.argv[2]
    output_rgb_file = "output_rgb_file.rgb"

    frame_width, frame_height = 960, 544

    block_size = 8
    quant_fg, quant_bg, blocks = read_compressed_file(output_cmp_file)

    # Calculate the number of blocks per frame
    blocks_per_frame = ((frame_height + block_size - 1) // block_size) * (
        (frame_width + block_size - 1) // block_size
    )

    # Group blocks by frame
    frames = [
        blocks[i : i + blocks_per_frame]
        for i in range(0, len(blocks), blocks_per_frame)
    ]

    with open(output_rgb_file, "wb") as f:
        for frame_idx, frame_blocks in enumerate(frames):
            print(f"Decoding frame {frame_idx + 1}/{len(frames)}...")
            frame = reconstruct_frame(
                frame_blocks, frame_height, frame_width, quant_fg, quant_bg, block_size
            )
            # Crop to 960x540 to remove black edges
            frame_cropped = frame[:540, :960]
            # Write raw RGB data to the file
            f.write(frame_cropped.tobytes())

    # Run Comparison Player UI
    app = QApplication(sys.argv)
    player = AVPlayer(vfile2=output_rgb_file, afile=audio_file)
    player.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    # output_cmp_file = "./output_video_sb_sr16.cmp"
    # input_video_file = "./demo/rgbs/WalkingStaticBackground.rgb"
    # output_video_file = "./output_video_sb_sr16.rgb"
    # audio_file = "./demo/wavs/WalkingStaticBackground.wav"

    # # Run decoder
    # main(output_cmp_file, output_video_file, frame_width=960, frame_height=544)

    # # Run Comparison Player UI
    # app = QApplication(sys.argv)
    # player = AVPlayer(vfile2=output_video_file, afile=audio_file)
    # player.show()
    # sys.exit(app.exec_())
