import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLineEdit,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap


class AVPlayer(QMainWindow):
    def __init__(self, vfile1=None, vfile2=None, afile=None):
        super().__init__()

        self.setWindowTitle("A/V Player with Dynamic Resolution")
        self.resize(1000, 600)

        # Initialize attributes
        self.video_file_1 = vfile1
        self.video_file_2 = vfile2
        self.audio_file = afile
        self.video_width_1 = 960
        self.video_height_1 = 540
        self.video_width_2 = 960
        self.video_height_2 = 540
        self.frame_rate = 30
        self.frame_timer = None
        self.audio_player = QMediaPlayer()
        self.video_data_1 = None
        self.video_data_2 = None
        self.current_frame = 0

        # Create UI elements
        self.init_ui()

        # Load files if provided
        if self.video_file_1:
            self.video_path_1.setText(self.video_file_1)
            self.load_video(1)
        if self.video_file_2:
            self.video_path_2.setText(self.video_file_2)
            self.load_video(2)
        if self.audio_file:
            self.audio_path.setText(self.audio_file)
            absolute_path = os.path.abspath(self.audio_file)
            resolved_url = QUrl.fromLocalFile(absolute_path)
            self.audio_player.setMedia(QMediaContent(resolved_url))
            # self.audio_player.setMedia(
            #     QMediaContent(QUrl.fromLocalFile(self.audio_file))
            # )

    def init_ui(self):
        # File selection boxes for first video
        self.video_label_1 = QLabel("Video File 1 (.rgb):")
        self.video_path_1 = QLineEdit(self)
        self.video_path_1.setReadOnly(True)
        self.video_button_1 = QPushButton("Select Video 1")
        self.video_button_1.clicked.connect(lambda: self.select_video_file(1))

        # Resolution input for first video
        self.resolution_label_1 = QLabel("Resolution 1 (Width x Height):")
        self.resolution_input_1 = QLineEdit(self)
        self.resolution_input_1.setPlaceholderText("e.g., 960x540")

        # File selection boxes for second video
        self.video_label_2 = QLabel("Video File 2 (.rgb):")
        self.video_path_2 = QLineEdit(self)
        self.video_path_2.setReadOnly(True)
        self.video_button_2 = QPushButton("Select Video 2")
        self.video_button_2.clicked.connect(lambda: self.select_video_file(2))

        # Resolution input for second video
        self.resolution_label_2 = QLabel("Resolution 2 (Width x Height):")
        self.resolution_input_2 = QLineEdit(self)
        self.resolution_input_2.setPlaceholderText("e.g., 960x540")

        # Audio file selection
        self.audio_label = QLabel("Audio File (.wav):")
        self.audio_path = QLineEdit(self)
        self.audio_path.setReadOnly(True)
        self.audio_button = QPushButton("Select Audio")
        self.audio_button.clicked.connect(self.select_audio_file)

        # Player controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.step_frame)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)

        # Video displays
        self.video_display_1 = QLabel(self)
        self.video_display_1.setFixedSize(self.video_width_1, self.video_height_1)
        self.video_display_1.setStyleSheet("background-color: black;")

        self.video_display_2 = QLabel(self)
        self.video_display_2.setFixedSize(self.video_width_2, self.video_height_2)
        self.video_display_2.setStyleSheet("background-color: black;")

        # Layout setup
        video_1_layout = QVBoxLayout()
        video_1_layout.addWidget(self.video_label_1)
        video_1_layout.addWidget(self.video_path_1)
        video_1_layout.addWidget(self.video_button_1)
        video_1_layout.addWidget(self.video_display_1)
        video_1_layout.addWidget(self.resolution_label_1)
        video_1_layout.addWidget(self.resolution_input_1)

        video_2_layout = QVBoxLayout()
        video_2_layout.addWidget(self.video_label_2)
        video_2_layout.addWidget(self.video_path_2)
        video_2_layout.addWidget(self.video_button_2)
        video_2_layout.addWidget(self.video_display_2)
        video_2_layout.addWidget(self.resolution_label_2)
        video_2_layout.addWidget(self.resolution_input_2)

        audio_layout = QHBoxLayout()
        audio_layout.addWidget(self.audio_label)
        audio_layout.addWidget(self.audio_path)
        audio_layout.addWidget(self.audio_button)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.step_button)
        control_layout.addWidget(self.reset_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_1_layout)
        main_layout.addLayout(video_2_layout)

        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(audio_layout)
        final_layout.addLayout(control_layout)

        container = QWidget()
        container.setLayout(final_layout)
        self.setCentralWidget(container)

    def select_video_file(self, video_number):
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Select Video File {video_number}", "", "RGB Files (*.rgb)"
        )
        if file_name:
            if video_number == 1:
                self.video_file_1 = file_name
                self.video_path_1.setText(file_name)
                self.load_video(1)
            elif video_number == 2:
                self.video_file_2 = file_name
                self.video_path_2.setText(file_name)
                self.load_video(2)

    def select_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "WAV Files (*.wav)"
        )
        if file_name:
            self.audio_file = file_name
            self.audio_path.setText(file_name)
            self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))

    def load_video(self, video_number):
        video_file = self.video_file_1 if video_number == 1 else self.video_file_2
        if not video_file:
            return

        try:
            with open(video_file, "rb") as f:
                # Use resolution from user input if provided
                resolution_input = (
                    self.resolution_input_1.text()
                    if video_number == 1
                    else self.resolution_input_2.text()
                )
                if resolution_input:
                    try:
                        width, height = map(int, resolution_input.split("x"))
                        if video_number == 1:
                            self.video_width_1 = width
                            self.video_height_1 = height
                        else:
                            self.video_width_2 = width
                            self.video_height_2 = height
                    except ValueError:
                        print("Invalid resolution format. Using default.")

                frame_size = (
                    self.video_width_1 * self.video_height_1 * 3
                    if video_number == 1
                    else self.video_width_2 * self.video_height_2 * 3
                )
                video_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                    -1,
                    self.video_height_1 if video_number == 1 else self.video_height_2,
                    self.video_width_1 if video_number == 1 else self.video_width_2,
                    3,
                )

                if video_number == 1:
                    self.video_data_1 = video_data
                elif video_number == 2:
                    self.video_data_2 = video_data

                self.update_video_display_size(video_number)
        except Exception as e:
            print(f"Error loading video file {video_number}: {e}")

    def update_video_display_size(self, video_number):
        if video_number == 1:
            self.video_display_1.setFixedSize(self.video_width_1, self.video_height_1)
        elif video_number == 2:
            self.video_display_2.setFixedSize(self.video_width_2, self.video_height_2)
        self.adjustSize()
        print(
            f"Updated video display {video_number} to {self.video_width_1 if video_number == 1 else self.video_width_2}x{self.video_height_1 if video_number == 1 else self.video_height_2}"
        )

    def play(self):
        if (
            self.video_data_1 is None and self.video_data_2 is None
        ) and self.audio_file is None:
            print("Please select at least one file (video or audio).")
            return

        if self.video_data_1 is not None or self.video_data_2 is not None:
            if not self.frame_timer:
                self.frame_timer = QTimer()
                self.frame_timer.timeout.connect(self.update_frame)
            self.frame_timer.start(1000 // self.frame_rate)

        if self.audio_file is not None:
            if self.audio_player.state() != QMediaPlayer.PlayingState:
                self.audio_player.play()

    def pause(self):
        if self.frame_timer:
            self.frame_timer.stop()
        if self.audio_player.state() == QMediaPlayer.PlayingState:
            self.audio_player.pause()

    def step_frame(self):
        self.pause()
        self.update_frame()

    def update_frame(self):
        if self.video_data_1 is None and self.video_data_2 is None:
            return

        if self.current_frame >= max(
            len(self.video_data_1) if self.video_data_1 is not None else 0,
            len(self.video_data_2) if self.video_data_2 is not None else 0,
        ):
            self.stop_playback()
            return

        if self.video_data_1 is not None and self.current_frame < len(
            self.video_data_1
        ):
            frame_1 = self.video_data_1[self.current_frame]
            image_1 = QImage(
                frame_1.data,
                self.video_width_1,
                self.video_height_1,
                QImage.Format_RGB888,
            )
            pixmap_1 = QPixmap.fromImage(image_1)
            self.video_display_1.setPixmap(pixmap_1)

        if self.video_data_2 is not None and self.current_frame < len(
            self.video_data_2
        ):
            frame_2 = self.video_data_2[self.current_frame]
            image_2 = QImage(
                frame_2.data,
                self.video_width_2,
                self.video_height_2,
                QImage.Format_RGB888,
            )
            pixmap_2 = QPixmap.fromImage(image_2)
            self.video_display_2.setPixmap(pixmap_2)

        self.current_frame += 1

    def stop_playback(self):
        if self.frame_timer:
            self.frame_timer.stop()
        if self.audio_player and self.audio_player.state() == QMediaPlayer.PlayingState:
            self.audio_player.stop()

        self.current_frame = 0
        self.video_display_1.clear()
        self.video_display_1.setStyleSheet("background-color: black;")
        self.video_display_2.clear()
        self.video_display_2.setStyleSheet("background-color: black;")
        print("Playback finished. Ready to play again.")

    def reset(self):
        self.pause()

        self.audio_player.stop()
        self.audio_player.setMedia(QMediaContent())

        self.video_path_1.clear()
        self.video_path_2.clear()
        self.audio_path.clear()
        self.resolution_input_1.clear()
        self.resolution_input_2.clear()

        self.video_file_1 = None
        self.video_file_2 = None
        self.audio_file = None
        self.video_data_1 = None
        self.video_data_2 = None
        self.current_frame = 0

        self.video_display_1.clear()
        self.video_display_1.setStyleSheet("background-color: black;")
        self.video_display_2.clear()
        self.video_display_2.setStyleSheet("background-color: black;")
        print("Reset completed. Ready for new file selection.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # player = AVPlayer()
    player = AVPlayer(afile="./Stairs.wav")
    player.show()
    sys.exit(app.exec_())
