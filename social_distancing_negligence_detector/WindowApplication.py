# QT imports
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

# System imports
import sys

# OpenCV imports
import cv2
import numpy as np

# Submodule imports
from Detector import Detector, Models


class DetectorThread(QThread):
    update_image_signal = pyqtSignal(np.ndarray)
    video_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.Detector = Detector()
        self.current_frame_count = 0
        self.is_running = False
        self.is_paused = False
        self.is_video_finished = True
        self.video_path = ""
        self.model_name = ""

    def set_video_path(self, video_path):
        self.reset()
        self.video_path = video_path

    def set_model_name(self, model_name):
        self.reset()
        self.model_name = model_name

    def initialize_detector(self):
        if self.is_video_finished:
            self.Detector.initialize(self.model_name, self.video_path)

    def run(self):
        self.is_running = True
        while self.is_running:
            while not self.is_paused:
                can_analyze, analyzed_frame = self.Detector.analyze_frame(self.current_frame_count)
                self.current_frame_count += 1
                if can_analyze:
                    self.is_video_finished = False
                    if analyzed_frame is not None:
                        self.update_image_signal.emit(analyzed_frame)
                else:
                    self.video_finished_signal.emit()
                    self.is_paused = True

    def reset(self):
        self.is_video_finished = True
        self.current_frame_count = 0

    def pause(self):
        self.is_paused = True


class WindowApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.is_initialized = False
        self.initialize_window()
        self.initialize_data()
        self.initialize_components()
        self.initialize_layouts()
        self.initialize_video_thread()

        self.setWindowIcon(QIcon('Resources/icon.ico'))

        self.is_initialized = True

    def initialize_window(self):
        assert not self.is_initialized

        self.setWindowTitle("Social Distancing Negligence Detector")
        self.display_width = 925
        self.display_height = 850
        self.setMaximumSize(self.display_width, self.display_height)
        self.setFixedWidth(self.display_width)
        self.setFixedHeight(self.display_height)

    def initialize_data(self):
        assert not self.is_initialized

    def initialize_components(self):
        assert not self.is_initialized

        # Title text
        self.title_text = QLabel()
        self.title_text.setText("Social Distancing Negligence Detector")
        self.title_text.setFont(QFont("Segoe UI", 24))
        self.title_text.setAlignment(Qt.AlignHCenter)

        # Video mode buttons
        self.video_mode_recorded_button = QRadioButton()
        self.video_mode_recorded_button.setText("Recorded")
        self.video_mode_recorded_button.setChecked(True)
        self.video_mode_recorded_button.toggled.connect(self.set_video_mode_recorded)
        self.video_mode_live_button = QRadioButton()
        self.video_mode_live_button.setText("Live")
        self.video_mode_live_button.toggled.connect(self.set_video_mode_live)

        # Video file input
        self.video_input_label = QLabel()
        self.video_input_label.setText("Video Path:")
        self.video_input_label.setFixedWidth(100)
        self.video_input_line_edit = QLineEdit()
        self.video_input_line_edit.setEnabled(False)
        self.video_input_browse_button = QPushButton()
        self.video_input_browse_button.setText("Browse")
        self.video_input_browse_button.clicked.connect(self.show_file_selection_dialog)

        # Model input
        self.model_input_label = QLabel()
        self.model_input_label.setText("Model: ")
        self.model_input_label.setFixedWidth(100)
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(Models().get_models())
        self.model_combo_box.currentTextChanged.connect(self.on_model_combobox_changed)

        # Start button
        self.start_button = QPushButton()
        self.start_button.setText("Start")
        self.start_button.clicked.connect(self.start_video)
        self.start_button.setEnabled(False)

        # Pause button
        self.pause_button = QPushButton()
        self.pause_button.setText("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setEnabled(False)

        # File selection dialog
        self.video_selection_dialog = QFileDialog()
        self.video_selection_dialog.setDirectory("./Inputs")
        self.video_selection_dialog.setWindowTitle("Select a video file")

        # Video frames
        self.video_width = 900
        self.video_height = 575
        self.frame_label = QLabel()
        self.frame_label.setFixedWidth(self.video_width)
        self.frame_label.setFixedHeight(self.video_height)
        self.frame_label.setStyleSheet("background-color: lightgray")
        self.frame_title_label = QLabel()
        self.frame_title_label.setText("Video")
        self.frame_title_label.setFont(QFont("Segoe UI", 16))
        self.frame_title_label.setAlignment(Qt.AlignHCenter)

    def initialize_layouts(self):
        assert not self.is_initialized

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)

        title_layout = QGridLayout()
        title_layout.addWidget(self.title_text)

        input_contents_layout = QVBoxLayout()
        input_contents_layout.setContentsMargins(50, 20, 50, 20)

        video_mode_layout = QHBoxLayout()
        video_mode_layout.setContentsMargins(0, 5, 0, 5)
        video_mode_layout.setAlignment(Qt.AlignRight)
        video_mode_layout.addWidget(self.video_mode_recorded_button)
        video_mode_layout.addWidget(self.video_mode_live_button)

        video_input_layout = QHBoxLayout()
        video_input_layout.setContentsMargins(0, 5, 0, 5)
        video_input_layout.addWidget(self.video_input_label)
        video_input_layout.addWidget(self.video_input_line_edit)
        video_input_layout.addWidget(self.video_input_browse_button)

        model_input_layout = QHBoxLayout()
        model_input_layout.setContentsMargins(0, 5, 0, 5)
        model_input_layout.addWidget(self.model_input_label)
        model_input_layout.addWidget(self.model_combo_box)

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setContentsMargins(0, 5, 0, 5)
        action_buttons_layout.setAlignment(Qt.AlignRight)
        action_buttons_layout.addWidget(self.start_button)
        action_buttons_layout.addWidget(self.pause_button)

        video_frames_layout = QVBoxLayout()
        video_frames_layout.addWidget(self.frame_title_label)
        video_frames_layout.addWidget(self.frame_label)
        video_frames_layout.setAlignment(self.frame_label, Qt.AlignCenter)

        input_contents_layout.addLayout(video_mode_layout)
        input_contents_layout.addLayout(video_input_layout)
        input_contents_layout.addLayout(model_input_layout)
        input_contents_layout.addLayout(action_buttons_layout)

        main_layout.addLayout(title_layout)
        main_layout.addLayout(input_contents_layout)
        main_layout.addLayout(video_frames_layout)
        self.setLayout(main_layout)

    def initialize_video_thread(self):
        self.thread = DetectorThread()
        self.thread.set_model_name(Models().get_models()[0])
        self.thread.update_image_signal.connect(self.update_image)
        self.thread.video_finished_signal.connect(self.on_video_finished)

    def set_video_mode_recorded(self):
        self.video_input_browse_button.setEnabled(True)

    def set_video_mode_live(self):
        self.video_input_browse_button.setEnabled(False)
        self.start_video()

    def show_file_selection_dialog(self):
        file_path, _ = self.video_selection_dialog.getOpenFileName(filter="*.mp4")

        if file_path != "":
            self.video_input_line_edit.setText(file_path[file_path.rfind("/") + 1:])
            self.thread.set_video_path(file_path)
            self.start_button.setText("Start")
            self.start_button.setEnabled(True)
            self.video_input_line_edit.setEnabled(True)

    def on_model_combobox_changed(self, value):
        self.thread.set_model_name(value)
        self.start_button.setText("Start")

    def update_widgets_on_video_play_state(self, is_playing):
        self.start_button.setEnabled(not is_playing)
        self.pause_button.setEnabled(is_playing)
        self.video_input_browse_button.setEnabled(not is_playing)
        self.model_combo_box.setEnabled(not is_playing)
        self.video_input_line_edit.setEnabled(not is_playing)

    def start_video(self):
        self.thread.initialize_detector()
        if not self.thread.is_running:
            self.thread.start()
        else:
            self.thread.is_paused = False

        if self.video_input_line_edit.text() != "":
            self.update_widgets_on_video_play_state(True)
            self.video_mode_live_button.setEnabled(False)
        else:
            self.video_mode_recorded_button.setEnabled(False)
            self.model_combo_box.setEnabled(False)

    def pause_video(self):
        self.thread.pause()
        self.start_button.setText("Continue")
        self.update_widgets_on_video_play_state(False)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.frame_label.setPixmap(qt_img)

    @pyqtSlot()
    def on_video_finished(self):
        self.start_button.setText("Start")
        self.update_widgets_on_video_play_state(False)
        self.start_button.setEnabled(False)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_width, self.video_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    q_app = QApplication(sys.argv)
    window_application = WindowApplication()
    window_application.show()
    sys.exit(q_app.exec_())
