import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PyQt5.QtMultimediaWidgets import QVideoWidget
from test_gui import Ui_MainWindow  # Import the generated UI class

class VideoPlayerApp(QMainWindow):
    def __init__(self, parent=None):
        super(VideoPlayerApp, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Video related variables
        self.video_path = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Create a QLabel to display video frames
        self.video_label = QLabel(self)
        self.video_label.setGeometry(20, 10, 1280, 720)

        # Button connections
        self.ui.pushButton.clicked.connect(self.play_video)
        self.ui.pushButton_2.clicked.connect(self.pause_video)
        self.ui.pushButton_3.clicked.connect(self.stop_video)
        self.ui.pushButton_4.clicked.connect(self.browse_video)


    def play_video(self):
        if not self.timer.isActive():
            self.timer.start(30)  # Update the frame every 30 milliseconds

    def pause_video(self):
        if self.timer.isActive():
            self.timer.stop()

    def stop_video(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind the video to the beginning
            self.pause_video()

    def browse_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.video_path = file_name
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap.scaled(1280, 720, Qt.KeepAspectRatio))
            self.video_label.setAlignment(Qt.AlignCenter)
        else:
            self.stop_video()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoPlayerApp()
    window.setWindowTitle('Video Player')
    window.show()
    sys.exit(app.exec_())
