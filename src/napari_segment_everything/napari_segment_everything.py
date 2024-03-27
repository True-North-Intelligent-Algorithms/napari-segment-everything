# qypt improts
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
)

class NapariSegmentEverything(QWidget):

    def __init__(self, napari_viewer, parent=None):
        QWidget.__init__(self, parent=parent)
        #super().__init__()

        self.viewer = napari_viewer

        self.initUI()

    def initUI(self):
        
        self.setWindowTitle('Segment Everything')

        self.layout = QVBoxLayout()

        self.segment_button = QPushButton('Segment')
        #self.segment_button.clicked.connect(self.segment)

        self.layout.addWidget(self.segment_button)

        self.setLayout(self.layout)