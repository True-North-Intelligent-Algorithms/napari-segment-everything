# qypt imports
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QWidget,
    QSlider,
)

from qtpy.QtCore import Qt

class LabeledMinMaxSlider(QWidget):
    def __init__(self, text, min_value, max_value, initial_min_value, initial_max_value, tick_interval, stat, change_value_method, scale_slider = False):
        super().__init__()

        self.scale_slider = scale_slider

        self.label = QLabel(text)
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setRange(min_value, max_value)
        self.min_spinbox.setValue(initial_min_value)
        self.min_spinbox.valueChanged.connect(self.change_value)
        self.min_spinbox.setKeyboardTracking(False)

        self.min_slider = QSlider(Qt.Horizontal)
        if scale_slider == False:
            self.min_slider.setMinimum(min_value)
            self.min_slider.setMaximum(max_value)
            self.min_slider.setValue(initial_min_value)
            self.min_slider.setTickPosition(QSlider.TicksBelow)
            self.min_slider.setTickInterval(tick_interval)
        else:
            self.min_slider.setMinimum(0)
            self.min_slider.setMaximum(100)
            self.min_slider.setValue(100 * int((initial_min_value - min_value) / (max_value - min_value)))
            self.min_slider.setTickPosition(QSlider.TicksBelow)
            self.min_slider.setTickInterval(1)
        self.min_slider.valueChanged.connect(self.change_value)

        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setRange(min_value, max_value)
        self.max_spinbox.setValue(initial_max_value)
        self.max_spinbox.valueChanged.connect(self.change_value)
        self.max_spinbox.setKeyboardTracking(False)

        self.max_slider = QSlider(Qt.Horizontal)
        
        if scale_slider == False:
            self.max_slider.setMinimum(min_value)
            self.max_slider.setMaximum(max_value)
            self.max_slider.setValue(initial_max_value)
            self.max_slider.setTickPosition(QSlider.TicksBelow)
            self.max_slider.setTickInterval(tick_interval)
        else:
            self.max_slider.setMinimum(0)
            self.max_slider.setMaximum(100)
            self.max_slider.setValue(100 * int((initial_max_value - min_value) / (max_value - min_value)))
            self.max_slider.setTickPosition(QSlider.TicksBelow)
            self.max_slider.setTickInterval(1)
        
        self.max_slider.valueChanged.connect(self.change_value)

        self.min_layout = QHBoxLayout()
        self.min_layout.setContentsMargins(5, 1, 5, 1)  # adjust the margins around the layout
        self.min_layout.addWidget(self.min_slider)
        self.min_layout.addWidget(self.min_spinbox)

        self.max_layout = QHBoxLayout()
        self.max_layout.setContentsMargins(5, 1, 5, 1)  # adjust the margins around the layout
        self.max_layout.addWidget(self.max_slider)
        self.max_layout.addWidget(self.max_spinbox)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 1, 5, 1)  # adjust the margins around the layout
        # set spacing
        self.layout.setSpacing(0)
        
        #self.layout.addWidget(self.label)
        self.layout.addLayout(self.min_layout)
        self.layout.addLayout(self.max_layout)
        self.setLayout(self.layout)

        self.stat = stat
        self.change_value_method = change_value_method 

    def change_value(self):
        sender = self.sender()
        if sender in [self.min_slider, self.max_slider]:
            # Block signals from the spinboxes while changing their values
            self.min_spinbox.blockSignals(True)
            self.max_spinbox.blockSignals(True)
            if self.scale_slider == False:
                self.min_spinbox.setValue(self.min_slider.value())
                self.max_spinbox.setValue(self.max_slider.value())
            else:
                self.min_spinbox.setValue(self.min_slider.value() * (self.max_spinbox.maximum() - self.max_spinbox.minimum()) / 100 + self.max_spinbox.minimum())
                self.max_spinbox.setValue(self.max_slider.value() * (self.max_spinbox.maximum() - self.max_spinbox.minimum()) / 100 + self.max_spinbox.minimum())
            self.min_spinbox.blockSignals(False)
            self.max_spinbox.blockSignals(False)
        elif sender in [self.min_spinbox, self.max_spinbox]:
            # Block signals from the sliders while changing their values
            self.min_slider.blockSignals(True)
            self.max_slider.blockSignals(True)
            if self.scale_slider == False:
                self.min_slider.setValue(int(self.min_spinbox.value()))
                self.max_slider.setValue(int(self.max_spinbox.value()))
            else:
                self.min_slider.setValue(int(100 * (self.min_spinbox.value() - self.max_spinbox.minimum()) / (self.max_spinbox.maximum() - self.max_spinbox.minimum())))
                self.max_slider.setValue(int(100 * (self.max_spinbox.value() - self.max_spinbox.minimum()) / (self.max_spinbox.maximum() - self.max_spinbox.minimum())))
            self.min_slider.blockSignals(False)
            self.max_slider.blockSignals(False)

        self.change_value_method(self.stat, self.min_spinbox.value(), self.max_spinbox.value())

    def update_min_max(self, min_value, max_value):
        self.min_spinbox.setRange(min_value, max_value)
        self.max_spinbox.setRange(min_value, max_value)
        self.min_slider.setRange(min_value, max_value)
        self.max_slider.setRange(min_value, max_value)

class LabeledMinMaxSpinners(QWidget):

    def __init__(self, min_label_text, max_label_text, min_value, max_value, initial_min_value, initial_max_value, change_value_method):
        super().__init__()

        self.min_label = QLabel(min_label_text)
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setRange(min_value, max_value)
        self.min_spinbox.setValue(initial_min_value)

        if change_value_method is not None:
            self.min_spinbox.valueChanged.connect(change_value_method)

        self.max_label = QLabel(max_label_text)
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setRange(min_value, max_value)
        self.max_spinbox.setValue(initial_max_value)
        
        if change_value_method is not None:
            self.max_spinbox.valueChanged.connect(change_value_method)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 2, 10, 2)  # adjust the margins around the layout
        self.layout.addWidget(self.min_label)
        self.layout.addWidget(self.min_spinbox)
        self.layout.addWidget(self.max_label)
        self.layout.addWidget(self.max_spinbox)

        self.setLayout(self.layout)

class LabeledSpinner(QWidget):
    def __init__(self, label_text, min_value, max_value, default_value, change_value_method, is_double=False):
        super().__init__()

        self.label = QLabel(label_text)
        self.spinner = QDoubleSpinBox() if is_double else QSpinBox()
        self.spinner.setRange(min_value, max_value)
        self.spinner.setValue(default_value)

        if change_value_method is not None:
            self.spinner.valueChanged.connect(change_value_method)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 2, 10, 2)  # adjust the margins around the layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spinner)

        self.setLayout(self.layout)
