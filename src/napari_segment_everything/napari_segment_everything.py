from magicgui.widgets import create_widget
from napari.layers import Image
import numpy as np
from skimage import color, util
from typing import Optional
from napari_segment_everything.widgets import LabeledSpinner, LabeledMinMaxSlider

# qypt improts
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
    QComboBox,
    QLabel,
    QHBoxLayout,
)


class NapariSegmentEverything(QWidget):

    def __init__(self, napari_viewer, parent=None):
        QWidget.__init__(self, parent=parent)
        self.viewer = napari_viewer

        self.initUI()

    def initUI(self):
        
        self.setWindowTitle('Segment Everything')
        layout = QVBoxLayout()
        layout.setSpacing(2)

        # add open results button
        self.open_project_button = QPushButton("Open SAM project")
        self.open_project_button.clicked.connect(self.open_project)
        layout.addWidget(self.open_project_button)

        # Dropdown for selecting the model
        model_label = QLabel("Select Model")# Dropdown for selecting the model
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["vit_b"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)

        self.im_layer_widget = create_widget(annotation=Image, label="Image:")
        self.im_layer_widget.changed.connect(self.load_image)
        layout.addWidget(self.im_layer_widget.native)
        self.im_layer_widget.reset_choices()
        self.viewer.layers.events.inserted.connect(self.im_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.im_layer_widget.reset_choices)

        self.points_per_side_spinner = LabeledSpinner("Points per side", 4, 100, 32, None)
        self.pred_iou_thresh_spinner = LabeledSpinner("Pred IOU Threshold", 0, 1, 0.1, None, is_double=True)
        self.stability_score_thresh_spinner = LabeledSpinner("Stability Score Threshold", 0, 1, 0.1, None, is_double=True)
        self.box_nms_thresh_spinner = LabeledSpinner("Box NMS Threshold", 0, 1, 0.5, None, is_double=True)

        layout.addWidget(self.points_per_side_spinner)
        layout.addWidget(self.pred_iou_thresh_spinner)
        layout.addWidget(self.stability_score_thresh_spinner)
        layout.addWidget(self.box_nms_thresh_spinner)

        # add process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process)
        layout.addWidget(self.process_button)

        self.min_max_area_slider = LabeledMinMaxSlider("Min Area", "Max Area", 0, 1000000, 0, 1000000, 1000, 'area', self.change_stat)
        self.min_max_label_num_slider = LabeledMinMaxSlider("Min Label Num", "Max Label Num", 0, 1000, 0, 1000, 10, 'label_num', self.change_stat)
        self.min_max_solidity_slider = LabeledMinMaxSlider("Min Solidity", "Max Solidity", 0, 100, 0, 100, 1, 'solidity', self.change_stat)
        self.min_max_mean_intensity_slider = LabeledMinMaxSlider("Min Mean Intensity", "Max Mean Intensity", 0, 255, 0, 255, 10, 'mean_intensity', self.change_stat)
        self.min_max_p10_intensity_slider = LabeledMinMaxSlider("Min 10th Percentile Intensity", "Max 10th Percentile Intensity", 0, 1000, 0, 1000, 10, '10th_percentile_intensity', self.change_stat)
        self.min_max_hue_slider = LabeledMinMaxSlider("Min Hue", "Max Hue", 0, 255, 0, 255, 10, 'hue', self.change_stat)
        self.min_max_saturation_slider = LabeledMinMaxSlider("Min Saturation", "Max Saturation", 0, 255, 0, 255, 10, 'saturation', self.change_stat)

        layout.addWidget(self.min_max_area_slider)
        layout.addWidget(self.min_max_label_num_slider)
        layout.addWidget(self.min_max_solidity_slider)
        layout.addWidget(self.min_max_mean_intensity_slider)
        layout.addWidget(self.min_max_p10_intensity_slider)
        layout.addWidget(self.min_max_hue_slider)
        layout.addWidget(self.min_max_saturation_slider)

        # add save results button
        self.save_project_button = QPushButton("Save SAM project")
        self.save_project_button.clicked.connect(self.save_project)
        layout.addWidget(self.save_project_button)

        # add make 2d labels button
        self.make_2d_labels_button = QPushButton("Make 2D Labels")
        self.make_2d_labels_button.clicked.connect(self.make_2d_labels)
        make_2d_label_layout = QHBoxLayout()
        make_2d_label_layout.addWidget(self.make_2d_labels_button)

        # add 2d labels options
        self.labels_options_combo = QComboBox()
        self.labels_options_combo.addItems(["Big in front", "Small in front"])
        make_2d_label_layout.addWidget(self.labels_options_combo)
        
        layout.addLayout(make_2d_label_layout)
        self.setLayout(layout)

    def open_project(self):
        pass

    def process(self):
        pass

    def change_stat(self, stat, min_value, max_value):
        pass

    def save_project(self):
        pass

    def make_2d_labels(self):
        pass
     
    def load_image(self, im_layer: Optional[Image]) -> None:
        
        if im_layer is None:
            return

        if im_layer.ndim != 2:
            raise ValueError(
                f"Only 2D images supported. Got {im_layer.ndim}-dim image."
            )

        image = im_layer.data
        if not im_layer.rgb:
            image = color.gray2rgb(image)

        elif image.shape[-1] == 4:
            # images with alpha
            image = color.rgba2rgb(image)

        if np.issubdtype(image.dtype, np.floating):
            image = image - image.min()
            image = image / image.max()

        self._image = util.img_as_ubyte(image)

        #self._mask_layer.data = np.zeros(self._image.shape[:2], dtype=int)
        #self._labels_layer.data = np.zeros(self._image.shape[:2], dtype=int)

        max_area = image.shape[0] * image.shape[1]

        self.min_max_area_slider.min_spinbox.setRange(0, max_area)
        self.min_max_area_slider.max_spinbox.setRange(0, max_area)
        self.min_max_area_slider.min_slider.setRange(0, max_area)
        self.min_max_area_slider.max_slider.setRange(0, max_area)

        self.min_max_label_num_slider.min_spinbox.setRange(0, 1000)
        self.min_max_label_num_slider.max_spinbox.setRange(0, 1000)
        self.min_max_label_num_slider.min_slider.setRange(0, 1000)
        self.min_max_label_num_slider.max_slider.setRange(0, 1000)
