from magicgui.widgets import create_widget
from napari.layers import Image
import numpy as np
from skimage import color, util
from typing import Optional

from napari_segment_everything.widgets import LabeledSpinner, LabeledMinMaxSlider
from napari_segment_everything.sam_helper import get_sam_automatic_mask_generator, make_label_image_3d, add_properties_to_label_image, filter_labels_3d_multi
import pickle

# qypt improts
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
    QComboBox,
    QLabel,
    QHBoxLayout,
    QFileDialog,
    QStackedWidget,
    QGroupBox,
    QSizePolicy,
    QMessageBox
)

class NapariSegmentEverything(QWidget):

    def __init__(self, napari_viewer, parent=None):

        QWidget.__init__(self, parent=parent)
        self.viewer = napari_viewer

        self.results = None
        self.block_stats = False
        
        self.initUI()

        self._3D_labels_layer = self.viewer.add_labels(
            data=np.zeros((256, 256, 256), dtype=int),
            name="SAM 3D labels",
        )

        self._3D_labels_layer.mouse_double_click_callbacks.append(self.test_double_click)
        self.load_image(self.im_layer_widget.value)

        self.msg = QMessageBox()

    def test_double_click(self, layer, event):
        index = int(layer._value)-1  

        result = self.results[index]
        stats = [
            f"Area: {result['area']}",
            f"Circularity: {result['circularity']:.2f}",
            f"Solidity: {result['solidity']:.2f}",
            f"Mean intensity: {result['mean_intensity']:.2f}",
            f"10th percentile intensity: {result['10th_percentile_intensity']:.2f}",
            f"Mean hue: {result['mean_hue']:.2f}",
            f"Mean saturation: {result['mean_saturation']:.2f}",
            f"Predicted IOU: {result['predicted_iou']:.2f}",
            f"Stability score: {result['stability_score']:.2f}",
        ]
        stats_text = "\n".join(stats)

        self.msg.setIcon(QMessageBox.Information)
        self.msg.setText("Result Stats")
        self.msg.setInformativeText(stats_text)
        self.msg.setWindowTitle("Result Stats")
        self.msg.setModal(True)
        self.msg.show()

    def initUI(self):
        
        self.setWindowTitle('Segment Everything')
        layout = QVBoxLayout()
        layout.setSpacing(30)

        ##### 1.  SAM Parameters ####

        self.sam_parameters_group = QGroupBox("1. Generate 3D labels")
        self.sam_layout = QVBoxLayout()
        self.sam_parameters_group.setLayout(self.sam_layout)

        # add open results button
        self.open_project_button = QPushButton("Open previous...")
        self.open_project_button.clicked.connect(self.open_project)
        self.sam_layout.addWidget(self.open_project_button)

        # Dropdown for selecting the model
        model_label = QLabel("Select Model")# Dropdown for selecting the model
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["vit_b"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        self.sam_layout.addLayout(model_layout)

        # control for selecting the image
        image_layout = QHBoxLayout()
        image_label = QLabel("Select Image")
        self.im_layer_widget = create_widget(annotation=Image, label="Image:")
        self.im_layer_widget.changed.connect(self.load_image)
        image_layout.addWidget(image_label)
        image_layout.addWidget(self.im_layer_widget.native)
        self.sam_layout.addLayout(image_layout)
        
        self.im_layer_widget.reset_choices()
        self.viewer.layers.events.inserted.connect(self.im_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.im_layer_widget.reset_choices)

        self.points_per_side_spinner = LabeledSpinner("Points per side", 4, 100, 32, None)
        self.pred_iou_thresh_spinner = LabeledSpinner("Pred IOU Threshold", 0, 1, 0.1, None, is_double=True)
        self.stability_score_thresh_spinner = LabeledSpinner("Stability Score Threshold", 0, 1, 0.1, None, is_double=True)
        self.box_nms_thresh_spinner = LabeledSpinner("Box NMS Threshold", 0, 1, 0.7, None, is_double=True)
        self.crop_n_layers_spinner = LabeledSpinner("Crop N Layers", 0, 10, 0, None)

        self.sam_layout.addWidget(self.points_per_side_spinner)
        self.sam_layout.addWidget(self.pred_iou_thresh_spinner)
        self.sam_layout.addWidget(self.stability_score_thresh_spinner)
        self.sam_layout.addWidget(self.box_nms_thresh_spinner)
        self.sam_layout.addWidget(self.crop_n_layers_spinner)

        # add process button
        self.process_button = QPushButton("Generate 3D labels")
        self.process_button.clicked.connect(self.process)
        self.sam_layout.addWidget(self.process_button)

        layout.addWidget(self.sam_parameters_group)

        ##### 2.  Filter Results #####

        self.filter_results_group = QGroupBox("2. Filter 3D Labels")
        self.filter_layout = QVBoxLayout()
        self.filter_results_group.setLayout(self.filter_layout)

        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.filter_results_group.setSizePolicy(sizePolicy)

        self.min_max_area_slider = LabeledMinMaxSlider("Area", 0, 1000000, 0, 1000000, 1000, 'area', self.change_stat)
        self.min_max_label_num_slider = LabeledMinMaxSlider("Label Num", 0, 500, 0, 500, 1, 'label_num', self.change_stat)
        self.min_max_solidity_slider = LabeledMinMaxSlider("Solidity", 0, 1, 0, 1, .01, 'solidity', self.change_stat, True)
        self.min_max_circularity_slider = LabeledMinMaxSlider("Circularity", 0, 1, 0, 1, .01, 'circularity', self.change_stat, True)
        self.min_max_mean_intensity_slider = LabeledMinMaxSlider("Mean Intensity", 0, 255, 0, 255, 10, 'mean_intensity', self.change_stat)
        self.min_max_p10_intensity_slider = LabeledMinMaxSlider("10th Percentile Intensity", 0, 1000, 0, 1000, 10, '10th_percentile_intensity', self.change_stat)
        self.min_max_hue_slider = LabeledMinMaxSlider("Hue", 0, 255, 0, 255, 10, 'hue', self.change_stat)
        self.min_max_saturation_slider = LabeledMinMaxSlider("Saturation", 0, 255, 0, 255, 10, 'saturation', self.change_stat)
        self.min_max_iou_slider = LabeledMinMaxSlider("IOU", 0, 1, 0, 1, .01, 'iou', self.change_stat, True)
        self.min_max_stability_score_slider = LabeledMinMaxSlider("Stability", 0, 1, 0, 1, .01, 'stability_score', self.change_stat, True)

        self.sliders = [self.min_max_area_slider, self.min_max_label_num_slider, self.min_max_solidity_slider, self.min_max_circularity_slider, self.min_max_mean_intensity_slider, self.min_max_p10_intensity_slider, self.min_max_hue_slider, self.min_max_saturation_slider, self.min_max_iou_slider, self.min_max_stability_score_slider]

        self.combo_box = QComboBox()
        self.combo_box.addItems(["Area", "Label Num", "Solidity", "Circularity", "Mean Intensity", "10th Percentile Intensity", "Hue", "Saturation", "IOU", "Stability Score"])
        self.combo_box.currentIndexChanged.connect(self.change_slider)

        self.stacked_widget = QStackedWidget()
        for slider in self.sliders:
            self.stacked_widget.addWidget(slider)

        select_slider_layout = QHBoxLayout()
        select_slider_layout.addWidget(QLabel("Select Stat:"))
        select_slider_layout.addWidget(self.combo_box)
        self.filter_layout.addLayout(select_slider_layout)
        self.stacked_widget.setFixedHeight(100)
        self.filter_results_group.setFixedHeight(160)
        self.filter_layout.addWidget(self.stacked_widget)        
        
        layout.addWidget(self.filter_results_group)

        self.make_2d_labels_groups = QGroupBox("3. Generate 2D Labels")
        self.make_2d_labels_layout = QVBoxLayout()
        self.make_2d_labels_groups.setLayout(self.make_2d_labels_layout)
        
        # add 2d labels options
        self.labels_options_layout = QHBoxLayout()
        self.labels_options_label = QLabel("2D Labels Options")
        self.labels_options_combo = QComboBox()
        self.labels_options_combo.addItems(["Small in front", "Big in front"])
        self.labels_options_layout.addWidget(self.labels_options_label)
        self.labels_options_layout.addWidget(self.labels_options_combo)
        
        # add make 2d labels button
        self.make_2d_labels_button = QPushButton("Make 2D Labels")
        self.make_2d_labels_button.clicked.connect(self.make_2d_labels)

        self.make_2d_labels_layout.addLayout(self.labels_options_layout)
        self.make_2d_labels_layout.addWidget(self.make_2d_labels_button)

        layout.addWidget(self.make_2d_labels_groups)

        # add save results button
        self.save_project_button = QPushButton("Save SAM project")
        self.save_project_button.clicked.connect(self.save_project)
        layout.addWidget(self.save_project_button)
        
        self.setLayout(layout)

    def change_slider(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def open_project(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Pickle Files (*.pkl)", options=options)
        if file_name:
            with open(file_name, 'rb') as f:
                project = pickle.load(f)
                self.results = project['results']
                image = project['image']
                self.viewer.add_image(image)
                self.image = image
        
        self._3D_labels_layer.data = make_label_image_3d(self.results)
        self.viewer.dims.ndisplay = 3
        self._3D_labels_layer.translate = (-len(self.results), 0, 0)

        add_properties_to_label_image(self.image, self.results)
        self.update_slider_min_max() 

    def save_project(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "Pickle Files (*.pkl)", options=options)
        if file_name:
            project = {'results': self.results, 'image': self.image}
            with open(file_name, 'wb') as f:
                pickle.dump(project, f)

    def process(self):
        if self.image is None:
            return
        
        points_per_side = self.points_per_side_spinner.spinner.value()
        pred_iou_thresh = self.pred_iou_thresh_spinner.spinner.value()
        stability_score_thresh = self.stability_score_thresh_spinner.spinner.value()
        box_nms_thresh = self.box_nms_thresh_spinner.spinner.value()
        crop_n_layers = self.crop_n_layers_spinner.spinner.value()
        
        self._predictor = get_sam_automatic_mask_generator("vit_b", points_per_side=points_per_side, 
                                                           pred_iou_thresh=pred_iou_thresh, 
                                                           stability_score_thresh=stability_score_thresh, 
                                                           box_nms_thresh=box_nms_thresh,
                                                           crop_n_layers=crop_n_layers
                                                           )
        
        self.results = self._predictor.generate(self.image)
        
        self.results = sorted(self.results, key=lambda x: x['area'], reverse=False)

        print(len(self.results), 'objects found')
        
        label_num = 1
        for result in self.results:
            result['keep'] = True
            result['label_num'] = label_num
            label_num += 1

        add_properties_to_label_image(self.image, self.results)

        label_image = make_label_image_3d(self.results)

        print(label_image.shape)

        self.update_slider_min_max() 
        self._3D_labels_layer.data = label_image
        self.viewer.dims.ndisplay = 3
        self._3D_labels_layer.translate = (-len(self.results), 0, 0)
    def update_slider_min_max(self):

        stats = ['area', 'label_num', 'solidity', 'circularity', 'mean_intensity', '10th_percentile_intensity', 'mean_hue', 'mean_saturation', 'predicted_iou', 'stability_score']
        for stat in stats:
            min_ = min([result[stat] for result in self.results])
            max_ = max([result[stat] for result in self.results])

            print(stat, min_, max_)

        num_labels = len(self.results)

        self.block_stats = True
        self.min_max_label_num_slider.min_spinbox.setRange(0, num_labels)
        self.min_max_label_num_slider.max_spinbox.setRange(0, num_labels)
        self.min_max_label_num_slider.min_slider.setRange(0, num_labels)
        self.min_max_label_num_slider.max_slider.setRange(0, num_labels)
        #self.min_max_label_num_slider.min_slider.setValue(0)
        self.min_max_label_num_slider.max_slider.setValue(num_labels)
        self.block_stats = False
        
    def change_stat(self, stat, min_value, max_value):
        if (self.block_stats == True):
            return
        
        if (self.viewer.dims.ndisplay != 3):
            return
        
        if (self.results is None):
            return
        
        # max of predicted_iou stat
        min_iou = min([result['predicted_iou'] for result in self.results])
        max_iou = max([result['predicted_iou'] for result in self.results])

        min_stability_score = min([result['stability_score'] for result in self.results])
        max_stability_score = max([result['stability_score'] for result in self.results])

        min_iou = self.min_max_iou_slider.min_spinbox.value()
        max_iou = self.min_max_iou_slider.max_spinbox.value()

        min_stability_score = self.min_max_stability_score_slider.min_spinbox.value()
        max_stability_score = self.min_max_stability_score_slider.max_spinbox.value()

        min_circularity = self.min_max_circularity_slider.min_spinbox.value()
        max_circularity = self.min_max_circularity_slider.max_spinbox.value()

        # TODO: change below to a dictionary to make it easier to add new stats

        stats = ['area', 'label_num', 'solidity', 'circularity', 'mean_intensity', '10th_percentile_intensity', 'mean_hue', 'mean_saturation', 'predicted_iou', 'stability_score']
        mins = [self.min_max_area_slider.min_spinbox.value(), self.min_max_label_num_slider.min_spinbox.value(), 
                self.min_max_solidity_slider.min_spinbox.value(), min_circularity, self.min_max_mean_intensity_slider.min_spinbox.value(), 
                self.min_max_p10_intensity_slider.min_spinbox.value(), self.min_max_hue_slider.min_spinbox.value(), self.min_max_saturation_slider.min_spinbox.value(),
                min_iou, min_stability_score]
        maxs = [self.min_max_area_slider.max_spinbox.value(), self.min_max_label_num_slider.max_spinbox.value(), 
                self.min_max_solidity_slider.max_spinbox.value(), max_circularity, self.min_max_mean_intensity_slider.max_spinbox.value(), 
                self.min_max_p10_intensity_slider.max_spinbox.value(), self.min_max_hue_slider.max_spinbox.value(), self.min_max_saturation_slider.max_spinbox.value(),
                max_iou, max_stability_score]
        if self.results is not None:
            filter_labels_3d_multi(self._3D_labels_layer.data, self.results, stats, mins, maxs, napari_label=self._3D_labels_layer)
     
    def refresh_slider(self):
        area_min, area_max = 0, np.inf
        label_num_min, label_num_max = 0, np.inf
        solidity_min, solidity_max = 0, 1
        mean_intensity_min, mean_intensity_max = 0, np.inf
        p10_intensity_min, p10_intensity_max = 0, np.inf

        for result in self.results:
            area_min = min(area_min, result['area'])
            area_max = max(area_max, result['area'])
            label_num_min = min(label_num_min, result['label_num'])
            label_num_max = max(label_num_max, result['label_num'])
            solidity_min = min(solidity_min, result['solidity'])
            solidity_max = max(solidity_max, result['solidity'])
            mean_intensity_min = min(mean_intensity_min, result['mean_intensity'])
            mean_intensity_max = max(mean_intensity_max, result['mean_intensity'])
            p10_intensity_min = min(p10_intensity_min, result['10th_percentile_intensity'])
            p10_intensity_max = max(p10_intensity_max, result['10th_percentile_intensity'])

    def make_2d_labels(self):
        min_max = self.labels_options_combo.currentText()
        
        label_image = self._3D_labels_layer.data
        
        if min_max == "Small in front":
             # Create a masked array where zeros are masked
            masked_label_image = np.ma.masked_equal(label_image, 0)
            # Perform the min projection on the masked array
            labels = np.ma.min(masked_label_image, axis=0).filled(0)
        else:
            labels = np.max(label_image, axis=0)
        self.viewer.add_labels(labels, name="2D labels")
     
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

        self.image = util.img_as_ubyte(image)

        #self._mask_layer.data = np.zeros(self.image.shape[:2], dtype=int)
        #self._labels_layer.data = np.zeros(self.image.shape[:2], dtype=int)

        max_area = image.shape[0] * image.shape[1]

        self.min_max_area_slider.min_spinbox.setRange(0, max_area)
        self.min_max_area_slider.max_spinbox.setRange(0, max_area)
        self.min_max_area_slider.min_slider.setRange(0, max_area)
        self.min_max_area_slider.max_slider.setRange(0, max_area)

        for i in range(len(self.viewer.layers)):
            if self.viewer.layers[i].name == "SAM 3D labels":
                self.viewer.layers.move(i, -1)
                break

        

