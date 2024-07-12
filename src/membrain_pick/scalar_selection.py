from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox
import numpy as np

class ScalarSelectionWidget(QWidget):
    def __init__(self, surface_layer, normal_values):
        super().__init__()
        self.surface_layer = surface_layer
        self.normal_values = normal_values
        self.len_features = normal_values.shape[1]

        self.layout = QVBoxLayout(self)


        self.channel_label = QLabel("Select Feature Channel:")
        self.layout.addWidget(self.channel_label)

        self.channel_selector = QSpinBox()
        self.channel_selector.setRange(0, self.len_features - 1)
        self.layout.addWidget(self.channel_selector)

        self.channel_selector.valueChanged.connect(self.update_coloring)

    def update_coloring(self):
        selected_channel = self.channel_selector.value()
        
        scalars = self.normal_values[:, self.len_features - selected_channel - 1] # reverse order to go from inside to outside
        colors = self.normalize_colors(scalars)

        if colors is not None:
            self.surface_layer.data = (
                self.surface_layer.data[0],
                self.surface_layer.data[1],
                colors,
            )

    def normalize_colors(self, scalars):
        normalized_values = (scalars - scalars.min()) / (
            scalars.max() - scalars.min() + np.finfo(float).eps
        )
        return normalized_values