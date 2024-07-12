from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox
from matplotlib.pyplot import get_cmap
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
        
        scalars = self.normal_values[:, selected_channel]  # Select the feature channel
        colors = self.get_colored_mesh(scalars)

        if colors is not None:
            self.surface_layer.data[2] = colors

    def get_colored_mesh(self, scalars):

        normalized_values = (scalars - scalars.min()) / (
            scalars.max() - scalars.min() + np.finfo(float).eps
        )
        # normalized_scalars = scalars / 10.0
        # normalized_scalars[normalized_scalars < 0] = 0
        # normalized_scalars[normalized_scalars > 1] = 1
        # normalized_scalars = 1 - normalized_scalars
        # colors = np.stack([normalized_scalars] * 3, axis=-1)
        # cmap = get_cmap('RdBu')
        # colors = cmap(normalized_scalars)[:, :3]  # Get RGB values and discard the alpha channel
        return normalized_values