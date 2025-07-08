from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import sys

class RenderWindow(QWidget):
    def __init__(self, parent=None, title="Render Window"):
        super(RenderWindow, self).__init__(parent)
        self.setWindowTitle(title)

        # Create a vertical layout
        layout = QVBoxLayout(self)
        
        # Create a matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Add the canvas to the layout
        layout.addWidget(self.canvas)

        # Add a navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.setLayout(layout)

    def update_plot(self, fig):
        """Update the plot with new figure."""
        self.figure = fig