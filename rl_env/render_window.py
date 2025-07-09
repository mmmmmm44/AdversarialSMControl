from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
import numpy as np


import socket
import threading
from collections import deque
from datetime import datetime

import sys
sys.path.append(".")  # Adjust path to import utils from parent directory
from utils import print_log

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

        # Data buffers (deques)
        self.datetimes = deque(maxlen=20000)
        self.user_load = deque(maxlen=20000)
        self.grid_load = deque(maxlen=20000)
        self.battery_soc = deque(maxlen=20000)

        # Start a timer for real-time plotting
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(200)  # 5 Hz

    def push_data(self, data_dict):
        """Push new data from the environment into the deques. Handles both single values and lists."""
        # Debug print
        print(f"[RenderWindow] push_data received: {data_dict}")

        def _append_or_extend(deq, val):
            if isinstance(val, list):
                deq.extend(val)
            elif val is not None:
                deq.append(val)

        # convert the timestamp to a datetime object if it's a string
        if isinstance(data_dict.get('timestamp'), list):
            new_timestamps = []
            for ts in data_dict['timestamp']:
                try:
                    new_timestamps.append(datetime.fromisoformat(ts))
                except ValueError:
                    print_log(f"[RenderWindow] Invalid timestamp format: {ts}")
            data_dict['timestamp'] = new_timestamps
        elif isinstance(data_dict.get('timestamp'), str):
            try:
                timestamp = datetime.fromisoformat(data_dict['timestamp'])
            except ValueError:
                print_log(f"[RenderWindow] Invalid timestamp format: {data_dict['timestamp']}")
                timestamp = None
            data_dict['timestamp'] = timestamp


        _append_or_extend(self.datetimes, data_dict.get('timestamp'))
        _append_or_extend(self.user_load, data_dict.get('user_load'))
        _append_or_extend(self.grid_load, data_dict.get('grid_load'))
        _append_or_extend(self.battery_soc, data_dict.get('battery_soc'))

    def update_plot(self):
        """Plot the latest data in the deques."""
        
        self.figure.clf()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)
        # Convert to numpy arrays for plotting
        try:
            t = np.array(self.datetimes)
            user = np.array(self.user_load, dtype=np.float32)
            grid = np.array(self.grid_load, dtype=np.float32)
            soc = np.array(self.battery_soc, dtype=np.float32)
            # Plot user and grid load
            ax1.plot(t, grid, label="Grid Load")
            ax1.plot(t, user, label="User Load")
            ax1.set_title("Grid Load and User Load")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Load (W)")
            ax1.set_ylim(0, 5000)  # Adjust y-axis limit as needed
            ax1.legend()
            # Plot battery SOC
            ax2.plot(t, soc, label="Battery SOC")
            ax2.set_title("Battery State of Charge")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("State of Charge (%)")
            ax2.legend()
            ax2.set_ylim(0, 1.1)
            self.figure.tight_layout()
        except Exception as e:
            print(f"[RenderWindow] Plotting error: {e}")
        self.canvas.draw()


# TCP server to receive matplotlib figures and update the GUI
class RenderServer:
    def __init__(self, host='127.0.0.1', port=50007, window=None):
        self.host = host
        self.port = port
        self.window = window
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def listen(self):
        import json
        import struct
        print(f"[RenderServer] Listening on {self.host}:{self.port}")
        while True:
            conn, addr = self.server_socket.accept()
            print_log(f"[RenderServer] Connection from {addr}")
            try:
                while True:
                    # Read 4 bytes for the message length
                    msg_len_bytes = b''
                    while len(msg_len_bytes) < 4:
                        chunk = conn.recv(4 - len(msg_len_bytes))
                        if not chunk:
                            break
                        msg_len_bytes += chunk
                    if not msg_len_bytes or len(msg_len_bytes) < 4:
                        break
                    msg_len = struct.unpack('>I', msg_len_bytes)[0]
                    # Now read the actual message
                    data = b''
                    while len(data) < msg_len:
                        packet = conn.recv(msg_len - len(data))
                        if not packet:
                            break
                        data += packet
                    if not data:
                        break
                    # decode and parse JSON
                    data_dict = json.loads(data.decode('utf-8'))
                    print_log(f"[RenderServer] Received data: {data_dict}")
                    if self.window:
                        self.window.push_data(data_dict)
                        print_log(f"[RenderServer] Data pushed to window.")
            except Exception as e:
                print_log(f"[RenderServer] Error parsing data: {e}")
            conn.close()


def main():
    app = QApplication(sys.argv)
    window = RenderWindow()
    window.show()
    # Start the TCP server
    server = RenderServer(window=window)
    sys.exit(app.exec())
    # close the server socket when the application exits
    server.server_socket.close()

if __name__ == "__main__":
    main()
