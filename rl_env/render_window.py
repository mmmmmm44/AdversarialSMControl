from copy import deepcopy
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication, QLabel
from PySide6.QtCore import Qt, QMetaObject, Q_ARG
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
import numpy as np


import socket
import threading
from collections import deque
from datetime import datetime
from enum import Enum
import json
import struct

import sys
sys.path.append(".")  # Adjust path to import utils from parent directory
from utils import print_log

class RenderWindowControl(Enum):
    RESET = 0       # reset the window & buffers
    SAVE_GRAPH = 1  # save the current graph
    RECEIVE_ENV_INFO = 2  # receive environment info from the RL module

    CLOSE = 98      # close the connection. Open for new connection
    TERMINATE = 99  # terminate the server and close the window

class RenderWindowMessageType(Enum):
    DATA = 0       # data message containing the payload
    CONTROL = 1    # control message containing the command and payload

class RenderWindow(QWidget):
    def __init__(self, parent=None, title="Render Window", host='127.0.0.1', port=50007):
        super(RenderWindow, self).__init__(parent)
        self.setWindowTitle(title)

        # Create a vertical layout
        layout = QVBoxLayout(self)
        
        # Create a matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 10), dpi=150)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.suptitle_suffix = "Real-time Load and Battery SOC Monitoring"
        self.suptitle_text = self.suptitle_suffix
        self.env_info = {}      # to be set by the environment after making an initial connection

        # Add the canvas to the layout
        layout.addWidget(self.canvas)

        # Add a navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Add a status label at the bottom
        self.status_label = QLabel(f"Open for connection: {host}:{port}")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

        # Data buffers (deques)
        self.datetimes = deque(maxlen=20000)
        self.user_load = deque(maxlen=20000)
        self.grid_load = deque(maxlen=20000)
        self.battery_soc = deque(maxlen=20000)

        # Start a timer for real-time plotting
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(500)  # 2 Hz

    def set_status(self, text):
        # Thread-safe update of the status label using QMetaObject.invokeMethod
        QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, Q_ARG(str, text))
        print_log(f"[RenderWindow] Status updated: {text}")

    def push_data(self, data_dict):
        """Push new data from the environment into the deques. Handles both single values and lists."""

        def _append_or_extend(deq, val):
            if isinstance(val, list):
                deq.extend(val)
            elif val is not None:
                deq.append(val)

        # convert the timestamp to a datetime object
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

        st = self.figure.suptitle(self.suptitle_text, fontsize=14)

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
            # ax1.set_ylim(0, 8200)  # Adjust y-axis limit as needed
            ax1.legend()
            # Plot battery SOC
            ax2.plot(t, soc, label="Battery SOC")
            ax2.set_title("Battery State of Charge")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Normalized State of Charge")
            ax2.legend()
            ax2.set_ylim(0, 1.1)
            self.figure.tight_layout()

            # shift subplots down:
            st.set_y(0.95)
            self.figure.subplots_adjust(top=0.85)

        except Exception as e:
            print(f"[RenderWindow] Plotting error: {e}")
        self.canvas.draw()

    def reset_buffers(self):
        self.datetimes.clear()
        self.user_load.clear()
        self.grid_load.clear()
        self.battery_soc.clear()
        print_log("[RenderWindow] Buffers reset.")
        self.set_status("Buffers are reset")

    def save_graph(self, kwargs):
        try:
            curr_graph = deepcopy(self.figure)      # deepcopy to keep the current graph, as it will be re-drawn regularly
            curr_graph.savefig(**kwargs)
            print_log(f"[RenderWindow] Graph saved to {kwargs.get('fname')}")
            self.set_status(f"Graph is saved to {kwargs.get('fname')}")

            del curr_graph  # Free memory
        except Exception as e:
            print_log(f"[RenderWindow] Failed to save graph: {e}")
            self.set_status(f"Failed to save graph: {e}")

    def receive_env_info(self, env_info):
        """Receive environment info from the RL module and update the suptitle."""
        self.env_info = env_info
        if 'selected_idx' in env_info:
            self.suptitle_text = f"{self.suptitle_suffix} - index: {env_info['selected_idx']}"
        else:
            self.suptitle_text = self.suptitle_suffix
        print_log(f"[RenderWindow] Environment info received: {env_info}")


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
        if self.window:
            self.window.set_status(f"Open for connection: {self.host}:{self.port}")

    def listen(self):
        print_log(f"[RenderServer] Listening on {self.host}:{self.port}")
        while True:
            conn, addr = self.server_socket.accept()
            print_log(f"[RenderServer] Connection from {addr}")
            if self.window:
                self.window.set_status(f"Connected with {addr[0]}:{addr[1]}")
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
                    msg = json.loads(data.decode('utf-8'))
                    print_log(f"[RenderServer] Received message: {msg}")
                    if not isinstance(msg, dict) or 'type' not in msg:
                        continue
                    if msg['type'] == RenderWindowMessageType.DATA.name.lower() and self.window:
                        self.window.push_data(msg['payload'])
                        print_log(f"[RenderServer] Data pushed to window.")
                    elif msg['type'] == RenderWindowMessageType.CONTROL.name.lower():
                        cmd = msg.get('command')
                        payload = msg.get('payload', {})
                        if cmd == RenderWindowControl.RESET.name and self.window:
                            self.window.reset_buffers()
                        elif cmd == RenderWindowControl.SAVE_GRAPH.name and self.window:
                            fname = payload.get('fname')      # check whether a path is provided, only save if there is a path
                            if fname:
                                self.window.save_graph(payload)

                        elif cmd == RenderWindowControl.RECEIVE_ENV_INFO.name and self.window:
                            env_info = payload.get('env_info', {})
                            if env_info:
                                self.window.receive_env_info(env_info)

                        elif cmd == RenderWindowControl.CLOSE.name:
                            print_log("[RenderServer] CLOSE command received. Closing connection.")
                            if self.window:
                                self.window.set_status(f"Closed connection with {addr[0]}:{addr[1]}")
                            break  # Close this connection, but keep server running
                        elif cmd == RenderWindowControl.TERMINATE.name:
                            print_log("[RenderServer] TERMINATE command received. Exiting application.")
                            if self.window:
                                self.window.set_status("Render window terminated.")
                                self.window.close()
                            conn.close()
                            sys.exit(0)
            except Exception as e:
                print_log(f"[RenderServer] Error parsing data: {e}")
            conn.close()


def main():
    app = QApplication(sys.argv)
    host = '127.0.0.1'
    port = 50007
    window = RenderWindow(host=host, port=port)
    window.show()
    # Start the TCP server
    server = RenderServer(host=host, port=port, window=window)
    sys.exit(app.exec())
    # close the server socket when the application exits
    server.server_socket.close()

if __name__ == "__main__":
    main()
