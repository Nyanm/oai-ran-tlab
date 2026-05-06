# live_srs.py
import socket, struct, numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

HOST, PORT = '127.0.0.1', 2021
ofdm_size = 2048
target_id = 33  # GNB_PHY_UL_FREQ_CHANNEL_ESTIMATE

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
lines = []
for i, ax in enumerate(axes):
    line, = ax.plot([], [], 'b-', linewidth=0.8)
    ax.set_xlim(0, ofdm_size)
    ax.set_ylim(0, 600)
    ax.set_title(f'RX Antenna {i} — SRS |H(f)|')
    ax.set_xlabel('Subcarrier')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    lines.append(line)

# Latest data per antenna
latest = {0: None, 1: None, 2: None, 3: None}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
sock.setblocking(False)

# Send activation message for our event
# T-tracer protocol: send "on EVENT_NAME\n"
sock.setblocking(True)
sock.send(b'on GNB_PHY_UL_FREQ_CHANNEL_ESTIMATE\n')
sock.setblocking(False)

buf = b''

def recv_events():
    global buf
    try:
        buf += sock.recv(65536)
    except BlockingIOError:
        return
    
    while len(buf) >= 20:
        # [8B ts][4B id][4B gnb_id][4B rnti][4B frame][4B ant][4B port][4B buflen][data]
        if len(buf) < 36:
            break
        ev_id = struct.unpack_from('<I', buf, 8)[0]
        if ev_id != target_id:
            buf = buf[1:]  # shouldn't happen but safety
            continue
        rx_ant  = struct.unpack_from('<i', buf, 24)[0]
        buf_len = struct.unpack_from('<I', buf, 32)[0]
        total   = 36 + buf_len
        if len(buf) < total:
            break
        if buf_len > 0 and rx_ant in latest:
            raw = buf[36:36+buf_len]
            s   = np.frombuffer(raw, dtype=np.int16)
            I   = s[0::2].astype(float)
            Q   = s[1::2].astype(float)
            latest[rx_ant] = np.sqrt(I**2 + Q**2)
        buf = buf[total:]

def update(frame):
    recv_events()
    for ant, line in enumerate(lines):
        if latest[ant] is not None:
            line.set_data(np.arange(ofdm_size), latest[ant])
    return lines

ani = animation.FuncAnimation(fig, update, interval=100, blit=True)
plt.suptitle('Live SRS Channel Estimates — 4 RU Antenna Positions')
plt.tight_layout()
plt.show()
