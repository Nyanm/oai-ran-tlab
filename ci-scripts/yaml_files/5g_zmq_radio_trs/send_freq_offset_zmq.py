#!/usr/bin/env python3
"""
Send a freq offset to the GNURadio doppler injector every second via ZMQ PUSH.

Usage:
    python3 send_freq_offset_zmq.py --carrier-freq 3489420000 --freq-off 50.0 --interval 0.5 --addr tcp://127.0.0.1:5555
"""

import sys
import time
import struct
import signal
import threading
import argparse

import zmq

def main():
    parser = argparse.ArgumentParser(description="Send freq offset to GNURadio via ZMQ PUSH")
    parser.add_argument(
        "--freq-off", type=float, default=50.0,
        help="Frequency offset in Hz to send (e.g. 50.0)"
    )
    parser.add_argument(
        "--addr", type=str, default="tcp://127.0.0.1:5555",
        help="GNURadio PULL address to send freq_off to (default: tcp://127.0.0.1:5555)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Send interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--num-msg", type=int, default=15,
        help="Number of freq offsets to send (default: 15)"
    )
    args = parser.parse_args()

    ctx = zmq.Context()
    # PUSH - send freq_off to GNURadio PULL
    push_sock = ctx.socket(zmq.PUSH)
    push_sock.setsockopt(zmq.LINGER, 0)
    push_sock.connect(args.addr)
    print(f"[zmq] PUSH connected to {args.addr}")

    def sig_handler(sig, frame):
        print("\nStopping.")
        push_sock.close()
        ctx.term()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    print(f"Sending freq_off = {args.freq_off} Hz every {args.interval}s - Ctrl+C to stop")

    msg_count = 0
    while msg_count < args.num_msg:
        push_sock.send(struct.pack('d', args.freq_off))
        print(f"[sent] freq_offset = {args.freq_off} Hz")
        time.sleep(args.interval)
        msg_count += 1

    print("\nStopping.")
    push_sock.close()
    ctx.term()
    sys.exit(0)


if __name__ == "__main__":
    main()
