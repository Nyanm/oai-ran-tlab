#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Simple doppler injector
# GNU Radio version: 3.10.9.2

import threading
import struct
import sys
import signal
from argparse import ArgumentParser

import zmq

from gnuradio import analog, blocks, gr
from gnuradio.filter import firdes
from gnuradio.fft import window
from gnuradio import zeromq


class untitled(gr.top_block):

    def __init__(self, samp_rate=23040000, rx_freq=0, freq_off=0, zmq_req_src='tcp://127.0.0.1:4445', zmq_req_snk='tcp://127.0.0.1:4446', zmq_pull_addr='tcp://*:5555', zmq_pull_oai_addr='tcp://*5556'):
        gr.top_block.__init__(self, "Simple doppler injector", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.freq_off = freq_off
        self.rx_freq = rx_freq

        ##################################################
        # Blocks
        ##################################################

        self.zeromq_req_source_0 = zeromq.req_source(gr.sizeof_gr_complex, 1, zmq_req_src, 100, False, (-1), False)
        self.zeromq_rep_sink_0 = zeromq.rep_sink(gr.sizeof_gr_complex, 1, zmq_req_snk, 100, False, (-1), True)
        self.blocks_throttle2_0_0 = blocks.throttle(
            gr.sizeof_gr_complex*1, samp_rate, True,
            0 if "auto" == "auto" else max(
                int(float(0.1) * samp_rate) if "auto" == "time" else int(0.1), 1)
        )
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.analog_sig_source_x_0_1_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, freq_off, 1, 0, 0)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0_1_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.zeromq_rep_sink_0, 0))
        self.connect((self.blocks_throttle2_0_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.zeromq_req_source_0, 0), (self.blocks_throttle2_0_0, 0))

        ##################################################
        # ZMQ sockets
        ##################################################
        self._zmq_ctx = zmq.Context()
        self._zmq_ctx_oai = zmq.Context()

        # PULL - receive freq offset updates from Python script
        self._pull_sock = self._zmq_ctx.socket(zmq.PULL)
        self._pull_sock.setsockopt(zmq.LINGER, 0)
        self._pull_sock.bind(zmq_pull_addr)
        print(f"[zmq] PULL bound to {zmq_pull_addr}")

        # PULL - receive carrier freq updates from OAI
        self._pull_sock_oai = self._zmq_ctx_oai.socket(zmq.PULL)
        self._pull_sock_oai.setsockopt(zmq.LINGER, 0)
        self._pull_sock_oai.bind(zmq_pull_oai_addr)
        print(f"[zmq] PULL bound to {zmq_pull_oai_addr}")

        self._zmq_thread = threading.Thread(target=self._zmq_listener, daemon=True)
        self._zmq_thread_oai = threading.Thread(target=self._zmq_listener_oai, daemon=True)

    # ------------------------------------------------------------------
    # Getters / setters
    # ------------------------------------------------------------------

    def get_samp_rate(self):
        return self.samp_rate

    def get_rx_freq(self):
        return self.rx_freq

    def set_rx_freq(self, rx_freq):
        self.rx_freq = rx_freq
        print(f"[rx_freq] -> {self.rx_freq}")

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0_1_0.set_sampling_freq(self.samp_rate)
        self.blocks_throttle2_0_0.set_sample_rate(self.samp_rate)
        print(f"[samp_rate] -> {self.samp_rate}")

    def get_freq_off(self):
        return self.freq_off

    def set_freq_off(self, freq_off):
        self.freq_off = freq_off
        self.analog_sig_source_x_0_1_0.set_frequency(self.freq_off)
        print(f"[freq_off] -> {self.freq_off}")


    # ------------------------------------------------------------------
    # ZMQ PULL listener
    # ------------------------------------------------------------------
 
    def _zmq_listener(self):
        while True:
            try:
                # Accept either a raw 8-byte double or a UTF-8 string
                raw = self._pull_sock.recv()
                if len(raw) == 8:
                    val = struct.unpack('d', raw)[0]   # double from C/numpy sender
                else:
                    val = float(raw.decode('utf-8').strip())
                self.set_freq_off(self.freq_off + val)
            except (ValueError, struct.error) as e:
                print(f"[zmq] invalid message: {e}")
            except zmq.ZMQError as e:
                print(f"[zmq] error: {e}")
                break

    def _zmq_listener_oai(self):
        while True:
            try:
                # Accept either a raw 8-byte double or a UTF-8 string
                raw = self._pull_sock_oai.recv()
                if len(raw) == 8:
                    val = struct.unpack('d', raw)[0]   # double from C/numpy sender
                else:
                    val = float(raw.decode('utf-8').strip())
                self.set_rx_freq(val)
                self.set_freq_off(0)
            except (ValueError, struct.error) as e:
                print(f"[zmq] invalid message: {e}")
            except zmq.ZMQError as e:
                print(f"[zmq] error: {e}")
                break

    def start(self):
        self._zmq_thread.start()
        self._zmq_thread_oai.start()
        super().start()

def main(top_block_cls=untitled, options=None):
    parser = ArgumentParser(description="Simple doppler injector")
    parser.add_argument(
        "--samp-rate", type=int, default=23040000,
        help="Sample rate in Hz (default: 23040000)"
    )
    parser.add_argument(
        "--rx-freq", type=int, default=3619200000,
        help="Carrier freq in Hz (default: 3619200000)"
    )
    parser.add_argument(
        "--freq-off", type=float, default=0.0,
        help="Initial frequency offset in Hz (default: 0.0)"
    )
    parser.add_argument(
        "--zmq-req-src", type=str, default="tcp://127.0.0.1:4445",
        help="ZMQ REQ source address (default: tcp://127.0.0.1:4445)"
    )
    parser.add_argument(
        "--zmq-req-snk", type=str, default="tcp://127.0.0.1:4446",
        help="ZMQ REQ sink address (default: tcp://127.0.0.1:4446)"
    )
    parser.add_argument(
        "--zmq-pull-addr", type=str, default="tcp://*:5555",
        help="ZMQ PULL bind address for incoming freq_off updates (default: tcp://*:5555)"
    )
    parser.add_argument(
        "--zmq-pull-oai-addr", type=str, default="tcp://*:5556",
        help="ZMQ PULL bind address for incoming carrier_freq updates from OAI UE (default: tcp://*:5556)"
    )
    args = parser.parse_args()

    tb = top_block_cls(
        samp_rate=args.samp_rate,
        rx_freq=args.rx_freq,
        freq_off=args.freq_off,
        zmq_req_src=args.zmq_req_src,
        zmq_req_snk=args.zmq_req_snk,
        zmq_pull_addr=args.zmq_pull_addr,
        zmq_pull_oai_addr=args.zmq_pull_oai_addr,
    )

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print(f"Flowgraph running: samp_rate={args.samp_rate}, rx_freq={args.rx_freq}")
    print(f"PULL (freq_off in):    {args.zmq_pull_addr}")
    print(f"PULL (carrier_freq in): {args.zmq_pull_oai_addr}")

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass

    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
