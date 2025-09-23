#!/usr/bin/env python3

import argparse
import logging
import itertools
import usb.core
import usb.util
import sys
import time

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(message)s"
)

class ValidateChAtt(argparse.Action):
    def __call__(self, parse, args, values, option_string=None):
        ch, att = values
        if not ch.isdigit():
            parse.exit(1, f'expected number for channel, but got {ch}\n')
        if not 1 <= int(ch) <= 4:
            parse.exit(1, f'channel number must be within [1,4], but is {ch}\n')
        if not att.isdigit():
            parse.exit(1, f'expected number for attenuation, but got {att}\n')
        if not 0 <= int(att) <= 80:
            parse.exit(1, f'attenuation must be within [0,80], but is {attr}\n')
        opts = getattr(args, self.dest) or []
        opts.append((int(ch), int(att)))
        setattr(args, self.dest, opts)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Minicircuits attenuator: set attenuations for channels',
                                     epilog='--set and --reach are mutually exclusive',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--info', '-i', action='store_true', default=False, help='Get Infos for all connected Mini-Circuits RC*DAT',)
    parser.add_argument('--set', '-s', nargs=2, metavar=("[CH]","[ATT]"), action=ValidateChAtt, help="Set a fixed attenuation ATT on channel CH")
    parser.add_argument('--reach', '-r', nargs=2, metavar=("[CH]","[ATT]"), action=ValidateChAtt, help="Perform an \"attenuation sweep\": reach attenuation ATT on channel CH")
    parser.add_argument('--duration', '-d', action='store', type=float, default=5.0, help='Duration for attenuation sweep to --reach given attenuation (Default: 5)',)
    parser.add_argument('--progress', '-p', action='store_true', default=False, help='If provided, will show progress during attenuation sweep',)
    return parser.parse_args()

def _exec(dev, cmd):
    dev.write(1, cmd)
    ans = dev.read(0x81,64)
    # take from ans(wer) while valid bytes
    sn = itertools.takewhile(lambda x: x < 255 and x > 0, ans)
    # turn bytes into characters and accumulate into string
    return "".join(map(chr, sn))

def devId(dev):
    return f"{dev.idVendor:04x}:{dev.idProduct:04x}"

def _info(dev):
    ident = devId(dev)
    SerialN = _exec(dev, "*:SN?")
    ModelN = _exec(dev, "*:MN?")
    Fw = _exec(dev, "*:FIRMWARE?")

    logging.info(f'{ident}: model is {ModelN}')
    logging.info(f'{ident}: serial number is {SerialN}')
    logging.info(f'{ident}: FW version is {Fw}')

def _get_attenuation(dev):
    ident = devId(dev)
    att = _exec(dev, "*:ATT?")
    # remove leading "*" and split by channels
    return [float(n) for n in att[1:].split(" ")]

def _set_attenuation(dev, ch_att):
    ident = devId(dev)
    for ch, att in ch_att:
        #logging.debug(f"{ident}: Set attenuation channel {ch}: {att} dB")
        att = round(4 * att) / 4 # round to closest 0.25dB
        resp = _exec(dev, f"*:CHAN:{ch}:SETATT:{att};" )

def _continuous_set_attenuation(dev, ch_att_end, duration, show_progress=False, step_duration=0.25):
    ident = devId(dev)

    current = _get_attenuation(dev)
    steps = duration / step_duration
    ch_att_step = [(ch, (att_end - current[ch - 1]) / steps) for ch, att_end in ch_att_end]
    logging.info(f"{ident}: sweep attenuation during {duration} s in {steps} steps: increments {ch_att_step}")
    for i in range(1, int(steps)+1):
        next_att = [(ch, current[ch - 1] + i * att_step) for ch, att_step in ch_att_step]
        _set_attenuation(dev, next_att)
        if show_progress:
            logging.debug(f"{ident}: attenuation for channel(s): {next_att}")
        time.sleep(step_duration)
    _set_attenuation(dev, ch_att_end)

def _get_usb_devs(idVendor, idProduct):
    devs = []
    for dev in usb.core.find(idVendor=idVendor, idProduct=idProduct, find_all=True):
        for configuration in dev:
            for interface in configuration:
                ifnum = interface.bInterfaceNumber
                if not dev.is_kernel_driver_active(ifnum):
                    continue
                try:
                    dev.detach_kernel_driver(ifnum)
                except e:
                    pass
        devs.append(dev)
    return devs

if __name__ == '__main__':
    # Parse the arguments
    args = _parse_args()
    if args.set and args.reach:
        logging.error("--set and --reach are mutually exclusive")
        sys.exit(1)

    # 20ce:0023 is Mini-Circuit RC*DAT device
    devs = _get_usb_devs(idVendor=0x20ce, idProduct=0x0023)
    if len(devs) == 0:
        logging.error('no Mini-Circuits RC*DAT device detected')
        sys.exit(-1)
    if len(devs) > 1:
        logging.error('more than one Mini-Circuits RC*DAT device detected')
        sys.exit(-1)
    dev = devs[0]

    if args.info:
        _info(dev)

    if args.set:
        _set_attenuation(dev, args.set)

    if args.reach:
        _continuous_set_attenuation(dev, args.reach, args.duration, args.progress)

    ident = devId(dev)
    for i, a in enumerate(_get_attenuation(dev)):
        logging.info(f"{ident}: Attenuation channel {i+1}: {a} dB")
    sys.exit(0)
