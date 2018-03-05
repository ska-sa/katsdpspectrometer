#!/usr/bin/env python3

###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: spt@ska.ac.za                                                       #
# Copyright (c) 2018, National Research Foundation (Square Kilometre Array).  #
# All rights reserved.                                                        #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

"""Capture spectrometer SPEAD streams produced by MeerKAT digitisers."""

import logging
import signal
import enum
import json
import asyncio

import numpy as np
from aiokatcp import DeviceServer, Sensor
import spead2
import spead2.recv.asyncio
import katsdptelstate
import katsdpservices
import katsdpspectrometer


class Status(enum.Enum):
    IDLE = 1
    WAIT_DATA = 2
    CAPTURING = 3
    FINISHED = 4


def _warn_if_positive(value):
    return Sensor.Status.WARN if value > 0 else Sensor.Status.NOMINAL


def channel_ordering(n_chans):
    """Ordering of spectrometer channels in an ECP-64 SPEAD item.

    Parameters
    ----------
    n_chans : int
        Number of spectrometer channels

    Returns
    -------
    spead_index_per_channel : array of int, shape (`n_chans`,)
        Index into SPEAD item of each spectrometer channel, allowing i'th
        channel to be accessed as `spead_data[spead_index_per_channel[i]]`
    """
    pairs = np.arange(n_chans).reshape(-1, 2)
    first_half = pairs[:n_chans // 4]
    second_half = pairs[n_chans // 4:]
    return np.c_[first_half, second_half].ravel()


def unpack_bits(x, partition):
    """Extract a series of bit fields from an integer.

    Parameters
    ----------
    x : uint
        Unsigned integer to be interpreted as a series of bit fields
    partition : sequence of int
        Bit fields to extract from `x` as indicated by their size in bits,
        with the last field ending at the LSB of `x` (as per ECP-64 document)

    Returns
    -------
    fields : list of uint
        The value of each bit field as an unsigned integer
    """
    out = []
    for size in reversed(partition):  # Grab fields starting from LSB
        out.append(x & ((1 << size) - 1))
        x >>= size
    return out[::-1]    # Put back into MSB-to-LSB order


class SpectrometerServer(DeviceServer):
    VERSION = 'sdp-spectrometer-0.1'
    BUILD_STATE = 'katsdpspectrometer-' + katsdpspectrometer.__version__

    def __init__(self, host, port, loop, input_streams, input_interface,
                 telstate, output_stream_name):
        self._telstate = telstate
        self._streams = input_streams

        self._build_state_sensor = Sensor(
            str, 'build-state', 'SDP Spectrometer build state',
            default=self.BUILD_STATE)
        self._status_sensor = Sensor(
            Status, 'status', 'The current status of the spectrometer process',
            default=Status.IDLE)
        self._input_heaps_sensor = Sensor(
            int, 'input-heaps-total',
            'Number of input heaps captured in this session',
            default=0)
        self._input_dumps_sensor = Sensor(
            int, 'input-dumps-total',
            'Number of complete input dumps captured in this session',
            default=0)
        self._input_incomplete_sensor = Sensor(
            int, 'input-incomplete-total',
            'Number of heaps dropped due to being incomplete',
            default=0, status_func=_warn_if_positive)

        super().__init__(host, port, loop=loop)

        self.sensors.add(self._build_state_sensor)
        self.sensors.add(self._status_sensor)
        self.sensors.add(self._input_heaps_sensor)
        self.sensors.add(self._input_dumps_sensor)
        self.sensors.add(self._input_incomplete_sensor)

        n_heaps_per_dump = 3 * len(self._streams)
        self.rx = spead2.recv.asyncio.Stream(
            spead2.ThreadPool(), max_heaps=20 * n_heaps_per_dump,
            ring_heaps=20 * n_heaps_per_dump, contiguous_only=False)

        n_memory_buffers = 80 * n_heaps_per_dump
        heap_size = 2 * 128 * 4 + 64
        memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096,
                                        n_memory_buffers, n_memory_buffers)
        self.rx.set_memory_pool(memory_pool)
        interface_address = katsdpservices.get_interface_address(input_interface)
        for stream in self._streams.values():
            endpoint = katsdptelstate.endpoint.endpoint_parser(7150)(stream)
            for port_offset in range(4):
                if interface_address is not None:
                    self.rx.add_udp_reader(
                        endpoint.host, endpoint.port + port_offset,
                        buffer_size=heap_size + 4096,
                        interface_address=interface_address)
                else:
                    self.rx.add_udp_reader(endpoint.port + port_offset,
                                           bind_hostname=endpoint.host,
                                           buffer_size=heap_size + 4096)

    async def do_capture(self):
        self._status_sensor.set_value(Status.WAIT_DATA)
        logger.info('Waiting for data...')
        ig = spead2.ItemGroup()
        no_heaps_yet = True
        while True:
            try:
                heap = await self.rx.get()
            except spead2.Stopped:
                break
            if isinstance(heap, spead2.recv.IncompleteHeap):
                logger.warning('Dropped incomplete heap %d (received '
                               '%d/%d bytes of payload)', heap.cnt,
                               heap.received_length, heap.heap_length)
                self._input_incomplete_sensor.value += 1
                continue
            if no_heaps_yet:
                logger.info('First spectrometer heap received...')
                self._status_sensor.set_value(Status.CAPTURING)
                no_heaps_yet = False
            new_items = ig.update(heap)
            # If SPEAD descriptors have not arrived yet, keep waiting
            if 'timestamp' not in new_items:
                continue
            timestamp = ig['timestamp'].value
            dig_id = ig['digitiser_id'].value
            dig_status = ig['digitiser_status'].value
            dig_serial, dig_type, receptor, pol = unpack_bits(dig_id,
                                                              (24, 8, 14, 2))
            saturation, nd_on = unpack_bits(dig_status, (8, 1))
            stream = [s[5:] for s in new_items if s.startswith('data_')][0]
            print(receptor, dig_serial, timestamp, nd_on, stream)
        self._input_heaps_sensor.value = 0
        self._input_dumps_sensor.value = 0
        self._status_sensor.value = Status.FINISHED


def on_shutdown(loop, server):
    loop.remove_signal_handler(signal.SIGINT)
    loop.remove_signal_handler(signal.SIGTERM)
    server.rx.stop()
    server.halt()


async def run(loop, server):
    await server.start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: on_shutdown(loop, server))
    await server.do_capture()
    await server.join()


if __name__ == '__main__':
    katsdpservices.setup_logging()
    logger = logging.getLogger("katsdpspectrometer")
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--input-streams', default={}, type=json.loads,
                        help='JSON dict mapping receptor names to spectrometer '
                             'SPEAD stream endpoints (i.e. inputs to process)')
    parser.add_argument('--input-interface', metavar='INTERFACE',
                        help='Network interface to subscribe to for '
                             'spectrometer streams [default=auto]')
    parser.add_argument('--output-stream-name', default='sdp_spectrometer',
                        help='Telstate name of the spectrometer output stream',
                        metavar='NAME')
    parser.add_argument('-p', '--port', type=int, default=2045, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default='', metavar='HOST',
                        help='KATCP host address [default=all hosts]')
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    server = SpectrometerServer(args.host, args.port, loop, args.input_streams,
                                args.input_interface, args.telstate,
                                args.output_stream_name)
    logger.info("Started digitiser spectrometer server")
    loop.run_until_complete(run(loop, server))
    loop.close()
