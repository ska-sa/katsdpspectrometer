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


class SpectrometerServer(DeviceServer):
    VERSION = 'sdp-spectrometer-0.1'
    BUILD_STATE = 'katsdpspectrometer-' + katsdpspectrometer.__version__

    def __init__(self, host, port, loop, input_streams, input_interface,
                 telstate, output_stream_name):
        self._telstate = telstate
        self._streams = input_streams
        self._interface_address = katsdpservices.get_interface_address(
            input_interface)

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
            spead2.ThreadPool(), max_heaps=2 * n_heaps_per_dump,
            ring_heaps=2 * n_heaps_per_dump, contiguous_only=False)

        n_memory_buffers = 8 * n_heaps_per_dump
        heap_size = 2 * 128 * 4 + 64
        memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096,
                                        n_memory_buffers, n_memory_buffers)
        self.rx.set_memory_pool(memory_pool)
        self.rx.stop_on_stop_item = False
        for stream in self._streams.values():
            endpoint = katsdptelstate.endpoint.endpoint_parser(7150)(stream)
            for port_offset in range(4):
                if self._interface_address is not None:
                    self.rx.add_udp_reader(
                        endpoint.host, endpoint.port + port_offset,
                        buffer_size=heap_size + 4096,
                        interface_address=self._interface_address)
                else:
                    self.rx.add_udp_reader(endpoint.port + port_offset,
                                           bind_hostname=endpoint.host,
                                           buffer_size=heap_size + 4096)

    async def do_capture(self):
        n_stops = 0
        self._status_sensor.set_value(Status.WAIT_DATA)
        logger.info('Waiting for data...')
        ig = spead2.ItemGroup()
        first = True
        while True:
            try:
                heap = await self.rx.get()
            except spead2.Stopped:
                break
            if first:
                logger.info('First spectrometer heap received...')
                self._status_sensor.set_value(Status.CAPTURING)
                first = False
            if heap.is_end_of_stream():
                n_stops += 1
                logger.debug("%d/%d spectrometer streams stopped",
                             n_stops, len(self._streams))
                if n_stops >= len(self._streams):
                    self.rx.stop()
                    break
                else:
                    continue
            elif isinstance(heap, spead2.recv.IncompleteHeap):
                logger.warning('Dropped incomplete heap %d (received '
                               '%d/%d bytes of payload)', heap.cnt,
                               heap.received_length, heap.heap_length)
                self._input_incomplete_sensor.value += 1
                continue
            updated = ig.update(heap)
            if 'timestamp' in updated:
                print(updated.keys())
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