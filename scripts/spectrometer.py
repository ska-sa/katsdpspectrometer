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
from collections import namedtuple

import numpy as np
import scipy.interpolate as interpolate
from aiokatcp import DeviceServer, Sensor
import spead2
import spead2.recv.asyncio
import katsdptelstate
import katsdpservices
import katsdpspectrometer


# Time after end of L0 dump when all associated spectrometer heaps are deemed
# to be received, in seconds
SETTLE_TIME = 1.0

# Number of frequency channels in spectrometer (currently fixed for L-band)
N_CHANS = 128

# Basic noise diode model, as a temperature in K
NOISE_DIODE_MODEL = 20.0

# Ordering of polarisations in output products
POL_ORDERING = ['v', 'h']

# Number of knots in spline representation of noise diode power spectra
# (used for "post-processing correction" of bandpass shapes)
N_KNOTS = 17

# Spline order used to fit noise diode power spectra
SPLINE_ORDER = 3


class Status(enum.Enum):
    IDLE = 1
    WAIT_DATA = 2
    CAPTURING = 3
    FINISHED = 4


HeapData = namedtuple('HeapData', 'timestamp nd_on dig_serial n_accs data')


def warn_if_positive(value):
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
    x : uint or positive int
        Unsigned / positive integer to be interpreted as a series of bit fields
    partition : sequence of int
        Bit fields to extract from `x` as indicated by their size in bits,
        with the last field ending at the LSB of `x` (as per ECP-64 document)

    Returns
    -------
    fields : list of uint or positive int
        The value of each bit field as an unsigned / positive integer
    """
    out = []
    for size in reversed(partition):  # Grab fields starting from LSB
        out.append(x & ((1 << size) - 1))
        x >>= size
    return out[::-1]    # Put back into MSB-to-LSB order


def cbf_telstate_view(telstate, l0_stream):
    """Create a telstate view that allows querying properties of CBF streams.

    This starts with the parent stream of the main SDP stream (typically
    `baseline-correlation-products`). Properties that don't exist on that CBF
    stream are searched on the upstream `antenna-channelised-voltage` stream,
    and then the CBF instrument of that stream.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`
        Telstate view on main SDP stream

    Returns
    -------
    view : :class:`katsdptelstate.TelescopeState`
        Telstate view that allows CBF stream properties to be searched
    """
    x_stream = telstate.view(l0_stream, exclusive=True)['src_streams'][0]
    f_stream = telstate.view(x_stream, exclusive=True)['src_streams'][0]
    instrument = telstate.view(f_stream, exclusive=True)['instrument_dev_name']
    return telstate.view(instrument, exclusive=True).view(f_stream).view(x_stream)


class SpectrometerServer(DeviceServer):
    VERSION = 'sdp-spectrometer-0.1'
    BUILD_STATE = 'katsdpspectrometer-' + katsdpspectrometer.__version__

    def __init__(self, host, port, loop, input_streams, input_interface,
                 telstate, l0_stream_name, output_stream_name):
        super().__init__(host, port, loop=loop)
        self._streams = input_streams
        self._l0_stream_name = l0_stream_name
        self._output_stream_name = output_stream_name
        self._telstate = telstate.view(output_stream_name)
        cbf_telstate = cbf_telstate_view(telstate, l0_stream_name)
        self._dig_sync_time = cbf_telstate['sync_time']
        self._dig_time_scale = cbf_telstate['scale_factor_timestamp']
        self._l0_int_time = telstate.view(l0_stream_name)['int_time']
        self._l0_dump_end = None
        # Generate uniform sequence of knots across spectrum for spline fits
        self._knots = np.arange(N_KNOTS, dtype=float)
        # Make them slightly denser at edges to get better model of roll-off
        self._knots[0] += 0.5
        self._knots[-1] -= 0.5
        self._knots *= N_CHANS / (N_KNOTS - 1.0)

        # Set up KATCP sensors
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
            default=0, status_func=warn_if_positive)
        self.sensors.add(self._build_state_sensor)
        self.sensors.add(self._status_sensor)
        self.sensors.add(self._input_heaps_sensor)
        self.sensors.add(self._input_dumps_sensor)
        self.sensors.add(self._input_incomplete_sensor)

        # Set up SPEAD receiver and listen to input streams
        n_heaps_per_dump = 3 * len(self._streams)
        self.rx = spead2.recv.asyncio.Stream(
            spead2.ThreadPool(), max_heaps=20 * n_heaps_per_dump,
            ring_heaps=20 * n_heaps_per_dump, contiguous_only=False)
        n_memory_buffers = 80 * n_heaps_per_dump
        heap_size = 2 * N_CHANS * 4 + 72
        memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096,
                                        n_memory_buffers, n_memory_buffers)
        self.rx.set_memory_pool(memory_pool)
        interface_address = katsdpservices.get_interface_address(input_interface)
        for stream in self._streams.values():
            endpoint = katsdptelstate.endpoint.endpoint_parser(7150)(stream)
            for port_offset in range(1):
                if interface_address is not None:
                    self.rx.add_udp_reader(
                        endpoint.host, endpoint.port + port_offset,
                        buffer_size=heap_size + 4096,
                        interface_address=interface_address)
                else:
                    self.rx.add_udp_reader(endpoint.port + port_offset,
                                           bind_hostname=endpoint.host,
                                           buffer_size=heap_size + 4096)
        # Put stream metadata into telstate in the top-level stream namespace
        self._telstate.add('receptors', sorted(self._streams), immutable=True)
        self._telstate.add('pols', POL_ORDERING, immutable=True)
        self._telstate.add('n_chans', N_CHANS, immutable=True)
        self._telstate.add('knots', self._knots, immutable=True)
        # XXX What do you mean it's not L-band???
        self._telstate.add('center_freq', 1284e6, immutable=True)
        self._telstate.add('bandwidth', 856e6, immutable=True)

    def process_l0_dump(self, heaps):
        """Turn an L0 dump worth of heaps into spectrometer products."""
        l0_dump_start = self._l0_dump_end - self._l0_int_time
        l0_timestamp = self._l0_dump_end - 0.5 * self._l0_int_time
        logger.info('Dump %.3f, %d streams, %d heaps', l0_timestamp,
                    len(heaps), sum([len(v) for v in heaps.values()]))
        receptor_lookup = {int(rcp_name[1:]): n
                           for n, rcp_name in enumerate(sorted(self._streams))}
        n_receptors = len(receptor_lookup)
        n_pols = len(POL_ORDERING)
        n_coefs = len(self._knots) + SPLINE_ORDER + 1
        products = {'gain': np.full((n_pols, n_receptors), np.nan),
                    'tsys': np.full((n_pols, n_receptors), np.nan),
                    'nd_power': np.full((n_pols, n_receptors, n_coefs), np.nan,
                                        dtype=np.float16),
                    'nd_phase': np.full((2, n_receptors, n_coefs), np.nan,
                                        dtype=np.float16)}
        dig_serial = [0] * n_receptors
        chans = np.arange(N_CHANS, dtype=float)
        for (receptor_number, stream), stream_heaps in heaps.items():
            receptor_index = receptor_lookup[receptor_number]
            if stream_heaps:
                dig_serial[receptor_index] = stream_heaps[0].dig_serial
            dump_heaps = sorted([heap for heap in stream_heaps
                                 if heap.timestamp >= l0_dump_start and
                                 heap.timestamp < self._l0_dump_end])
            if not dump_heaps:
                continue
            timestamps = np.array([heap.timestamp for heap in dump_heaps])
            nd_on = np.array([heap.nd_on for heap in dump_heaps])
            n_accs = np.array([heap.n_accs for heap in dump_heaps])
            data = np.vstack([heap.data[np.newaxis] for heap in dump_heaps])
            on = np.where(nd_on == 1)[0]
            off = np.where(nd_on == 0)[0]
            if min(len(on), len(off)) < 8:
                continue
            # intervals = np.r_[np.diff(timestamps), np.inf]
            # on_time = min(intervals[on])
            # off_time = min(intervals[off])
            # on_accums = on_time * self._dig_time_scale / (2. * N_CHANS)
            # off_accums = off_time * self._dig_time_scale / (2. * N_CHANS)
            broadcast = (slice(None),) + (data.ndim - 1) * (np.newaxis,)
            logger.info('n_accs[on]:  %s', n_accs[on][:4])
            logger.info('n_accs[off]: %s', n_accs[off][:4])
            data_on = data[on] / n_accs[on][broadcast]
            data_off = data[off] / n_accs[off][broadcast]
            delta = data_on.mean(axis=0) - data_off.mean(axis=0)
            std_delta = np.sqrt(data_on.var(axis=0) + data_off.var(axis=0))
            if stream in ('hh', 'vv'):
                pol_index = POL_ORDERING.index(stream[0])
                power_gain = N_CHANS * np.median(delta) / NOISE_DIODE_MODEL
                voltage_gain = np.sqrt(power_gain)
                tsys = N_CHANS * data_off.mean() / power_gain
                products['gain'][pol_index, receptor_index] = voltage_gain
                products['tsys'][pol_index, receptor_index] = tsys
                tck = interpolate.splrep(chans, delta, 1. / std_delta,
                                         k=SPLINE_ORDER, t=self._knots)
                coefs = tck[1][:n_coefs]
                products['nd_power'][pol_index, receptor_index] = coefs
            else:
                # Channel 0 in FFT is always real, hence stdev of imag part = 0
                # Replace it with massive stdev instead to ignore data point
                std_delta[1, 0] = 100. * std_delta[1].max()
                # Real part of crosscorrelation product VH*
                tck = interpolate.splrep(chans, delta[0], 1. / std_delta[0],
                                         k=SPLINE_ORDER, t=self._knots)
                coefs = tck[1][:n_coefs]
                products['nd_phase'][0, receptor_index] = coefs
                # Imaginary part of crosscorrelation product VH*
                tck = interpolate.splrep(chans, delta[1], 1. / std_delta[1],
                                         k=SPLINE_ORDER, t=self._knots)
                coefs = tck[1][:n_coefs]
                products['nd_phase'][1, receptor_index] = coefs
        for key, value in products.items():
            self._telstate.add(key, value, l0_timestamp)
        if 'dig_serial_number' not in self._telstate and any(dig_serial):
            stream_telstate = self._telstate.view(self._output_stream_name)
            stream_telstate.add('dig_serial_number', dig_serial, immutable=True)

    async def do_capture(self):
        """Receive SPEAD heaps from digitisers while the subarray is up."""
        self._status_sensor.set_value(Status.WAIT_DATA)
        logger.info('Waiting for data...')
        ig = spead2.ItemGroup()
        no_heaps_yet = True
        chans = channel_ordering(N_CHANS)
        heaps = {}
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
            # Extract the relevant items from spectrometer heap
            adc_timestamp = int(ig['timestamp'].value)
            timestamp = self._dig_sync_time + adc_timestamp / self._dig_time_scale
            dig_id = int(ig['digitiser_id'].value)
            dig_status = int(ig['digitiser_status'].value)
            id_fields = unpack_bits(dig_id, (24, 8, 14, 2))
            dig_serial, dig_type, receptor_number, pol = id_fields
            saturation, nd_on = unpack_bits(dig_status, (8, 1))
            n_accs = int(ig['n_accs'].value)
            stream = [s[5:] for s in new_items if s.startswith('data_')][0]
            if stream == 'vh':
                revh = ig['data_' + stream].value[:N_CHANS][chans]
                imvh = ig['data_' + stream].value[N_CHANS:][chans]
                data = np.vstack((revh, imvh))
            else:
                data = ig['data_' + stream].value[chans]
            # Put new heap onto the queue of recent heaps
            key = (receptor_number, stream)
            stream_heaps = heaps.get(key, [])
            stream_heaps.append(HeapData(timestamp, nd_on, dig_serial, n_accs, data))
            heaps[key] = stream_heaps
            # While not capturing correlator data, keep sliding window of heaps
            if self._l0_dump_end is None:
                cutoff = timestamp - self._l0_int_time - 1.1 * SETTLE_TIME
                heaps[key] = [h for h in stream_heaps if h.timestamp >= cutoff]
                continue
            # Batch process an entire L0 dump some time after the dump
            if timestamp > self._l0_dump_end + SETTLE_TIME:
                self.process_l0_dump(heaps)
                for key in heaps:
                    heaps[key] = [heap for heap in heaps[key]
                                  if heap.timestamp >= self._l0_dump_end]
                self._l0_dump_end += self._l0_int_time
        self._input_heaps_sensor.value = 0
        self._input_dumps_sensor.value = 0
        self._status_sensor.value = Status.FINISHED

    async def request_capture_init(self, ctx, capture_block_id: str) -> None:
        """Start a capture block, triggering spectrometer output to telstate."""
        output_capture_stream = self._telstate.join(capture_block_id, self._output_stream_name)
        self._telstate = self._telstate.view(output_capture_stream)
        l0_capture_stream = self._telstate.join(capture_block_id, self._l0_stream_name)
        l0_telstate = self._telstate.view(self._l0_stream_name)
        cb_l0_telstate = self._telstate.view(l0_capture_stream)
        self._l0_dump_end = (l0_telstate['sync_time'] +
                             cb_l0_telstate['first_timestamp'] +
                             0.5 * self._l0_int_time)
        logger.info('Init capture block %s', capture_block_id)

    async def request_capture_done(self, ctx, capture_block_id: str) -> None:
        """Stop capture block, , flush final spectrometer dump and stop output."""
        self._telstate = self._telstate.root().view(self._output_stream_name)
        self._l0_dump_end = None
        logger.info('Done with capture block %s', capture_block_id)


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
    parser.add_argument('--l0-stream-name', default='sdp_l0',
                        help='Name of the associated SDP L0 stream',
                        metavar='NAME')
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
                                args.l0_stream_name, args.output_stream_name)
    logger.info("Started digitiser spectrometer server")
    loop.run_until_complete(run(loop, server))
    loop.close()
