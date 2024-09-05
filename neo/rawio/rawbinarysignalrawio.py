"""
Class for reading data in a raw binary interleaved compact file.
Sampling rate, units, number of channel and dtype must be externally known.
This generic format is quite widely used in old acquisition systems
and is quite universal for sharing data.

The write part of this IO is only available at neo.io level with the other
class RawBinarySignalIO

Important release note:
  * Since the version neo 0.6.0 and the neo.rawio API,
    arguments of the IO (dtype, nb_channel, sampling_rate) must be
    given at the __init__ and not at read_segment() because there is
    no read_segment() in neo.rawio classes.


Author: Samuel Garcia
"""

import numpy as np

import os

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


class RawBinarySignalRawIO(BaseRawIO):
    """
    Class for reading raw binary files with user specified values
    Parameters
    ----------
    filename: str, default: ''
        The *.raw, *.bin, or *.dat binary file to load
    dtype: np.dtype, default: 'int16'
        The dtype that the data is stored with. Must be acceptable by the numpy.dtype constructor
    sampling_rate: float, default: 10000.0
        The sampling rate of the file
    nb_channel: int, default: 2
        The number of channels for the file
    signal_gain: float, default: 1.0
        The gain for the signal in the binary file
    signal_offset: float, default: 0.0
        The offset for the signal in the binary file
    bytesoffset: int: 0
        The offset for the bytes
    """

    extensions = ["raw", "bin", "dat"]
    rawmode = "one-file"

    def __init__(
        self,
        filename="",
        dtype="int16",
        sampling_rate=10000.0,
        nb_channel=2,
        signal_gain=1.0,
        signal_offset=0.0,
        bytesoffset=0,
    ):
        BaseRawIO.__init__(self)
        self.filename = filename
        self.dtype = dtype
        self.sampling_rate = sampling_rate
        self.nb_channel = nb_channel
        self.signal_gain = signal_gain
        self.signal_offset = signal_offset
        self.bytesoffset = bytesoffset

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        if os.path.exists(self.filename):
            self._raw_signals = np.memmap(self.filename, dtype=self.dtype, mode="r", offset=self.bytesoffset).reshape(
                -1, self.nb_channel
            )
        else:
            # The the neo.io.RawBinarySignalIO is used for write_segment
            self._raw_signals = None

        signal_channels = []
        if self._raw_signals is not None:
            for c in range(self.nb_channel):
                name = f"ch{c}"
                chan_id = f"{c}"
                units = ""
                stream_id = "0"
                buffer_id = "0"
                signal_channels.append(
                    (
                        name,
                        chan_id,
                        self.sampling_rate,
                        self.dtype,
                        units,
                        self.signal_gain,
                        self.signal_offset,
                        stream_id,
                        buffer_id
                    )
                )

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # one unique stream and buffer
        if signal_channels.size > 0:
            signal_buffers = np.array([("Signals", "0")], dtype=_signal_buffer_dtype)
            signal_streams = np.array([("Signals", "0", "0")], dtype=_signal_stream_dtype)
        else:
            signal_streams = np.array([], dtype=_signal_stream_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self.sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("stream_index must be 0")
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        if stream_index != 0:
            raise ValueError("stream_index must be 0")
        return 0.0

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals
