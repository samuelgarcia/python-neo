# -*- coding: utf-8 -*-
"""
msrd data format from multichannel system.
This is the new format.

"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

import os
import sys


class MsrdRawIO(BaseRawIO):
    extensions = ['msrd']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        self._info = info = parse_msrd_raw_header(self.filename)
        
        
        self._data_blocks = info['block_by_entities']
        #~ print(info)
        #~ for k, v in info.items():
            #~ print(k, v)
        #~ exit()

        #~ self.dtype = 'uint16'
        #~ self.sampling_rate = info['sampling_rate']
        #~ self.nb_channel = len(info['channel_names'])
        
        #~ self._raw_signals = np.memmap(self.filename, dtype=self.dtype, mode='r',
                                      #~ offset=info['header_size']).reshape(-1, self.nb_channel)
        self._memmap = np.memmap(self.filename, dtype='uint8', mode='r')
        
        
        sig_channels = []
        sig_entities = []
        for s, stream in info['streams'].items():
            if stream['DataType'] == 'Analog':
                print(stream)
                #~ exit()
                #~ nb_chan = int(stream['Entities'])
                for chan in stream['channels']:
                    #~ print(chan)
                    sampling_rate = 1e6 / float(chan['Tick'])
                    if chan['RawDataType'] == 'Int' and chan['ADCBits'] == '16':
                        dt = 'int16'
                    else:
                        raise(NotImplementedError('Only int16 at the moment'))
                    gain = float(chan['ConversionFactor'])
                    offset = 0.
                    #~ print(chan['ChID'], chan['Entity'], chan['Label'])
                    sig_channels.append((chan['Label'], chan['ChID'], sampling_rate,
                                dt, chan['Unit'], gain, offset, int(chan['GroupID'])))
                    sig_entities.append(int(chan['Entity']))
            
            elif stream['DataType'] == 'Event':
                pass
            
            print(sig_channels)
            
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        assert np.all(sig_channels['group_id'] == sig_channels['group_id'][0]), \
                    'Only one signal group is supported'
        
        # this is entity ID
        self.sig_entities = np.array(sig_entities)
        #~ exit()

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        #TODO
        self._seg_t_start = 0.
        self._seg_t_start = 10.
        self._sig_t_start = 0.
        
        entity = self.sig_entities[0]
        #~ print('entity', entity)
        print(self._data_blocks[entity]['size'])
        self._signal_length  = np.sum(self._data_blocks[entity]['size']) // 2
        print('self._signal_length', self._signal_length)
        #~ return size
        #~ exit()
        
        
        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_start
        #~ return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stop
        #~ t_stop = self._raw_signals.shape[0] / self.sampling_rate
        #~ return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._signal_length

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return self._sig_t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._signal_length

        if channel_indexes is None:
            channel_indexes = np.arange(self.header['signal_channels'].size)

        #~ raw_signals = np.zeros((i_stop - i_start, len(channel_indexes)), dtype='int16')
        raw_signals = np.zeros((i_stop - i_start, len(channel_indexes)), dtype='int32')
        for c, channel_index in enumerate(channel_indexes):
            chan_header = self.header['signal_channels'][channel_index]
            #~ chan_id = chan_header['id']
            entity = self.sig_entities[c]
            
            data_blocks = self._data_blocks[entity]
            
            #~ data_blocks = self._data_blocks[5][chan_id]

            # loop over data blocks and get chunks
            bl0 = np.searchsorted(data_blocks['cumsize'], i_start, side='left')
            bl1 = np.searchsorted(data_blocks['cumsize'], i_stop, side='left')
            ind = 0
            for bl in range(bl0, bl1):
                ind0 = data_blocks[bl]['pos']
                ind1 = data_blocks[bl]['size'] + ind0
                data = self._memmap[ind0:ind1].view('int16')
                if bl == bl1 - 1:
                    # right border
                    # be carfull that bl could be both bl0 and bl1!!
                    border = data.size - (i_stop - data_blocks[bl]['cumsize'])
                    data = data[:-border]
                if bl == bl0:
                    # left border
                    border = i_start - data_blocks[bl]['cumsize']
                    data = data[border:]
                raw_signals[ind:data.size + ind, c] = data
                ind += data.size

        return raw_signals


def parse_msrd_raw_header(filename):
    info = {}
    
    with open(filename, mode='rb') as f:
        f.readline()
        f.readline()
        
        
        # header part 0
        hdr0_size = f.readline()
        hdr0_size = int(hdr0_size.split(b'=')[1])
        f.seek(0)
        hdr0_txt = f.read(hdr0_size)
        for line in hdr0_txt.split(b'\n'):
            #~ print(line)
            if b'=' in line:
                k, v = line.split(b'=')
                info[k.decode()] = v.decode()
                
        
        #~ print(info['FPosEntitySizes'])
        
        #~ exit()
        
        
        #~ print('********')
        # header part 1
        f.seek(hdr0_size)
        RecordingID = int(f.readline().split(b'=')[1])
        #~ print('RecordingID', RecordingID)
        first_data_block = int(f.readline().split(b'=')[1])
        #~ print('first_data_block', first_data_block)
        hdr1_size1 = first_data_block - hdr0_size
        f.seek(hdr0_size)
        hdr1_txt = f.read(hdr1_size1)
        
        #~ for line in hdr1_txt.split(b'\r\n'):
            #~ print(line)
        
        #~ streams = hdr1_txt.split(b'\r\nStream=')
        #~ print(len(streams))
        
        txt_streams = hdr1_txt.split(b'\r\nStream=')
        #~ print(len(txt_streams))
        #~ exit()
        for line in txt_streams[0].split(b'\r\n'):
            #~ print(line)
            if b'=' in line:
                k, v = line.split(b'=')
                info[k.decode()] = v.replace(b' ', b'').decode()
            
        
        info['FPosEntitySizes'] = int(info['FPosEntitySizes'])
        #~ print(info['FPosEntitySizes'])
        
        #~ exit()
        
        info['streams'] = streams = {}
        for i, txt_stream in enumerate(txt_streams[1:]):
            #~ print('*****')
            #~ print('stream', i)
            #~ print('*****')
            streams[i] = {}
            #~ for line in txt.split(b'\r\n'):
                #~ print(line)
            txt_channels = txt_stream.split(b'Entity=')
            print(txt_channels[0])
            for line in txt_channels[0].split(b'\r\n'):
                #~ print(line)
                if b'=' in line:
                    k, v = line.split(b'=')
                    streams[i][k.decode()] = v.replace(b' ', b'').decode()
                
            #~ exit()
            
            channels = []
            for c, txt_chan in enumerate(txt_channels[1:]):
                chan = {}
                
                lines = txt_chan.split(b'\r\n')
                chan['Entity'] = int(lines[0])
                for line in lines[1:]:
                    #~ print(line)
                    if b'=' in line:
                        k, v = line.split(b'=')
                        chan[k.decode()] = v.replace(b' ', b'').decode()
                #~ print(chan)
                #~ exit()
                channels.append(chan)
                
                #~ print('channel', c)
                #~ print(txt_chan)
            
            streams[i]['channels'] = channels
            #~ print(len(txt_channels))
            #~ print(streams[i])
            
        
        
        #~ print('********')
        
        
        next_data_block = first_data_block
        
        
        raw_blocks = []
        while next_data_block<info['FPosEntitySizes']: # is not None:
            #~ print('********')
            #~ print('next_data_block', next_data_block)
            data_block = read_one_data_block(f, next_data_block)
            #~ for k, v in data_block.items():
                #~ print('  ', k, ':', v)
            raw_blocks.append((data_block['pos'], data_block['Entity'], data_block['TimeStamp'], data_block['Size'], 0))
            next_data_block = data_block['FPosNext']
            if data_block['FPosNext'] == -1:
                break
            #~ print(next_data_block<info['FPosEntitySizes'])
            
            #~ exit()
        #~ print(blocks)
        _dt = [('pos', 'int64'), ('entity', 'int64'),('timestamp', 'int64'), ('size', 'int64'), ('cumsize', 'int64')]
        info['raw_blocks'] = raw_blocks = np.array(raw_blocks, dtype=_dt)
        entities = np.unique(raw_blocks['entity'])
        block_by_entities = {}
        for entity in entities:
            keep = raw_blocks['entity'] == entity
            block_by_entity = raw_blocks[keep].copy()
            block_by_entity['cumsize'][0] = 0
            block_by_entity['cumsize'][1:] = np.cumsum(block_by_entity['size'][:-1])
            block_by_entities[entity] = block_by_entity
        
        info['block_by_entities'] = block_by_entities
            #~ print('**', entity, '**')
            #~ print(block_by_entity)

    
    #~ exit()

    return info
    


def read_one_data_block(f, pos):
    f.seek(pos)
    data_block = {}
    while True:
        line = f.readline()
        k, v = line.split(b'=')
        data_block[k.decode()] = int(v)
        if line.startswith(b'Size='):
            data_block['pos'] = f.tell()
            break
    
    #~ data_block ['data'] = f.read(data_block['Size'])
        
    #~ print(block)
    
    return data_block
    
    
