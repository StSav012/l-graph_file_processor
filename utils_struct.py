# coding: utf-8
import struct
import sys
from typing import Tuple, Union, Dict, List, BinaryIO

if sys.version_info >= (3, 8):
    from typing import Final
else:
    # stub
    class Final:
        @staticmethod
        def __getitem__(item):
            return item


class LCardParameters:
    _struct: Final[str] = '<20s 17s 26s 2x h I I d f f f 32x 32x 32x 32x'
    _max_adc_codes: Dict[bytes, int] = {
        b'E140': 8000,
        b'E502': 6000000,
    }
    _adc_ranges: Dict[bytes, List[float]] = {
        b'E140': [10., 2.5, .625, .15625],
        b'E502': [10., 5., 2., 1., .5, .2],
    }
    _data_type: Dict[bytes, str] = {
        b'E140': 'h',
        b'E502': 'd',
    }
    _rescaling_required: Dict[bytes, bool] = {
        b'E140': True,
        b'E502': False,
    }

    def __init__(self, source: Union[None, bytes, BinaryIO] = None) -> None:
        self.code: bytes = b''
        self.device_name: bytes = b''
        self.time: bytes = b''
        self.channels_count: int = 0
        self.frames_count: int = 0
        self.samples_count: int = 0
        self.duration: float = 0.
        self.adc_rate: float = 0.
        self.frame_delay: float = 0.
        self.channel_rate: float = 0.
        self.serial_number: bytes = b''
        raw_data: bytes
        if source is None:
            return
        elif isinstance(source, bytes):
            raw_data = source
        elif isinstance(source, BinaryIO):
            init_pos: int = 0
            if source.seekable():
                init_pos = source.tell()
            raw_data = source.read(max(struct.calcsize(self._struct), 0x439c + 8))
            if source.seekable():
                source.seek(init_pos)
        else:
            raise TypeError(f'Unknown source type: {type(source)}')
        (
            self.code,
            self.device_name,
            self.time,
            self.channels_count,
            self.frames_count,
            self.samples_count,
            self.duration,
            self.adc_rate,
            self.frame_delay,
            self.channel_rate,
        ) = struct.unpack_from(self._struct, raw_data)
        if len(raw_data) >= 0x439c + 8:
            self.serial_number = raw_data[0x439c:0x439c + 8]
        self.code = self.code.strip(b' \0')
        self.device_name = self.device_name.strip(b'\0')
        self.time = self.time.strip(b' \0')
        # hardware corrections
        self.correction_factor: Tuple[float, ...] = tuple()
        self.correction_offset: Tuple[float, ...] = tuple()
        self.MAX_ADC_CODE: int = 0
        if self.device_name in self._max_adc_codes:
            self.MAX_ADC_CODE = self._max_adc_codes[self.device_name]
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')
        self.MAX_ADC_VOLTAGE: float = 10.
        if self.device_name in self._adc_ranges:
            self.MAX_ADC_VOLTAGE = 10.  # FIXME: get the value from parameters
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')
        if self.device_name in self._adc_ranges:
            self.correction_factor = struct.unpack_from('<' + 'd' * len(self._adc_ranges[self.device_name]),
                                                        raw_data, offset=0x00eb)
            self.correction_offset = struct.unpack_from('<' + 'd' * len(self._adc_ranges[self.device_name]),
                                                        raw_data, offset=0x01eb)
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')

        self.DATA_TYPE: str
        self.DATA_TYPE_SIZE: int
        if self.device_name in self._data_type:
            self.DATA_TYPE = self._data_type[self.device_name]
            self.DATA_TYPE_SIZE = struct.calcsize(self.DATA_TYPE)
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')

        self.RESCALING_REQUIRED: bool
        if self.device_name in self._rescaling_required:
            self.RESCALING_REQUIRED = self._rescaling_required[self.device_name]
        else:
            raise NotImplementedError(f'Unknown device type: {self.device_name!r}')


def read_array(source: Union[bytes, BinaryIO], _type: str, count: int = 1) -> Tuple[Union[float, int, bool], ...]:
    if count < 1:
        raise ValueError(f'Invalid count: {count}')
    if _type not in ['c', 'b', 'B', '?', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'n', 'N', 'e', 'f', 'd']:
        raise ValueError(f'Invalid data type: {_type}')
    _full_struct: str = _type * count
    _struct_size = struct.calcsize(_type)
    _full_struct_size = struct.calcsize(_full_struct)
    raw_data: bytes
    if isinstance(source, bytes):
        raw_data = source
    elif isinstance(source, BinaryIO):
        raw_data = source.read(_full_struct_size)
    else:
        raise TypeError(f'Unknown source type: {type(source)}')
    data: Tuple[Union[float, int, bool], ...]
    if len(raw_data) == _full_struct_size:
        data = struct.unpack('<' + _full_struct, raw_data)
    else:
        available_count: int = len(raw_data) // _struct_size
        data = struct.unpack('<' + _type * available_count, raw_data[:_struct_size * available_count])
    return data
