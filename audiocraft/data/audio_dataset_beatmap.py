# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""AudioDataset support. In order to handle a larger number of files
without_origin having to scan again the folders, we precompute some metadata
(filename, sample rate, duration), and use that to efficiently sample audio segments.
"""
import argparse
import copy
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, fields
from contextlib import ExitStack
from functools import lru_cache
import gzip
import json
import logging
import os
from pathlib import Path
import random
import sys
import typing as tp
import math
import numpy as np
import librosa

import torch
import torch.nn.functional as F

from .audio import audio_read, audio_info
from .audio_utils import convert_audio
from .zip import PathInZip

try:
    import dora
except ImportError:
    dora = None  # type: ignore


@dataclass(order=True)
class BaseInfo:

    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
            field.name: dictionary[field.name]
            for field in fields(cls) if field.name in dictionary
        }

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {
            field.name: self.__getattribute__(field.name)
            for field in fields(self)
            }


@dataclass(order=True)
class AudioMeta(BaseInfo):
    id: str
    beatmap_info_path: str
    song_path: str # path to song
    sample_rate: int
    beatmap_file_path: str # path to beatmap_file
    difficulty: str
    bpm: float
    njs: float
    njsoffset: float
    colorNotes_num: int
    bombNotes_num: int
    obstacles_num: int
    arcs_num: int
    chains_num: int
    duration: tp.Optional[float] = None
    amplitude: tp.Optional[float] = None
    weight: tp.Optional[float] = None
    # info_path is used to load additional information about_origin the audio file that is stored in zip files.
    info_path: tp.Optional[PathInZip] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        if 'info_path' in base and base['info_path'] is not None:
            base['info_path'] = PathInZip(base['info_path'])
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d['info_path'] is not None:
            d['info_path'] = str(d['info_path'])
        return d

@dataclass
class Beatmap:
    minimum_note: float # 最小的beat eg 0.125 =>八分音符
    token_id_size: int # 以后记得更改！
    position_size: int
    #note toggle
    use_colorNotes: bool
    use_chains: bool 
    use_bombNotes: bool
    use_obstacles: bool
    use_arcs: bool

    # (posX, posY) => token pos
    position_map = {(0, 0): 0, (1, 0): 1, (2, 0): 2, (3, 0): 3, (0, 1): 4, (1, 1): 5, (2, 1): 6, (3, 1): 7, (0, 2): 8, (1, 2): 9, (2, 2): 10, (3, 2): 11}
    position_map_reversed = {value:key for key, value in position_map.items()}
    # (color, direction) => token id
    color_note_map = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5, (0, 6): 6, (0, 7): 7, (0, 8): 8, (1, 0): 9, (1, 1): 10, (1, 2): 11, (1, 3): 12, (1, 4): 13, (1, 5): 14, (1, 6): 15, (1, 7): 16, (1, 8): 17}
    color_note_map_reversed = {value:key for key, value in color_note_map.items()}
    def __post_init__(self):
        assert self.use_colorNotes, "use color note by default"
        # assign token id to note depending on note toggle
        # chain, color => tokenid
        padding_id = 18
        if self.use_chains:
            self.chain_red_id = padding_id
            self.chain_blue_id = padding_id + 1
            padding_id = padding_id + 2
            self.chain_color_map = {0: self.chain_red_id, 1: self.chain_blue_id}
            self.chain_color_map_reversed = {value:key for key, value in self.chain_color_map.items()}
            self.threadhold_duration_chain = 0.5
        # bomb    
        if self.use_bombNotes:
            self.bomb_note_id = padding_id
            padding_id += 1
        # arc
        if self.use_arcs:
            self.arc_head_id = padding_id
            self.arc_tail_id = padding_id + 1
            padding_id = padding_id + 2
        # obstacles
        if self.use_obstacles:
            self.obstacle_head_id = padding_id
            self.obstacle_tail_id = padding_id + 1  # if no tail follows, assume obstacle occupies only one time spot
            padding_id = padding_id + 2
        assert padding_id == self.token_id_size
    
    empty_light = {
            "customData": {},
            "waypoints": [],
            "basicEvents": [],
            "colorBoostEvents": [],
            "lightColorEventBoxGroups": [],
            "lightRotationEventBoxGroups": [],
            "lightTranslationEventBoxGroups": [],
            "fxEventBoxGroups": [],
            "basicEventTypesWithKeywords": {"list": []},
            "useNormalEventsAsCompatibleEvents": True
        }

    def sample_beatmap_file(self, beatmap: json, seek_time_in_quaver: int, end_time_in_quaver: int):
        colorNotes = []
        chains = []
        bombNotes = []
        arcs = []
        obstacles = []
        for color_note in beatmap['difficulty']['colorNotes']:
            is_integer, time_after = self.time_map(color_note['time'])
            if is_integer and time_after >= seek_time_in_quaver and time_after < end_time_in_quaver:
                color_note['time'] = color_note['time'] - seek_time_in_quaver * self.minimum_note
                colorNotes.append(color_note)
        beatmap['difficulty']['colorNotes'] = colorNotes
        beatmap['difficulty']['chains'] = chains
        beatmap['difficulty']['bombNotes'] = bombNotes
        beatmap['difficulty']['arcs'] = arcs
        beatmap['difficulty']['obstacles'] = obstacles
        #清空lightshow，方便后续测试
        beatmap["lightshow"] = self.empty_light
        return beatmap
    
    def time_map(self, time: float):
        # 返回note的时值，例如一个全音符=>8，两个=>16
        time_after = time / self.minimum_note
        if time_after.is_integer():
            return True, int(time_after)
        else:
            return False, time_after
    
    def tokenize(self, beatmap_origin: json, segment_duration_in_quaver) -> torch.Tensor:
        """
        :param beatmap data: JSON
        """
        unsupported_note = []
        token = torch.full((segment_duration_in_quaver, self.position_size), self.token_id_size, dtype=torch.int64)
        if self.use_colorNotes:
            for color_note in beatmap_origin['difficulty']['colorNotes']:
                is_integer, time_after = self.time_map(color_note['time'])
                if is_integer:
                    token_pos = self.position_map[(color_note['posX'], color_note['posY'])]
                    token_id = self.color_note_map[(color_note['color'], color_note['direction'])]
                    token[time_after, token_pos] = token_id
                else:
                    unsupported_note.append((color_note, "时间不是minimum note的倍数"))
        if self.use_chains:
            for chain in beatmap_origin['difficulty']['chains']:
                head_is_integer, head_time_after = self.time_map(chain['time'])
                tail_is_integer, tail_time_after = self.time_map(chain['tailTime'])
                if head_is_integer and tail_is_integer:
                    # 确保head time对应的note存在
                    token_pos = self.position_map[(chain['posX'], chain['posY'])]
                    token_id = self.color_note_map[(chain['color'], chain['direction'])]
                    if token[head_time_after, token_pos] == token_id:
                        token_pos = self.position_map[(chain['tailPosX'], chain['tailPosY'])]
                        token[tail_time_after, token_pos] = self.chain_color_map[chain['color']]
                    else:
                        unsupported_note.append(chain)
                else:
                    unsupported_note.append(chain)
        if self.use_bombNotes:
            for bomb in beatmap_origin['difficulty']['bombNotes']:
                is_integer, time_after = self.time_map(bomb['time'])
                if is_integer:
                    token_pos = self.position_map[(bomb['posX'], bomb['posY'])]
                    token_id = self.bomb_note_id
                    token[time_after, token_pos] = token_id
                else:
                    unsupported_note.append(bomb)
            
        # tokenize arc note
        if self.use_arcs:
            for arc in beatmap_origin['difficulty']['arcs']:
                head_is_integer, head_time_after = self.time_map(arc['time'])
                tail_is_integer, tail_time_after = self.time_map(arc['tailTime'])
                if head_is_integer and tail_is_integer:
                    # 确保head time和tail time对应的note存在
                    head_token_pos = self.position_map[(arc['posX'], arc['posY'])]
                    head_token_id = self.color_note_map[(arc['color'], arc['direction'])]
                    tail_token_pos = self.position_map[(arc['tailPosX'], arc['tailPosY'])]
                    tail_token_id = self.color_note_map[(arc['color'], arc['tailDirection'])]
                    if token[head_time_after, head_token_pos] == head_token_id and token[tail_time_after, tail_token_pos] == tail_token_id:
                        # 在tail note 后一个时间单位同一个位置添加，先检查是否有note
                        if token[head_time_after+1, head_token_pos] == self.token_id_size and token[tail_time_after-1, tail_token_pos] == self.token_id_size:
                            token[head_time_after+1, head_token_pos] = self.arc_head_id
                            token[tail_time_after-1, tail_token_pos] = self.arc_tail_id
                        else:
                            unsupported_note.append((arc, "arc添加的位置已经有note了"))
                    else:
                        unsupported_note.append((arc, "arc的head time和tail time对应的note不存在"))
                else:
                    unsupported_note.append((arc, "时间不是minimum note的倍数"))
        # tokenize obstacle note
        # if self.use_obstacles:
        #     for obstacle in beatmap_origin['difficulty']['obstacles']:
        #         is_integer, time_after = self.time_map(obstacle['time'])
        #         duration_is_integer, duration_time_after = self.time_map(obstacle['duration '])
        #         if is_integer and duration_is_integer:
        #             # 检查横轴需要放墙的地方
        #             posX = obstacle['posX']
        #             width = obstacle['width']
        #             posY = obstacle['posY']
        #             posX_list = list(range(posX, posX+width))
        #             posY_list = list(range(posY, 4))
                    
                    
        if unsupported_note:
            print("number of beatmap.unsupported_note:", len(unsupported_note))
            unsupported_note = [(tuple((key, value) for key, value in sorted(note.items()) if not isinstance(value, dict)),msg) for note, msg in unsupported_note]
            for item, msg in unsupported_note:
                print(item, msg)
        return token
    def detokenize(self, tokens: torch.Tensor):
        beatmap_reconstructe = {'difficulty':{}}
        colorNotes = []
        chains = []
        bombNotes = []
        arcs = []
        obstacles = []
        hold_stack = {}
        
        length = tokens.shape[0]
        for time in range(length):
            for pos in range(self.position_size):
                token_id = int(tokens[time][pos])
                time_after = time * self.minimum_note
                posX, posY = self.position_map_reversed[pos]
                # look for color note
                if self.use_colorNotes and token_id in self.color_note_map_reversed:
                    color, direction = self.color_note_map_reversed[token_id]
                    colorNotes.append({
                        "color": color,
                        "direction": direction,
                        "posX": posX,
                        "posY": posY,
                        "time": time_after,
                        "angleOffset": 0,
                        "laneRotation": 0,
                        "customData": {}
                        })
                    # look for arc at the same time
                    if hold_stack and color in hold_stack and tokens[time-1][pos] == self.arc_tail_id:
                        head_color_note = hold_stack.pop(color)
                        arcs.append({
                            "color": color,
                            "time": head_color_note['time'],
                            "posX": head_color_note['posX'],
                            "posY": head_color_note['posY'],
                            "direction": head_color_note['direction'],
                            "lengthMultiplier": 1,
                            "tailTime": time_after,
                            "tailPosX": posX,
                            "tailPosY": posY,
                            "tailDirection": direction,
                            "tailLengthMultiplier": 1,
                            "midAnchor": 0,
                            "laneRotation": 0,
                            "tailLaneRotation": 0,
                            "customData": {}
                            })
                # look for chain note
                elif self.use_chains and token_id in self.chain_color_map_reversed:
                    color = self.chain_color_map_reversed[token_id]
                    for color_note in reversed(colorNotes):
                        if time_after == color_note['time'] or color_note['color'] != color:
                            continue
                        elif time_after - color_note['time'] > self.threadhold_duration_chain:
                            break
                        chains.append({
                            "color": color,
                            "time": color_note['time'],
                            "posX": color_note['posX'],
                            "posY": color_note['posY'],
                            "direction": color_note['direction'],
                            "tailTime": time_after,
                            "tailPosX": posX,
                            "tailPosY": posY,
                            "sliceCount": 5,
                            "squish": 1,
                            "tailLaneRotation": 0,
                            "laneRotation": 0,
                            "customData": {}
                            })
                # look for bomb note
                elif self.use_bombNotes and token_id == self.bomb_note_id:
                    bombNotes.append({
                        "posX": posX,
                        "posY": posY,
                        "time": time_after,
                        "customData": {},
                        "laneRotation": 0,
                        "color": -1,
                        "direction": 0
                        })
                # look for arc 
                elif self.use_arcs and token_id == self.arc_head_id:
                    # 遇到arc head token，找上一个时间同一位置的token
                    for color_note in reversed(colorNotes):
                        if time_after - color_note['time'] == self.minimum_note and posX == color_note['posX'] and posY == color_note['posY']:
                            # 放入stack中直到遇到连接着的color note 和arc tail note再处理
                            hold_stack[color_note['color']] = color_note
                        elif time_after - color_note['time'] > self.minimum_note:
                            break
                
        beatmap_reconstructe['difficulty']['colorNotes'] = colorNotes
        beatmap_reconstructe['difficulty']['chains'] = chains
        beatmap_reconstructe['difficulty']['bombNotes'] = bombNotes
        beatmap_reconstructe['difficulty']['arcs'] = arcs
        beatmap_reconstructe['difficulty']['obstacles'] = obstacles
        beatmap_reconstructe["lightshow"] = self.empty_light
        return beatmap_reconstructe

    def check_difference(self, origin_data: json, reconstructed_data: json):
        note_types = []
        if self.use_colorNotes:
            note_types.append('colorNotes')
        if self.use_chains:
            note_types.append('chains')
        if self.use_bombNotes:
            note_types.append('bombNotes')
        if self.use_arcs:
            note_types.append('arcs')
        if self.use_obstacles:
            note_types.append('obstacles')
        output = ""
        for note_type in note_types:
            # with open('beatmap_origin.json', 'w') as json_file:
            #     json.dump(data['difficulty'][note_type], json_file)
            # with open('beatmap_reconstructed.json', 'w') as json_file:
            #     json.dump(data2['difficulty'][note_type], json_file)
            output += "*************************************\n"
            output += f"比较{note_type}, origin_len = {len(origin_data['difficulty'][note_type])}, reconstructed_len = {len(reconstructed_data['difficulty'][note_type])}\n"
            output += f"直接比较: {origin_data['difficulty'][note_type] == reconstructed_data['difficulty'][note_type]}\n"

            data1 = [tuple((key, float(value) if key == 'time' or key == 'tailTime' else value ) for key, value in sorted(note.items()) if not isinstance(value, dict)) for note in origin_data['difficulty'][note_type]]
            data2 = [tuple((key, float(value) if key == 'time' or key == 'tailTime' else value ) for key, value in sorted(note.items()) if not isinstance(value, dict)) for note in reconstructed_data['difficulty'][note_type]]
            counter1 = set(data1)
            counter2 = set(data2)

            if counter1 == counter2:
                output += "两个字典相同\n"
            else:
                output += "两个字典不同\n"
                output += "原始数据中有而还原数据中没有的元素:\n"
                for item in counter1 - counter2:
                    output += f"{item}\n"
                output += "还原数据中有而原始数据中没有的元素:\n"
                for item in counter2 - counter1:
                    output += f"{item}\n"
            if counter1 == counter2:
                output = ""
        return output

@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: AudioMeta
    # seek time in beat
    seek_time: int 
    n_frames: int      # actual number of frames without_origin padding
    total_frames: int  # total number of frames, padding included
    sample_rate: int   # actual sample rate
    channels: int      # number of audio channels.
    segment_duration_in_quaver: int


    origin_sample: torch.Tensor
    beatmap_file: json
    beatmap_class: Beatmap
    
    audio_token: tp.Optional[torch.Tensor] = None
    beatmap_token: tp.Optional[torch.Tensor] = None
    
DEFAULT_EXTS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.egg']

logger = logging.getLogger(__name__)


def _get_audio_meta(audio_meta: dict) -> AudioMeta:
    """AudioMeta from a path to an audio file.

    Args:
        file_path (str): Resolved path of valid audio file.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
    Returns:
        AudioMeta: Audio file path and its metadata.
    """
    info = audio_info(audio_meta['song_path'])
    audio_meta['sample_rate'] = info.sample_rate
    audio_meta['duration'] = info.duration
    amplitude: tp.Optional[float] = None
    if not audio_meta['minimal']:
        wav, sr = audio_read(audio_meta['song_path'])
        amplitude = wav.abs().max().item()
        audio_meta['amplitude'] = info.amplitude
    return AudioMeta.from_dict(audio_meta)


def _resolve_audio_meta(m: AudioMeta, fast: bool = True) -> AudioMeta:
    """If Dora is available as a dependency, try to resolve potential relative paths
    in list of AudioMeta. This method is expected to be used when loading meta from file.

    Args:
        m (AudioMeta): Audio meta to resolve.
        fast (bool): If True, uses a really fast check for determining if a file
            is already absolute or not. Only valid on Linux/Mac.
    Returns:
        AudioMeta: Audio meta with resolved path.
    """
    def is_abs(m):
        if fast:
            return str(m)[0] == '/'
        else:
            os.path.isabs(str(m))

    if not dora:
        return m

    if not is_abs(m.song_path):
        m.song_path = dora.git_save.to_absolute_path(m.song_path)
    if not is_abs(m.beatmap_file_path):
        m.beatmap_file_path = dora.git_save.to_absolute_path(m.beatmap_file_path)
    if not is_abs(m.beatmap_info_path):
        m.beatmap_info_path = dora.git_save.to_absolute_path(m.beatmap_info_path)
    if m.info_path is not None and not is_abs(m.info_path.zip_path):
        m.info_path.zip_path = dora.git_save.to_absolute_path(m.song_path)
    return m


def find_audio_files(input_meta: tp.List[dict], 
                     exts: tp.List[str] = DEFAULT_EXTS,
                     resolve: bool = True,
                     minimal: bool = True,
                     progress: bool = False,
                     workers: int = 0) -> tp.List[AudioMeta]:
    """Build a list of AudioMeta from a given path,
    collecting relevant audio files and fetching meta info.

    Args:
        path (str or Path): Path to folder containing audio files.
        exts (list of str): List of file extensions to consider for audio files.
        minimal (bool): Whether to only load the minimal set of metadata (takes longer if not).
        progress (bool): Whether to log progress on audio files collection.
        workers (int): number of parallel workers, if 0, use only the current thread.
    Returns:
        list of AudioMeta: List of audio file path and its metadata.
    """
    audio_files = []
    futures: tp.List[Future] = []
    pool: tp.Optional[ThreadPoolExecutor] = None
    with ExitStack() as stack:
        if workers > 0:
            pool = ThreadPoolExecutor(workers)
            stack.enter_context(pool)

        if progress:
            print("Finding audio files...")
        for item in input_meta:
            file_path = item['song_path']
            full_path = Path(file_path)
            if full_path.suffix.lower() in exts:
                item['minimal'] = minimal
                audio_files.append(item)
                if pool is not None:
                    futures.append(pool.submit(_get_audio_meta, audio_files[-1]))
                if progress:
                    print(format(len(audio_files), " 8d"), end='\r', file=sys.stderr)
        if progress:
            print("Getting audio metadata...")
        meta: tp.List[AudioMeta] = []
        for idx, item in enumerate(audio_files):
            try:
                if pool is None:
                    m = _get_audio_meta(item)
                else:
                    m = futures[idx].result()
                if resolve:
                    m = _resolve_audio_meta(m)
            except Exception as err:
                print("Error with", str(item), err, file=sys.stderr)
                continue
            meta.append(m)
            if progress:
                print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta


def load_audio_meta(path: tp.Union[str, Path],
                    resolve: bool = True, fast: bool = True) -> tp.List[AudioMeta]:
    """Load list of AudioMeta from an optionally compressed json file.

    Args:
        path (str or Path): Path to JSON file.
        resolve (bool): Whether to resolve the path from AudioMeta (default=True).
        fast (bool): activates some tricks to make things faster.
    Returns:
        list of AudioMeta: List of audio file path and its total duration.
    """
    open_fn = gzip.open if str(path).lower().endswith('.gz') else open
    with open_fn(path, 'rb') as fp:  # type: ignore
        lines = fp.readlines()
    meta = []
    for line in lines:
        d = json.loads(line)
        m = AudioMeta.from_dict(d)
        if resolve:
            m = _resolve_audio_meta(m, fast=fast)
        meta.append(m)
    return meta


def save_audio_meta(path: tp.Union[str, Path], meta: tp.List[AudioMeta]):
    """Save the audio metadata to the file pointer as json.

    Args:
        path (str or Path): Path to JSON file.
        metadata (list of BaseAudioMeta): List of audio meta to save.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    open_fn = gzip.open if str(path).lower().endswith('.gz') else open
    with open_fn(path, 'wb') as fp:  # type: ignore
        for m in meta:
            json_str = json.dumps(m.to_dict()) + '\n'
            json_bytes = json_str.encode('utf-8')
            fp.write(json_bytes)


class AudioDataset:
    """Base audio dataset.

    The dataset takes a list of AudioMeta and create a dataset composed of segments of audio
    and potentially additional information, by creating random segments from the list of audio
    files referenced in the metadata and applying minimal data pre-processing such as resampling,
    mixing of channels, padding, etc.

    If no segment_duration value is provided, the AudioDataset will return the full wav for each
    audio file. Otherwise, it will randomly sample audio files and create a segment of the specified
    duration, applying padding if required.

    By default, only the torch Tensor corresponding to the waveform is returned. Setting return_info=True
    allows to return a tuple containing the torch Tensor and additional metadata on the segment and the
    original audio meta.

    Note that you can call `start_epoch(epoch)` in order to get
    a deterministic "randomization" for `shuffle=True`.
    For a given epoch and dataset index, this will always return the same extract.
    You can get back some diversity by setting the `shuffle_seed` param.

    Args:
        meta (list of AudioMeta): List of audio files metadata.
        segment_duration (float, optional): Optional segment duration of audio to load.
            If not specified, the dataset will load the full audio segment from the file.
        shuffle (bool): Set to `True` to have the data reshuffled at every epoch.
        sample_rate (int): Target sample rate of the loaded audio samples.
        channels (int): Target number of channels of the loaded audio samples.
        sample_on_duration (bool): Set to `True` to sample segments with probability
            dependent on audio file duration. This is only used if `segment_duration` is provided.
        sample_on_weight (bool): Set to `True` to sample segments using the `weight` entry of
            `AudioMeta`. If `sample_on_duration` is also True, the actual weight will be the product
            of the file duration and file weight. This is only used if `segment_duration` is provided.
        min_segment_ratio (float): Minimum segment ratio to use when the audio file
            is shorter than the desired segment.
        max_read_retry (int): Maximum number of retries to sample an audio segment from the dataset.
        return_info (bool): Whether to return the wav only or return wav along with segment info and metadata.
        min_audio_duration (float, optional): Minimum audio file duration, in seconds, if provided
            audio shorter than this will be filtered out_origin.
        max_audio_duration (float, optional): Maximal audio file duration in seconds, if provided
            audio longer than this will be filtered out_origin.
        shuffle_seed (int): can be used to further randomize
        load_wav (bool): if False, skip loading the wav but returns a tensor of 0
            with the expected segment_duration (which must be provided if load_wav is False).
        permutation_on_files (bool): only if `sample_on_weight` and `sample_on_duration`
            are False. Will ensure a permutation on files when going through the dataset.
            In that case the epoch number must be provided in order for the model
            to continue the permutation across epochs. In that case, it is assumed
            that `num_samples = total_batch_size * num_updates_per_epoch`, with
            `total_batch_size` the overall batch size accounting for all gpus.
    """
    def __init__(self,
                 meta: tp.List[AudioMeta],
                 token_id_size: int,
                 position_size: int,
                 note_type: dict,
                 beatmap_sample_window: int,
                 minimum_note: float,
                 segment_duration: tp.Optional[float] = None,
                 shuffle: bool = True,
                 num_samples: int = 10_000,
                 sample_rate: int = 48_000,
                 channels: int = 2,
                 pad: bool = True,
                 sample_on_duration: bool = True,
                 sample_on_weight: bool = True,
                 min_segment_ratio: float = 0.5,
                 max_read_retry: int = 10,
                 return_info: bool = False,
                 min_audio_duration: tp.Optional[float] = None,
                 max_audio_duration: tp.Optional[float] = None,
                 shuffle_seed: int = 0,
                 load_wav: bool = True,
                 permutation_on_files: bool = False,
                 merge_text_p: float = 0,
                 drop_desc_p: float = 0., 
                 drop_other_p: float = 0.,
                 ):
        assert len(meta) > 0, "No audio meta provided to AudioDataset. Please check loading of audio meta."
        assert segment_duration is None or segment_duration > 0
        assert segment_duration is None or min_segment_ratio >= 0
        self.segment_duration = segment_duration
        self.min_segment_ratio = min_segment_ratio
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        if self.min_audio_duration is not None and self.max_audio_duration is not None:
            assert self.min_audio_duration <= self.max_audio_duration
        self.meta: tp.List[AudioMeta] = self._filter_duration(meta)
        assert len(self.meta)  # Fail fast if all data has been filtered.
        self.total_duration = sum(d.duration for d in self.meta)

        if segment_duration is None:
            num_samples = len(self.meta)
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.channels = channels
        self.pad = pad
        self.sample_on_weight = sample_on_weight
        self.sample_on_duration = sample_on_duration
        self.sampling_probabilities = self._get_sampling_probabilities()
        self.max_read_retry = max_read_retry
        self.return_info = return_info
        self.shuffle_seed = shuffle_seed
        self.current_epoch: tp.Optional[int] = None
        self.load_wav = load_wav
        self.token_id_size = token_id_size
        self.position_size = position_size
        self.note_type = note_type
        self.beatmap_sample_window = beatmap_sample_window
        self.minimum_note = minimum_note
        if not load_wav:
            assert segment_duration is not None
        self.permutation_on_files = permutation_on_files
        if permutation_on_files:
            assert not self.sample_on_duration
            assert not self.sample_on_weight
            assert self.shuffle

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return self.num_samples

    def _get_sampling_probabilities(self, normalized: bool = True):
        """Return the sampling probabilities for each file inside `self.meta`."""
        scores: tp.List[float] = []
        for file_meta in self.meta:
            score = 1.
            if self.sample_on_weight and file_meta.weight is not None:
                score *= file_meta.weight
            if self.sample_on_duration:
                score *= file_meta.duration
            scores.append(score)
        probabilities = torch.tensor(scores)
        if normalized:
            probabilities /= probabilities.sum()
        return probabilities

    @staticmethod
    @lru_cache(16)
    def _get_file_permutation(num_files: int, permutation_index: int, base_seed: int):
        # Used to keep the most recent files permutation in memory implicitely.
        # will work unless someone is using a lot of Datasets in parallel.
        rng = torch.Generator()
        rng.manual_seed(base_seed + permutation_index)
        return torch.randperm(num_files, generator=rng)

    def sample_file(self, index: int, rng: torch.Generator) -> AudioMeta:
        """Sample a given file from `self.meta`. Can be overridden in subclasses.
        This is only called if `segment_duration` is not None.

        You must use the provided random number generator `rng` for reproducibility.
        You can further make use of the index accessed.
        """
        if self.permutation_on_files:
            assert self.current_epoch is not None
            total_index = self.current_epoch * len(self) + index
            permutation_index = total_index // len(self.meta)
            relative_index = total_index % len(self.meta)
            permutation = AudioDataset._get_file_permutation(
                len(self.meta), permutation_index, self.shuffle_seed)
            file_index = permutation[relative_index]
            return self.meta[file_index]

        if not self.sample_on_weight and not self.sample_on_duration:
            file_index = int(torch.randint(len(self.sampling_probabilities), (1,), generator=rng).item())
        else:
            file_index = int(torch.multinomial(self.sampling_probabilities, 1, generator=rng).item())

        return self.meta[file_index]

    def _audio_read(self, path: str, seek_time: float = 0, duration: float = -1):
        # Override this method in subclass if needed.
        if self.load_wav:
            return audio_read(path, seek_time, duration, pad=False)
        else:
            assert self.segment_duration is not None
            n_frames = int(self.sample_rate * self.segment_duration)
            return torch.zeros(self.channels, n_frames), self.sample_rate
    
    def generate_y_beat_times(self, bpm: float, segment_duration_in_quaver: float, length:int, minimum_note = 0.25):
        def traditional_round(n):
            return int(n + 0.5) if n > 0 else int(n - 0.5)
        # [0, segment_duration_in_quaver*minimum_note] 范围内的四分音符，segment_duration_in_quaver*minimum_note是exlucsive
        segment_duration_in_quarter = traditional_round(segment_duration_in_quaver)
        note = list(range(segment_duration_in_quarter))
        note = [x * minimum_note * 60 / bpm for x in note]
        note = np.array(note)        
        time_interval = 60 / bpm * minimum_note
        y_beat_times = librosa.clicks(times=note, sr=self.sample_rate, click_duration=time_interval,length=length)
        return y_beat_times, len(note)
    
    def __getitem__(self, index: int) -> tp.Tuple[SegmentInfo, torch.Tensor, torch.Tensor]:
        # 抽取时间点，resample整个音频，原始音频对应被抽取的片段，对应beatmap的片段，还有对应beatmap片段tokenize后的tensor
        rng = torch.Generator()
        if self.shuffle:
            # We use index, plus extra randomness, either totally random if we don't know the epoch.
            # otherwise we make use of the epoch number and optional shuffle_seed.
            if self.current_epoch is None:
                rng.manual_seed(index + self.num_samples * random.randint(0, 2**24))
            else:
                rng.manual_seed(index + self.num_samples * (self.current_epoch + self.shuffle_seed))
        else:
            # We only use index
            rng.manual_seed(index)
        
        for retry in range(self.max_read_retry):
            # 随机抽取一个beatmap_file      
            file_meta = self.sample_file(index, rng)
            if file_meta.bpm <65 or file_meta.bpm> 280:
                continue
            duration_in_quaver = round(file_meta.duration / 60 * file_meta.bpm / self.minimum_note) # 音频长度用八分音符的数量来衡量
            segment_duration_in_quaver = round(self.segment_duration / 60 * file_meta.bpm /self.minimum_note)
            #选择抽取的时间点
            window = self.beatmap_sample_window
            duration_in_quaver_window = int(duration_in_quaver / window)
            segment_duration_window = int(segment_duration_in_quaver / window)
            max_seek = max(0, int(duration_in_quaver_window - segment_duration_window * self.min_segment_ratio))
            seek_time_in_quaver_window = torch.randint(0, max_seek + 1, (1,), generator=rng).item()  # +1 because randint upper bound is exclusive
            seek_time_in_quaver = seek_time_in_quaver_window * window
            seek_time_in_second = seek_time_in_quaver * self.minimum_note / file_meta.bpm * 60
            try:
                # 打开audio，使用encodec需要的sr resample整个audio
                error_path = file_meta.song_path
                origin_sample, sr = audio_read(file_meta.song_path, seek_time_in_second, self.segment_duration, pad=False)
                resample_sample = convert_audio(origin_sample, sr, self.sample_rate, self.channels)
                #pad resample 后的音频
                n_frames = resample_sample.shape[-1]
                target_frames = int(self.segment_duration * self.sample_rate)
                if self.pad:
                    resample_sample = F.pad(resample_sample, (0, target_frames - n_frames))
                # pad origin
                n_frames = origin_sample.shape[-1]
                target_frames = int(self.segment_duration * file_meta.sample_rate)
                if self.pad:
                    origin_sample = F.pad(origin_sample, (0, target_frames - n_frames))
                # 打开beatmap
                with open(file_meta.beatmap_file_path, 'r', encoding = 'utf-8') as f:
                    beatmap_file = json.load(f)
                # 打开beatmap
                error_path = file_meta.beatmap_file_path
                beatmap = Beatmap(minimum_note = self.minimum_note, token_id_size = self.token_id_size, position_size = self.position_size, **self.note_type)
                beatmap_file = beatmap.sample_beatmap_file(beatmap_file, seek_time_in_quaver,  seek_time_in_quaver + segment_duration_in_quaver)
                beatmap_token = beatmap.tokenize(beatmap_file, segment_duration_in_quaver)
                segment_info = SegmentInfo(file_meta, round(seek_time_in_quaver * self.minimum_note),segment_duration_in_quaver=segment_duration_in_quaver, n_frames=n_frames, total_frames=target_frames,
                                                sample_rate=self.sample_rate, channels=origin_sample.shape[0], origin_sample=origin_sample, beatmap_file=beatmap_file, beatmap_class=beatmap)
                y_beat_times, times = self.generate_y_beat_times(file_meta.bpm, segment_duration_in_quaver,resample_sample.shape[-1] )
                y_beat_times = torch.from_numpy(y_beat_times).unsqueeze(0).unsqueeze(0)

            except Exception as exc:
                logger.warning("Error opening file %s, seek time, %d,  %r", error_path, round(seek_time_in_quaver * self.minimum_note), exc)
                if retry == self.max_read_retry - 1:
                    raise
            else:
                break
        return segment_info, resample_sample, beatmap_token, y_beat_times

    def collater(self, samples):
        """The collater function has to be provided to the dataloader
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        segment_infos, resample_samples, beatmap_tokens, y_beat_times = zip(*samples)
        segment_infos = list(segment_infos)
        resample_samples = list(resample_samples)
        beatmap_tokens = list(beatmap_tokens)        
        y_beat_times = list(y_beat_times)
        new_resample_samples = []
        for y, r in zip(y_beat_times, resample_samples):
            combined = torch.cat((y, r.unsqueeze(0)), dim=0)
            new_resample_samples.append(combined)
        
        return segment_infos, new_resample_samples, beatmap_tokens

    def _filter_duration(self, meta: tp.List[AudioMeta]) -> tp.List[AudioMeta]:
        """Filters out_origin audio files with audio durations that will not allow to sample examples from them."""
        orig_len = len(meta)

        # Filter data that is too short.
        if self.min_audio_duration is not None:
            meta = [m for m in meta if m.duration >= self.min_audio_duration]

        # Filter data that is too long.
        if self.max_audio_duration is not None:
            meta = [m for m in meta if m.duration <= self.max_audio_duration]

        filtered_len = len(meta)
        removed_percentage = 100*(1-float(filtered_len)/orig_len)
        msg = 'Removed %.2f percent of the data because it was too short or too long.' % removed_percentage
        if removed_percentage < 10:
            logging.debug(msg)
        else:
            logging.warning(msg)
        return meta

    @classmethod
    def from_meta(cls, root: tp.Union[str, Path], **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_dir():
            if (root / 'data.jsonl').exists():
                root = root / 'data.jsonl'
            elif (root / 'data.jsonl.gz').exists():
                root = root / 'data.jsonl.gz'
            else:
                raise ValueError("Don't know where to read metadata from in the dir. "
                                 "Expecting either a data.jsonl or data.jsonl.gz file but none found.")
        meta = load_audio_meta(root)
        return cls(meta, **kwargs)

    @classmethod
    def from_path(cls, root: tp.Union[str, Path], minimal_meta: bool = True,
                  exts: tp.List[str] = DEFAULT_EXTS, **kwargs):
        """Instantiate AudioDataset from a path containing (possibly nested) audio files.

        Args:
            root (str or Path): Path to root folder containing audio files.
            minimal_meta (bool): Whether to only load minimal metadata or not.
            exts (list of str): Extensions for audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        if root.is_file():
            meta = load_audio_meta(root, resolve=True)
        else:
            meta = find_audio_files(root, exts, minimal=minimal_meta, resolve=True)
        return cls(meta, **kwargs)

url = "http://localhost:8001/read"
def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='audio_dataset',
        description='Generate .jsonl files by scanning a folder.')
    parser.add_argument('root', help='Root folder with all the audio files')
    parser.add_argument('out_originput_meta_file',
                        help='out_originput file to store the metadata, ')
    parser.add_argument('--complete',
                        action='store_false', dest='minimal', default=True,
                        help='Retrieve all metadata, even the one that are expansive '
                             'to compute (e.g. normalization).')
    parser.add_argument('--resolve',
                        action='store_true', default=False,
                        help='Resolve the paths to be absolute and with no symlinks.')
    parser.add_argument('--workers',
                        default=10, type=int,
                        help='Number of workers.')
    parser.add_argument('complex_beat_number', default=0.125, type=float,
                        help='out_originput file to store the metadata, ')
    parser.add_argument('--write_parse_switch', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    # use deno api to parse beatmap
    request_data = {
        "directory": args.root, 
        "complex_beat_number": args.complex_beat_number,
        "write_parse_switch": args.write_parse_switch
    }
    response = requests.get(url, params=request_data)
    if response.status_code == 200:
        data = response.json()
        out_originput_meta = data.pop('output_meta')  # 提取并删除 out_originput_meta
        print("request finished, summary:")
        print(data)

    else:
        print(f"Error: {response.status_code}, {response.text}")
        return
    
    meta = find_audio_files(out_originput_meta, DEFAULT_EXTS, progress=True,
                            resolve=args.resolve, minimal=args.minimal, workers=args.workers)
    save_audio_meta(args.out_originput_meta_file, meta)


if __name__ == '__main__':
    main()
