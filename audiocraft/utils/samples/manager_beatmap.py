# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
API that can manage the storage and retrieval of generated samples produced by experiments.

It offers the following benefits:
* Samples are stored in a consistent way across epoch
* Metadata about the samples can be stored and retrieved
* Can retrieve audio
* Identifiers are reliable and deterministic for prompted and conditioned samples
* Can request the samples for multiple XPs, grouped by sample identifier
* For no-input samples (not prompt and no conditions), samples across XPs are matched
  by sorting their identifiers
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import re
import typing as tp
import unicodedata
import uuid

import dora
import torch

import requests
import zipfile
import os

import shutil

from ...data.audio import audio_read, audio_write


logger = logging.getLogger(__name__)


@dataclass
class ReferenceSample:
    id: str
    path: str
    duration: float


@dataclass
class Sample:
    id: str
    path: str
    epoch: int
    duration: float
    conditioning: tp.Optional[tp.Dict[str, tp.Any]]
    prompt: tp.Optional[ReferenceSample]
    reference: tp.Optional[ReferenceSample]
    generation_args: tp.Optional[tp.Dict[str, tp.Any]]

    def __hash__(self):
        return hash(self.id)

    def audio(self) -> tp.Tuple[torch.Tensor, int]:
        return audio_read(self.path)

    def audio_prompt(self) -> tp.Optional[tp.Tuple[torch.Tensor, int]]:
        return audio_read(self.prompt.path) if self.prompt is not None else None

    def audio_reference(self) -> tp.Optional[tp.Tuple[torch.Tensor, int]]:
        return audio_read(self.reference.path) if self.reference is not None else None


class SampleManager:
    """Audio samples IO handling within a given dora xp.

    The sample manager handles the dumping and loading logic for generated and
    references samples across epochs for a given xp, providing a simple API to
    store, retrieve and compare audio samples.

    Args:
        xp (dora.XP): Dora experiment object. The XP contains information on the XP folder
            where all outputs are stored and the configuration of the experiment,
            which is useful to retrieve audio-related parameters.
        map_reference_to_sample_id (bool): Whether to use the sample_id for all reference samples
            instead of generating a dedicated hash id. This is useful to allow easier comparison
            with ground truth sample from the files directly without having to read the JSON metadata
            to do the mapping (at the cost of potentially dumping duplicate prompts/references
            depending on the task).
    """
    def __init__(self, xp: dora.XP, map_reference_to_sample_id: bool = False):
        self.xp = xp
        self.base_folder: Path = xp.folder / xp.cfg.generate.path
        self.reference_folder = self.base_folder / 'reference'
        self.map_reference_to_sample_id = map_reference_to_sample_id
        self.samples: tp.List[Sample] = []
        # self._load_samples()

    @property
    def latest_epoch(self):
        """Latest epoch across all samples."""
        return max(self.samples, key=lambda x: x.epoch).epoch if self.samples else 0

    def _load_samples(self):
        """Scan the sample folder and load existing samples."""
        jsons = self.base_folder.glob('**/*.json')
        with ThreadPoolExecutor(6) as pool:
            self.samples = list(pool.map(self._load_sample, jsons))

    @staticmethod
    @lru_cache(2**26)
    def _load_sample(json_file: Path) -> Sample:
        with open(json_file, 'r') as f:
            data: tp.Dict[str, tp.Any] = json.load(f)
        # fetch prompt data
        prompt_data = data.get('prompt')
        prompt = ReferenceSample(id=prompt_data['id'], path=prompt_data['path'],
                                 duration=prompt_data['duration']) if prompt_data else None
        # fetch reference data
        reference_data = data.get('reference')
        reference = ReferenceSample(id=reference_data['id'], path=reference_data['path'],
                                    duration=reference_data['duration']) if reference_data else None
        # build sample object
        return Sample(id=data['id'], path=data['path'], epoch=data['epoch'], duration=data['duration'],
                      prompt=prompt, conditioning=data.get('conditioning'), reference=reference,
                      generation_args=data.get('generation_args'))

    def _init_hash(self):
        return hashlib.sha1()

    def _get_tensor_id(self, tensor: torch.Tensor) -> str:
        hash_id = self._init_hash()
        hash_id.update(tensor.numpy().data)
        return hash_id.hexdigest()

    def _get_sample_id(self, index: int, prompt_wav: tp.Optional[torch.Tensor],
                       conditions: tp.Optional[tp.Dict[str, str]]) -> str:
        """Computes an id for a sample given its input data.
        This id is deterministic if prompt and/or conditions are provided by using a sha1 hash on the input.
        Otherwise, a random id of the form "noinput_{uuid4().hex}" is returned.

        Args:
            index (int): Batch index, Helpful to differentiate samples from the same batch.
            prompt_wav (torch.Tensor): Prompt used during generation.
            conditions (dict[str, str]): Conditioning used during generation.
        """
        # For totally unconditioned generations we will just use a random UUID.
        # The function get_samples_for_xps will do a simple ordered match with a custom key.
        if prompt_wav is None and not conditions:
            return f"noinput_{uuid.uuid4().hex}"

        # Human readable portion
        hr_label = ""
        # Create a deterministic id using hashing
        hash_id = self._init_hash()
        hash_id.update(f"{index}".encode())
        if prompt_wav is not None:
            hash_id.update(prompt_wav.numpy().data)
            hr_label += "_prompted"
        else:
            hr_label += "_unprompted"
        if conditions:
            encoded_json = json.dumps(conditions, sort_keys=True).encode()
            hash_id.update(encoded_json)
            cond_str = "-".join([f"{key}={slugify(value)}"
                                 for key, value in sorted(conditions.items())])
            cond_str = cond_str[:100]  # some raw text might be too long to be a valid filename
            cond_str = cond_str if len(cond_str) > 0 else "unconditioned"
            hr_label += f"_{cond_str}"
        else:
            hr_label += "_unconditioned"

        return hash_id.hexdigest() + hr_label

    def _store_audio(self, wav: torch.Tensor, stem_path: Path, overwrite: bool = False) -> Path:
        """Stores the audio with the given stem path using the XP's configuration.

        Args:
            wav (torch.Tensor): Audio to store.
            stem_path (Path): Path in sample output directory with file stem to use.
            overwrite (bool): When False (default), skips storing an existing audio file.
        Returns:
            Path: The path at which the audio is stored.
        """
        existing_paths = [
            path for path in stem_path.parent.glob(stem_path.stem + '.*')
            if path.suffix != '.json'
        ]
        exists = len(existing_paths) > 0
        if exists and overwrite:
            logger.warning(f"Overwriting existing audio file with stem path {stem_path}")
        elif exists:
            return existing_paths[0]

        audio_path = audio_write(stem_path, wav, **self.xp.cfg.generate.audio)
        return audio_path     
    
    def zip_folder(self, source_folder, zip_file):
        # 确保zip文件的父目录存在
        output_folder = os.path.dirname(zip_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 创建zip文件
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 遍历源文件夹及其子文件夹
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    # 计算相对路径并写入zip文件
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, source_folder)
                    zipf.write(file_path, relative_path)

    def add_sample(self, save_directory, audio, meta, beatmap, save_directory_zip, sample_id, write_info_switch):
        # save audio
        if audio is not None:
            song_name = 'song'
            audio_write(save_directory / song_name, audio, meta.sample_rate, format="ogg")
        # save beatmap json file
        with open((save_directory / meta.difficulty).with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(beatmap, f, ensure_ascii=False, indent=2)
        # use deno api to convert json to dat, and get info
        request_data = {
            "beatmap_file_path": (save_directory / meta.difficulty).with_suffix('.json'),
            "difficulty": meta.difficulty,
            "save_directory": save_directory,
            "beatmap_info_path": meta.beatmap_info_path,
            "beatmap_name": sample_id,
            "difficulty_version": self.xp.cfg.parser_pipeline.generate_difficulty_version,
            'info_version': self.xp.cfg.parser_pipeline.generate_info_version,
            "write_info_switch": write_info_switch,
        }
        response = requests.get(self.xp.cfg.parser_pipeline.generate_url, params=request_data)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
        # pack everything into zip
        self.zip_folder(save_directory, save_directory_zip.with_suffix('.zip'))

    def add_samples(self, sample_ids: list, audios: list, metas: list, epoch: int,
                    ground_truth_beatmaps: list,
                    gen_beatmaps: list,
                    conditioning: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
                    generation_args: tp.Optional[tp.Dict[str, tp.Any]] = None) -> tp.List[Sample]:
        """Adds a batch of samples.
        The samples are stored in the XP's sample output directory, under a corresponding
        epoch folder. Each sample is assigned an id which is computed using the input data and their batch index.
        In addition to the sample itself, a json file containing associated metadata is stored next to it.

        Args:
            audio (list): Reference audio where prompts were extracted from.
                Tensor of shape [batch_size, channels, shape].
            epoch (int): Current training epoch.
            ground_truth_beatmap (list): list of beatmap file. list
            gen_beatmap (list): list of generated beatmap file. list
            conditioning (list of dict[str, str], optional): List of conditions used during generation,
                one per sample in the batch.
            generation_args (dict[str, Any], optional): Dictionary of other arguments used during generation.
        """
        
        for index, (sample_id, audio, meta, ground_truth_beatmap, gen_beatmap) in enumerate(zip(sample_ids, audios, metas, ground_truth_beatmaps, gen_beatmaps)):
            reference_path = self.base_folder / 'reference' / sample_id
            generated_path = self.base_folder / 'generated' / f"{epoch}-{index}_{sample_id}"
            reference_path_zip = self.base_folder / 'reference_zip' / sample_id
            generated_path_zip = self.base_folder / 'generated_zip' / f"{epoch}-{index}_{sample_id}"
            # generate every for reference
            if ground_truth_beatmap is not None:
                self.add_sample(reference_path, audio, meta, ground_truth_beatmap, reference_path_zip, sample_id, True)
            # copy song.ogg from reference
            song_name = 'song.ogg'
            os.makedirs(generated_path, exist_ok=True)
            shutil.copy(reference_path / song_name, generated_path / song_name)
            # copy Info.dat from reference
            info_name = 'Info.dat'
            shutil.copy(reference_path / info_name, generated_path / info_name)
            # after copy these two, only need to generate beatmap dat file
            self.add_sample(generated_path, None, meta, gen_beatmap, generated_path_zip, sample_id, False)

    def get_samples(self, epoch: int = -1, max_epoch: int = -1, exclude_prompted: bool = False,
                    exclude_unprompted: bool = False, exclude_conditioned: bool = False,
                    exclude_unconditioned: bool = False) -> tp.Set[Sample]:
        """Returns a set of samples for this XP. Optionally, you can filter which samples to obtain.
        Please note that existing samples are loaded during the manager's initialization, and added samples through this
        manager are also tracked. Any other external changes are not tracked automatically, so creating a new manager
        is the only way detect them.

        Args:
            epoch (int): If provided, only return samples corresponding to this epoch.
            max_epoch (int): If provided, only return samples corresponding to the latest epoch that is <= max_epoch.
            exclude_prompted (bool): If True, does not include samples that used a prompt.
            exclude_unprompted (bool): If True, does not include samples that did not use a prompt.
            exclude_conditioned (bool): If True, excludes samples that used conditioning.
            exclude_unconditioned (bool): If True, excludes samples that did not use conditioning.
        Returns:
            Samples (set of Sample): The retrieved samples matching the provided filters.
        """
        if max_epoch >= 0:
            samples_epoch = max(sample.epoch for sample in self.samples if sample.epoch <= max_epoch)
        else:
            samples_epoch = self.latest_epoch if epoch < 0 else epoch
        samples = {
            sample
            for sample in self.samples
            if (
                (sample.epoch == samples_epoch) and
                (not exclude_prompted or sample.prompt is None) and
                (not exclude_unprompted or sample.prompt is not None) and
                (not exclude_conditioned or not sample.conditioning) and
                (not exclude_unconditioned or sample.conditioning)
            )
        }
        return samples


def slugify(value: tp.Any, allow_unicode: bool = False):
    """Process string for safer file naming.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def _match_stable_samples(samples_per_xp: tp.List[tp.Set[Sample]]) -> tp.Dict[str, tp.List[Sample]]:
    # Create a dictionary of stable id -> sample per XP
    stable_samples_per_xp = [{
        sample.id: sample for sample in samples
        if sample.prompt is not None or sample.conditioning
    } for samples in samples_per_xp]
    # Set of all stable ids
    stable_ids = {id for samples in stable_samples_per_xp for id in samples.keys()}
    # Dictionary of stable id -> list of samples. If an XP does not have it, assign None
    stable_samples = {id: [xp.get(id) for xp in stable_samples_per_xp] for id in stable_ids}
    # Filter out ids that contain None values (we only want matched samples after all)
    # cast is necessary to avoid mypy linter errors.
    return {id: tp.cast(tp.List[Sample], samples) for id, samples in stable_samples.items() if None not in samples}


def _match_unstable_samples(samples_per_xp: tp.List[tp.Set[Sample]]) -> tp.Dict[str, tp.List[Sample]]:
    # For unstable ids, we use a sorted list since we'll match them in order
    unstable_samples_per_xp = [[
        sample for sample in sorted(samples, key=lambda x: x.id)
        if sample.prompt is None and not sample.conditioning
    ] for samples in samples_per_xp]
    # Trim samples per xp so all samples can have a match
    min_len = min([len(samples) for samples in unstable_samples_per_xp])
    unstable_samples_per_xp = [samples[:min_len] for samples in unstable_samples_per_xp]
    # Dictionary of index -> list of matched samples
    return {
        f'noinput_{i}': [samples[i] for samples in unstable_samples_per_xp] for i in range(min_len)
    }


def get_samples_for_xps(xps: tp.List[dora.XP], **kwargs) -> tp.Dict[str, tp.List[Sample]]:
    """Gets a dictionary of matched samples across the given XPs.
    Each dictionary entry maps a sample id to a list of samples for that id. The number of samples per id
    will always match the number of XPs provided and will correspond to each XP in the same order given.
    In other words, only samples that can be match across all provided XPs will be returned
    in order to satisfy this rule.

    There are two types of ids that can be returned: stable and unstable.
    * Stable IDs are deterministic ids that were computed by the SampleManager given a sample's inputs
      (prompts/conditioning). This is why we can match them across XPs.
    * Unstable IDs are of the form "noinput_{idx}" and are generated on-the-fly, in order to map samples
      that used non-deterministic, random ids. This is the case for samples that did not use prompts or
      conditioning for their generation. This function will sort these samples by their id and match them
      by their index.

    Args:
        xps: a list of XPs to match samples from.
        start_epoch (int): If provided, only return samples corresponding to this epoch or newer.
        end_epoch (int): If provided, only return samples corresponding to this epoch or older.
        exclude_prompted (bool): If True, does not include samples that used a prompt.
        exclude_unprompted (bool): If True, does not include samples that did not use a prompt.
        exclude_conditioned (bool): If True, excludes samples that used conditioning.
        exclude_unconditioned (bool): If True, excludes samples that did not use conditioning.
    """
    managers = [SampleManager(xp) for xp in xps]
    samples_per_xp = [manager.get_samples(**kwargs) for manager in managers]
    stable_samples = _match_stable_samples(samples_per_xp)
    unstable_samples = _match_unstable_samples(samples_per_xp)
    return dict(stable_samples, **unstable_samples)
