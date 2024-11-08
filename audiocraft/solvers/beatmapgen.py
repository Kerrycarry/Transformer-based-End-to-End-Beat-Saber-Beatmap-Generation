# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import time
import typing as tp
import warnings

import flashy
import math
import omegaconf
import torch
from torch.nn import functional as F

from . import base, builders
from .compression import CompressionSolver
from .. import metrics as eval_metrics
from .. import models
from ..data.audio_dataset_beatmap import AudioDataset, SegmentInfo
from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from ..data.audio_utils import normalize_audio
from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition
from ..utils.cache import CachedBatchWriter, CachedBatchLoader
from ..utils.samples.manager_beatmap import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash


class BeatmapGenSolver(base.StandardSolver):
    """Solver for MusicGen training task.

    Used in: https://arxiv.org/abs/2306.05284
    """
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.BEATMAP

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.count =0
        # easier access to sampling parameters
        self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
        }
        self._best_metric_name: tp.Optional[str] = 'ce'

        self._cached_batch_writer = None
        self._cached_batch_loader = None
        if cfg.cache.path:
            if cfg.cache.write:
                self._cached_batch_writer = CachedBatchWriter(Path(cfg.cache.path))
                if self.cfg.cache.write_num_shards:
                    self.logger.warning("Multiple shard cache, best_metric_name will be set to None.")
                    self._best_metric_name = None
            else:
                self._cached_batch_loader = CachedBatchLoader(
                    Path(cfg.cache.path), cfg.dataset.batch_size, cfg.dataset.num_workers,
                    min_length=self.cfg.optim.updates_per_epoch or 1)
                self.dataloaders['original_train'] = self.dataloaders['train']
                self.dataloaders['train'] = self._cached_batch_loader  # type: ignore

    @staticmethod
    def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
                                 device: tp.Optional[str] = None, autocast: bool = True,
                                 batch_size: tp.Optional[int] = None,
                                 override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                                 **kwargs):
        """Mostly a convenience function around magma.train.get_solver_from_sig,
        populating all the proper param, deactivating EMA, FSDP, loading the best state,
        basically all you need to get a solver ready to "play" with in single GPU mode
        and with minimal memory overhead.

        Args:
            sig (str): signature to load.
            dtype (str or None): potential dtype, as a string, i.e. 'float16'.
            device (str or None): potential device, as a string, i.e. 'cuda'.
            override_cfg (dict or omegaconf.DictConfig or None): potential device, as a string, i.e. 'cuda'.
        """
        from audiocraft import train
        our_override_cfg: tp.Dict[str, tp.Any] = {'optim': {'ema': {'use': False}}}
        our_override_cfg['autocast'] = autocast
        if dtype is not None:
            our_override_cfg['dtype'] = dtype
        if device is not None:
            our_override_cfg['device'] = device
        if batch_size is not None:
            our_override_cfg['dataset'] = {'batch_size': batch_size}
        if override_cfg is None:
            override_cfg = {}
        override_cfg = omegaconf.OmegaConf.merge(
            omegaconf.DictConfig(override_cfg), omegaconf.DictConfig(our_override_cfg))  # type: ignore
        solver = train.get_solver_from_sig(
            sig, override_cfg=override_cfg,
            load_best=True, disable_fsdp=True,
            ignore_state_keys=['optimizer', 'ema'], **kwargs)
        solver.model.eval()
        return solver

    def get_formatter(self, stage_name: str) -> flashy.Formatter:
        return flashy.Formatter({
            'lr': '.2E',
            'ce': '.3f',
            'ppl': '.3f',
            'grad_norm': '.3E',
        }, exclude_keys=['ce_q*', 'ppl_q*'])

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        return self._best_metric_name

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg, self.cfg.compression_model_checkpoint, device=self.device)
        assert self.compression_model.sample_rate == self.cfg.sample_rate, (
            f"Compression model sample rate is {self.compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
            )
        # ensure we have matching configuration between LM and compression model
        assert self.cfg.transformer_lm.card == self.compression_model.cardinality, (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
            f"compression model cardinality is {self.compression_model.cardinality}"
        )
        assert self.cfg.transformer_lm.n_q == self.compression_model.num_codebooks, (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {self.compression_model.num_codebooks}"
        )
        self.logger.info("Compression model has %d codebooks with %d cardinality, and a framerate of %d",
                         self.compression_model.num_codebooks, self.compression_model.cardinality,
                         self.compression_model.frame_rate)
        # instantiate LM model
        self.model: models.BeatmapLMModel = models.builders.get_beatmapgen_lm_model(self.cfg).to(self.device)
        if self.cfg.fsdp.use:
            assert not self.cfg.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        self.register_ema('model')
        # initialize optimization
        self.optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg.optim)
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        self.register_stateful('model', 'optimizer', 'lr_scheduler')
        self.register_best_state('model')
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg.autocast_dtype]
        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
        if self.cfg.fsdp.use:
            need_scaler = self.cfg.fsdp.param_dtype == 'float16'
        else:
            need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            if self.cfg.fsdp.use:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                self.scaler = ShardedGradScaler()  # type: ignore
            else:
                self.scaler = torch.cuda.amp.GradScaler()
            self.register_stateful('scaler')
        
        with open('model_architecture.txt', 'w') as f:
            f.write(str(self.model))
        with open('model_architecture2.txt', 'w') as f:
            for n, m in self.model.named_modules():
                f.write(f'{n}: {type(m).__name__}\n')
        #transfer learning
        if self.cfg.transformer_lm.lora_kwargs.use_lora:
            trainable = ['lora_in_proj_a','lora_in_proj_b', 'mask_token_embedding', 'lora_a', 'lora_b']
            # 冻结模型中的所有参数
            for name,param in self.model.named_parameters():
                if name.split('.')[-1] not in trainable:  # 非LOra部分不计算梯度
                    param.requires_grad=False
                else:
                    param.requires_grad=True
        # Iterate over each module and unfreeze its parameters
        modules_to_unfreeze = [self.model.difficulty_emb, self.model.linear_transfer, self.model.transfer_lm, self.model.linear_out]
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True


    def build_dataloaders(self) -> None:
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)

    def show(self) -> None:
        """Show the compression model and LM model."""
        self.logger.info("Compression model:")
        self.log_model_summary(self.compression_model)
        self.logger.info("LM model:")
        self.log_model_summary(self.model)

    def load_state_dict(self, state: dict) -> None:
        if 'condition_provider' in state:
            model_state = state['model']
            condition_provider_state = state.pop('condition_provider')
            prefix = 'condition_provider.'
            for key, value in condition_provider_state.items():
                key = prefix + key
                assert key not in model_state
                model_state[key] = value
        if 'compression_model' in state:
            # We used to store the `compression_model` state in the checkpoint, however
            # this is in general not needed, as the compression model should always be readable
            # from the original `cfg.compression_model_checkpoint` location.
            compression_model_state = state.pop('compression_model')
            before_hash = model_hash(self.compression_model)
            self.compression_model.load_state_dict(compression_model_state)
            after_hash = model_hash(self.compression_model)
            if before_hash != after_hash:
                raise RuntimeError(
                    "The compression model state inside the checkpoint is different"
                    " from the one obtained from compression_model_checkpoint..."
                    "We do not support altering the compression model inside the LM "
                    "checkpoint as parts of the code, in particular for running eval post-training "
                    "will use the compression_model_checkpoint as the source of truth.")

        super().load_state_dict(state)

    def load_from_pretrained(self, name: str):
        # TODO: support native HF versions of MusicGen.
        lm_pkg = models.loaders.load_lm_model_ckpt(name)
        state: dict = {
            'best_state': {
                'model': lm_pkg['best_state'],
            },
        }
        return state

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, note_code_maps: list,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, S, P, card].
            targets (torch.Tensor): Target codes, of shape [B, S, P].
            
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, S, P = targets.shape
        ce = torch.zeros([], device=targets.device)
        for logit, target, note_code_map in zip(logits, targets, note_code_maps):
            target = target[note_code_map]
            assert logit.squeeze(0).shape[:-1] == target.view(-1).shape
            
            logits_k = logit.contiguous().view(-1,logit.size(-1))
            targets_k = target.view(-1)
            ce += F.cross_entropy(logits_k, targets_k)


        return ce
        # note_mask = (targets == self.model.outputLM.token_id_size).float()  # 0 for notes, 1 for rests
        # rest_logits = logits[..., self.model.outputLM.token_id_size] 
        # rhythm_loss = F.binary_cross_entropy_with_logits(rest_logits.view(-1), note_mask.view(-1))

        # return ce, rhythm_loss
    
    def tokenize_difficulty(self, segment_infos):
        difficulty_map = {'Easy': 0, 'Normal': 1, 'Hard': 2, 'Expert': 3, 'ExpertPlus': 4}
        difficulty = [difficulty_map[segment_info.meta.difficulty] for segment_info in segment_infos]
        difficulty = torch.tensor(difficulty, dtype=torch.int64).to(self.device)
        return difficulty
    
    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[tp.List[SegmentInfo], tp.List[torch.Tensor], torch.Tensor],
        check_synchronization_points: bool = False
    ) -> tp.Tuple[torch.Tensor, list, list, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        
        if self._cached_batch_loader is None or self.current_stage != "train":
            segment_infos, wav_resampleds, beatmap_tokens = batch
            wav_resampleds = wav_resampleds.to(self.device)
            audio_tokens = None
            assert wav_resampleds.size(0) == len(segment_infos) == beatmap_tokens.size(0), (
                f"Mismatch between number of items in audio batch ({wav_resampleds.size(0)})",
                f" and in metadata ({len(segment_infos)})"
            )
        else:
            # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
            segment_infos, = batch  # type: ignore
            assert all([isinstance(info, SegmentInfo) for info in segment_infos])
            assert all([info.audio_token is not None for info in segment_infos])  # type: ignore
            assert all([info.beatmap_token is not None for info in segment_infos])  # type: ignore
            audio_tokens = torch.stack([info.audio_token for info in segment_infos]).to(self.device)  # type: ignore
            beatmap_tokens = torch.stack([info.beatmap_token for info in segment_infos]).to(self.device)  # type: ignore
            audio_tokens = audio_tokens.long()
            beatmap_tokens = beatmap_tokens.long()
        
        sample_id_seek_time = [f"{segment_info.meta.id}_{segment_info.seek_time}" for segment_info in segment_infos]
        sample_id = [f"{segment_info.meta.id}" for segment_info in segment_infos]
        self.log_sample_usage("sample_id_seek_time.json", sample_id_seek_time)
        self.log_sample_usage("sample_id.json", sample_id)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if audio_tokens is None:
            with torch.no_grad():
                audio_tokens, scale = self.compression_model.encode(wav_resampleds)
                assert scale is None, "Scaled compression model not supported with LM."
        
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self._cached_batch_writer is not None and self.current_stage == 'train':
            assert self._cached_batch_loader is None
            assert audio_tokens is not None
            for segment_info, audio_token, beatmap_token in zip(segment_infos, audio_tokens, beatmap_tokens):
                assert isinstance(segment_info, SegmentInfo)
                assert audio_token.max() < 2**15, audio_token.max().item()
                assert beatmap_token.max() < 2**15, beatmap_token.max().item()
                segment_info.audio_token = audio_token.short().cpu()
                segment_info.beatmap_token = beatmap_token.short().cpu()
                
            self._cached_batch_writer.save(segment_infos)
        
        # get difficulty token
        difficulty = self.tokenize_difficulty(segment_infos)
        note_code_maps = [segment_info.note_code_map for segment_info in segment_infos]
        return audio_tokens, beatmap_tokens, difficulty, note_code_maps

    def log_sample_usage(self, filepath, sample_id):
        import json
        import os

        if not os.path.exists(filepath):
            data = {}
        else:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
        for id in sample_id:
            if id in data:
                data[id] = data[id]+1  # 修改已存在的键值
            else:
                data[id] = 1  # 添加新的键值

        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def run_step(self, idx: int, batch: tp.Tuple[tp.List[SegmentInfo], tp.List[torch.Tensor], torch.Tensor], metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        audio_tokens, beatmap_tokens, difficulty, note_code_maps = self._prepare_tokens_and_attributes(
            batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')
        # Debug: Add assertions or print to check the target values
        assert (beatmap_tokens >= 0).all(), "beatmap_tokens contains negative values!"
        assert (beatmap_tokens <= self.model.token_id_size).all(), f"beatmap_tokens contains invalid class indices! Max target: {beatmap_tokens.max()}"
        with self.autocast:
            logits = self.model.compute_predictions(audio_tokens, beatmap_tokens, difficulty, note_code_maps)  # type: ignore # [B, S, P, card]
            # ce, rhythm_loss = self._compute_cross_entropy(logits, beatmap_tokens)
            # loss = ce + rhythm_loss
            ce = self._compute_cross_entropy(logits, beatmap_tokens, note_code_maps)
            loss = ce
        self.deadlock_detect.update('loss')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('default')

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update('scale')
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                # this should always be slower but can be useful
                # for weird use cases like multiple backwards.
                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update('backward')

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update('optim')
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics['grad_scale'] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics['ce'] = ce
        metrics['ppl'] = torch.exp(ce)
        # metrics['rhythm_loss'] = rhythm_loss
        
        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[tp.List[SegmentInfo], tp.List[torch.Tensor], torch.Tensor],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_prompt: bool = False,
                          **generation_params) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
            remove_prompt (bool, optional): Whether to remove the prompt from the generated audio.
            generation_params: Additional generation parameters.
        Returns:
            gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
                and the prompt along with additional information.
        """
        bench_start = time.time()
        segment_infos, wav_resampleds, beatmap_tokens = batch
        audio_tokens = None
        assert wav_resampleds.size(0) == len(segment_infos) == beatmap_tokens.size(0), (
            f"Mismatch between number of items in audio batch ({wav_resampleds.size(0)})",
            f" and in metadata ({len(segment_infos)})"
        )
        
        with torch.no_grad():
            audio_tokens, scale = self.compression_model.encode(wav_resampleds.to(self.device))
            assert scale is None, "Scaled compression model not supported with LM."
        difficulty = self.tokenize_difficulty(segment_infos)
        note_code_maps = [segment_info.note_code_map for segment_info in segment_infos]
        with self.autocast:
            gen_beatmap_tokens = self.model.generate(
                audio_tokens, difficulty, note_code_maps, max_gen_len=self.model.outputLM.position_size,
                **self.generation_params)

        # generate audio from tokens
        for gen_beatmap_token in gen_beatmap_tokens:
            assert gen_beatmap_token.dim() == 3
        
        ref_audio = [segment_info.wav_slice for segment_info in segment_infos]
        ref_beatmap_file = [segment_info.beatmap_file for segment_info in segment_infos]
        gen_beatmap_file = [segment_info.beatmap_class.detokenize(gen_beatmap_token.squeeze(0)) for gen_beatmap_token, segment_info in zip(gen_beatmap_tokens, segment_infos) ]
        sample_id = [f"{segment_info.meta.id}_{segment_info.seek_time}" for segment_info in segment_infos]
        meta = [segment_info.meta for segment_info in segment_infos]
        bench_end = time.time()
        # 测试对齐
        beatmap_alignment_result = []
        for segment_info, beatmap_token in zip(segment_infos, beatmap_tokens):
            beatmap_token = beatmap_token[segment_info.note_code_map]
            reconstructed_beatmap_file = segment_info.beatmap_class.detokenize(beatmap_token)
            result = segment_info.beatmap_class.check_difference(reconstructed_beatmap_file, segment_info.beatmap_file, note_types = ['colorNotes'])
            beatmap_alignment_result.append(result)
        
        reconstructed_audios = self.compression_model.decode(audio_tokens, None)

        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': ref_audio,
            'ref_beatmap_file': ref_beatmap_file,
            'gen_beatmap_file': gen_beatmap_file,
            'sample_id': sample_id,
            'meta': meta,
            'beatmap_alignment_result': beatmap_alignment_result,
            'reconstructed_audio': reconstructed_audios
        }   
        return gen_outputs

    def generate_audio(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f'{self.current_stage}'
        sample_manager = SampleManager(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration / 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)"
        )

        def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if cond_key not in self.model.condition_provider.conditioners.keys():
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = cond_val.text  # only support text at inference for now
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            # audio, meta = batch
            # metadata for sample manager
            # hydrated_conditions = get_hydrated_conditions(meta)
            hydrated_conditions = []
            sample_generation_params = {
                **{f'classifier_free_guidance_{k}': v for k, v in self.cfg.classifier_free_guidance.items()},
                **self.generation_params
            }
            if self.cfg.generate.lm.unprompted_samples:
                if self.cfg.generate.lm.gen_gt_samples:
                    # get the ground truth instead of generation
                    self.logger.warn(
                        "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true")
                    rtf = 1.
                else:
                    gen_unprompted_outputs = self.run_generate_step(
                        batch, gen_duration=target_duration, prompt_duration=None,
                        **self.generation_params)
                    rtf = gen_unprompted_outputs['rtf']
                sample_manager.add_samples(
                    gen_unprompted_outputs['sample_id'],
                    gen_unprompted_outputs['ref_audio'],
                    gen_unprompted_outputs['meta'], self.epoch,
                    gen_unprompted_outputs['ref_beatmap_file'], gen_unprompted_outputs['gen_beatmap_file'],
                    gen_unprompted_outputs['beatmap_alignment_result'], gen_unprompted_outputs['reconstructed_audio'], 
                      hydrated_conditions, generation_args=sample_generation_params)            

            metrics['rtf'] = rtf
            metrics = average(metrics)

        flashy.distrib.barrier()
        return metrics

    def generate(self) -> dict:
        """Generate stage."""
        self.model.eval()
        with torch.no_grad():
            return self.generate_audio()

    def run_epoch(self):
        if self.cfg.cache.write:
            if ((self.epoch - 1) % self.cfg.cache.write_num_shards) != self.cfg.cache.write_shard:
                return
        super().run_epoch()

    def train(self):
        """Train stage.
        """
        if self._cached_batch_writer is not None:
            self._cached_batch_writer.start_epoch(self.epoch)
        if self._cached_batch_loader is None:
            dataset = get_dataset_from_loader(self.dataloaders['train'])
            assert isinstance(dataset, AudioDataset)
            dataset.current_epoch = self.epoch
        else:
            self._cached_batch_loader.start_epoch(self.epoch)
        return super().train()

    def evaluate_audio_generation(self) -> dict:
        """Evaluate audio generation with off-the-shelf metrics."""
        evaluate_stage_name = f'{self.current_stage}_generation'
        # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
        fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
        kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
        text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
        chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
        should_run_eval = False
        eval_chroma_wavs: tp.Optional[torch.Tensor] = None
        if self.cfg.evaluate.metrics.fad:
            fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.kld:
            kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.text_consistency:
            text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.chroma_cosine:
            chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
            # if we have predefind wavs for chroma we should purge them for computing the cosine metric
            has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
                                          self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
            if has_predefined_eval_chromas:
                warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
                                       'Resetting eval chromas to None for evaluation.')
                eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
                self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
            should_run_eval = True

        def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
            audio_tokens, scale = self.compression_model.encode(audio.to(self.device))
            compressed_audio = self.compression_model.decode(audio_tokens, scale)
            return compressed_audio[..., :audio.shape[-1]]

        metrics: dict = {}
        if should_run_eval:
            loader = self.dataloaders['evaluate']
            updates = len(loader)
            lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
            average = flashy.averager()
            dataset = get_dataset_from_loader(loader)
            assert isinstance(dataset, AudioDataset)
            self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")

            for idx, batch in enumerate(lp):
                audio, meta = batch
                assert all([self.cfg.sample_rate == m.sample_rate for m in meta])

                target_duration = audio.shape[-1] / self.cfg.sample_rate
                if self.cfg.evaluate.fixed_generation_duration:
                    target_duration = self.cfg.evaluate.fixed_generation_duration

                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration,
                    **self.generation_params
                )
                y_pred = gen_outputs['gen_audio'].detach()
                y_pred = y_pred[..., :audio.shape[-1]]

                normalize_kwargs = dict(self.cfg.generate.audio)
                normalize_kwargs.pop('format', None)
                y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
                y = audio.cpu()  # should already be on CPU but just in case
                sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
                sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
                audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]

                if fad is not None:
                    if self.cfg.metrics.fad.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    fad.update(y_pred, y, sizes, sample_rates, audio_stems)
                if kldiv is not None:
                    if self.cfg.metrics.kld.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    kldiv.update(y_pred, y, sizes, sample_rates)
                if text_consistency is not None:
                    texts = [m.description for m in meta]
                    if self.cfg.metrics.text_consistency.use_gt:
                        y_pred = y
                    text_consistency.update(y_pred, texts, sizes, sample_rates)
                if chroma_cosine is not None:
                    if self.cfg.metrics.chroma_cosine.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    chroma_cosine.update(y_pred, y, sizes, sample_rates)
                    # restore chroma conditioner's eval chroma wavs
                    if eval_chroma_wavs is not None:
                        self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)

            flashy.distrib.barrier()
            if fad is not None:
                metrics['fad'] = fad.compute()
            if kldiv is not None:
                kld_metrics = kldiv.compute()
                metrics.update(kld_metrics)
            if text_consistency is not None:
                metrics['text_consistency'] = text_consistency.compute()
            if chroma_cosine is not None:
                metrics['chroma_cosine'] = chroma_cosine.compute()
            metrics = average(metrics)
            metrics = flashy.distrib.average_metrics(metrics, len(loader))

        return metrics

    def evaluate(self) -> dict:
        """Evaluate stage."""
        self.model.eval()
        with torch.no_grad():
            metrics: dict = {}
            if self.cfg.evaluate.metrics.base:
                metrics.update(self.common_train_valid('evaluate'))
            gen_metrics = self.evaluate_audio_generation()
            return {**metrics, **gen_metrics}
