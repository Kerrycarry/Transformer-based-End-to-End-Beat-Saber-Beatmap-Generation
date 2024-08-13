from audiocraft.utils import export
from audiocraft import train

xp = train.main.get_xp_from_sig('f414f290')
export.export_lm(xp.folder / 'checkpoint.th', '/checkpoints/my_audio_lm/state_dict.bin')
## Case 2) you used a pretrained model. Give the name you used without the //pretrained/ prefix.
## This will actually not dump the actual model, simply a pointer to the right model to download.
export.export_pretrained_compression_model('facebook/encodec_32khz', '/checkpoints/my_audio_lm/compression_state_dict.bin')

import audiocraft.models
from audiocraft.data.audio import audio_write
model = audiocraft.models.MusicGen.get_pretrained('/checkpoints/my_audio_lm/')

model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
# descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
# wav = model.generate(descriptions)  # generates 3 samples.

# melody, sr = torchaudio.load('./assets/bach.mp3')
# # generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)