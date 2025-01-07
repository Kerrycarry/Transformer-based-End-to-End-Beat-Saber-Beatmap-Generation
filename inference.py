import torchaudio
from audiocraft.models import MAGNeT, MusicGen
from audiocraft.data.audio import audio_write

# model = MAGNeT.get_pretrained('facebook/magnet-small-10secs')
# descriptions = ['disco beat', 'energetic EDM', 'funky groove']
# wav = model.generate(descriptions)  # generates 3 samples.

# for idx, one_wav in enumerate(wav):
#     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
#     audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

model = MusicGen.get_pretrained('facebook/musicgen-small')
descriptions = ['disco beat', 'energetic EDM', 'funky groove']
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'2_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
