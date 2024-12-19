import json
from tqdm import tqdm
note_type = {'use_colorNotes': True,
              'use_chains': False,
                'use_bombNotes': False,
                  'use_obstacles': False,
                    'use_arcs': False}
note_size = {'use_colorNotes': 18, 'use_chains': 2, 'use_bombNotes': 1, 'use_obstacles': 2, 'use_arcs': 2}
token_id_size = sum(
            size for note, size in note_size.items()
            if note_type.get(note, False)
        )
kwargs = {'segment_duration': 30, 'num_samples': 1000000, 'shuffle': True, 'sample_on_duration': False, 'sample_on_weight': False, 'min_segment_ratio': 0.8, 'position_size': 12, 'token_id_size': token_id_size, 'note_type': note_type, 'shuffle_seed': 0, 'permutation_on_files': False, 'merge_text_p': 0.25, 'drop_desc_p': 0.5, 'drop_other_p': 0.5, 'sample_rate': 32000, 'channels': 1, 'beatmap_sample_window': 32, 'minimum_note': 0.25 }
path = '/home/kk/project/audiocraft/egs/bs_curated'
from audiocraft import data
dataset = data.audio_dataset_beatmap.AudioDataset.from_meta(path, **kwargs)
# if True:
#     map_arcs = 0
#     map_bombNotes = 0
#     map_obstacles = 0
#     map_chains = 0
#     for file_meta in tqdm(dataset.meta, desc="Processing files"):
#         if file_meta.arcs_num:
#             map_arcs += 1
#         if file_meta.bombNotes_num:
#             map_bombNotes += 1
#         if file_meta.obstacles_num:
#             map_obstacles += 1
#         if file_meta.chains_num:
#             map_chains += 1
#     print("map_arcs =", map_arcs)
#     print("map_bombNotes =", map_bombNotes)
#     print("map_obstacles =", map_obstacles)
#     print("map_chains =", map_chains)
start_index = 1954
start_index = 0
minimum_note = 0.25
# bs_curated 56 is buggy
for file_meta in tqdm(dataset.meta[57:], desc="Processing files"):
    # file_meta.beatmap_file_path = '/home/kk/project/audiocraft2/dataset/CustomLevels/636 (Burn - bennydabeast)/Easy.json'
    # file_meta.bpm = 174
    # file_meta.duration = 232.91977324263038
    if note_type['use_chains'] and file_meta.chains_num == 0:
        continue
    if note_type['use_bombNotes'] and file_meta.bombNotes_num == 0:
        continue
    if note_type['use_arcs'] and file_meta.arcs_num == 0:
        continue
    if note_type['use_obstacles'] and file_meta.obstacles_num == 0:
        continue
            
    with open(file_meta.beatmap_file_path, 'r', encoding = 'utf-8') as f:
                    beatmap_file = json.load(f)
    beatmap = data.audio_dataset_beatmap.Beatmap(minimum_note = minimum_note, token_id_size = dataset.token_id_size, position_size = dataset.position_size, **dataset.note_type)
    segment_duration_in_quaver = round(file_meta.duration / 60 * file_meta.bpm / minimum_note)
    beatmap_file = beatmap.sample_beatmap_file(beatmap_file, 0,  segment_duration_in_quaver, file_meta.bpm)
    beatmap_token = beatmap.tokenize(beatmap_file, segment_duration_in_quaver,file_meta.bpm)
    reconstructed_beatmap_file = beatmap.detokenize(beatmap_token, file_meta.bpm)
    result = beatmap.check_difference(beatmap_file, reconstructed_beatmap_file)

    if result:
        print(result)
    

# run command:
# python -m tests.data.test_beatmap