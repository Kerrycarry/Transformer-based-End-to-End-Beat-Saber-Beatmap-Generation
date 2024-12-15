
import numpy as np
import requests
from audiocraft.solvers.beatmapgen import BeatmapGenSolver
from audiocraft.data.audio_dataset_beatmap import AudioDataset
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
kwargs = {'segment_duration': 30, 'num_samples': 1000000, 'shuffle': True, 'sample_on_duration': False, 'sample_on_weight': False, 'min_segment_ratio': 0.8, 'position_size': 12, 'token_id_size': token_id_size, 'note_type': note_type, 'shuffle_seed': 0, 'permutation_on_files': False, 'merge_text_p': 0.25, 'drop_desc_p': 0.5, 'drop_other_p': 0.5, 'sample_rate': 32000, 'channels': 1, 'beatmap_sample_window': 32, 'minimum_note': 0.125}
path = '/home/kk/project/audiocraft/egs/bs_rank'
dataset = AudioDataset.from_meta(path, **kwargs)

# Client-side logic to call the API and perform remaining computations
def call_api(api_url, y_beat_times):
    response = requests.post(api_url, json={"y_beat_times": y_beat_times})
    response.raise_for_status()
    response_json = response.json()
    audio_token = np.array(response_json["audio_token"])
    return audio_token

def one_tried(bpm,length, log=False):
    api_url = "http://localhost:8000/generate_audio_token"  # Adjust URL as needed
    segment_duration_in_quaver = length *bpm /60 * 8
    length = length*32000
    y_beat_times, times = dataset.generate_y_beat_times(bpm, segment_duration_in_quaver,length)
    y_beat_times = y_beat_times.tolist()
    audio_token = call_api(api_url, y_beat_times)
    results = BeatmapGenSolver.clustering_audio_token(audio_token)
    continuous_blocks = results['continuous_blocks']
    
    is_fail = times != continuous_blocks
    if is_fail:
        print(f"bpm {bpm} failed, ")
    if log or is_fail:
        print(f"Number of beats: {times}")
        print(f"duration of beats: {60 / bpm * 0.25}")
        print(f"Count of label 0: {results['label_counts'][0]}")
        print(f"Count of label 1: {results['label_counts'][1]}")
        print(f"Number of continuous blocks: {continuous_blocks}")
        print(f"Sample labels start: {results['sample_labels_start']}")
        print(f"Sample labels end: {results['sample_labels_end']}")
        print()
# api 
# uvicorn tests.data.test_codec:app --host 0.0.0.0 --port 8000
# Example usage
if __name__ == "__main__":
    bpm = 220
    one_tried(bpm,length = 30, log=True)

    # for bpm in range(65, 280):
    #     one_tried(bpm, length = 30)
    
    # for file_meta in dataset.meta:
    #     bpm = file_meta.bpm
    #     if bpm <65 or bpm >280:
    #         continue    
    #     one_tried(bpm, length = 30)