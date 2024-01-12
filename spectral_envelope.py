import io
import os

import librosa
import torch
import shutil
from glob import glob

from mel_processing import spectrogram_torch
from se_extractor import split_audio_vad, split_audio_whisper, hash_numpy_array


def extract_model(ref_wav_list, vc_model):
    if isinstance(ref_wav_list, str):
        ref_wav_list = [ref_wav_list]

    hps = vc_model.hps
    model = vc_model.model
    device = "cpu"
    gs = []

    for fname in ref_wav_list:
        audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
        y = torch.FloatTensor(audio_ref)
        y = y.to(device)
        y = y.unsqueeze(0)
        y = spectrogram_torch(y, hps.data.filter_length,
                              hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                              center=False).to(device)
        with torch.no_grad():
            g = model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
            gs.append(g.detach())

    gs = torch.stack(gs).mean(0)

    # if se_save_path is not None:
    #     os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
    #     torch.save(gs.cpu(), se_save_path)

    return gs


def extract_se(audio_path, vc_model, audio_name):
    # Create a temporary directory
    temp_dir = os.path.join("temp", "temp_wavs")
    os.makedirs(temp_dir, exist_ok=True)

    # Split audio and save in temporary directory
    wavs_folder = split_audio_vad(audio_path, target_dir=temp_dir, audio_name=audio_name)

    # Process the audio segments
    audio_segs = glob(f'{wavs_folder}/*.wav')
    result = extract_model(audio_segs, vc_model)

    # Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)

    return result


def get_se(audio_path, vc_model, target_dir='processed', vad=True):
    device = vc_model.device

    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{hash_numpy_array(audio_path)}"
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if os.path.isfile(se_path):
        se = torch.load(se_path).to(device)
        return se, audio_name

    if os.path.isdir(audio_path):
        wavs_folder = audio_path
    elif vad:
        wavs_folder = split_audio_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
    else:
        wavs_folder = split_audio_whisper(audio_path, target_dir=target_dir, audio_name=audio_name)

    audio_segs = glob(f'{wavs_folder}/*.wav')
    if len(audio_segs) == 0:
        raise NotImplementedError('No audio segments found!')

    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name
