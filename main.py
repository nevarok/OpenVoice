import os
import time

import soundfile

import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
from json_config import load_json_data, get_hparams_from_json
from model_converter import convert_to_bytes, save_bytes_to_file, load_bytes_from_file, load_model_from_bytes, \
    load_model
from spectral_envelope import extract_se

# C:\ffmpeg\bin

ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
# device="cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
output_dir = 'outputs'

base_speaker_checkpoint_path = "converted/base_speaker_checkpoint.bin"
model = load_model(f'{ckpt_base}/checkpoint.pth')
buffer = convert_to_bytes(model)
save_bytes_to_file(buffer=buffer, file_path=base_speaker_checkpoint_path)
base_speaker_checkpoint = load_model_from_bytes(load_bytes_from_file(base_speaker_checkpoint_path))
base_speaker_hps = get_hparams_from_json(load_json_data(f'{ckpt_base}/config.json'))

# #MMS-TTS TEST
# mms_tts_checkpoint_path = "checkpoints/converted/mms_tts_checkpoint.bin"
# model = load_model(f'checkpoints/base_speakers/MMS-TTS-ENG/D_100000.pth')
# buffer = convert_to_bytes(model)
# save_bytes_to_file(buffer=buffer, file_path=mms_tts_checkpoint_path)
# base_speaker_checkpoint = load_model_from_bytes(load_bytes_from_file(mms_tts_checkpoint_path))
# base_speaker_hps = get_hparams_from_json(load_json_data(f'checkpoints/base_speakers/MMS-TTS-ENG/full_models_eng_config.json'))

base_speaker_tts = BaseSpeakerTTS(base_speaker_hps, device=device)
# base_speaker_tts.load_checkpoint_dict(f'{ckpt_base}/checkpoint.pth')
base_speaker_tts.load_checkpoint_dict(base_speaker_checkpoint)

tone_color_converter_checkpoint_path = "converted/tone_color_converter_checkpoint.bin"
model = load_model(f'{ckpt_converter}/checkpoint.pth')
buffer = convert_to_bytes(model)
save_bytes_to_file(buffer=buffer, file_path=tone_color_converter_checkpoint_path)
tone_color_converter_checkpoint = load_model_from_bytes(load_bytes_from_file(tone_color_converter_checkpoint_path))
tone_color_converter_hps = get_hparams_from_json(load_json_data(f'{ckpt_converter}/config.json'))

tone_color_converter = ToneColorConverter(tone_color_converter_hps, device=device)
# tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
tone_color_converter.load_checkpoint_dict(tone_color_converter_checkpoint)

os.makedirs(output_dir, exist_ok=True)
# source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

source_se_checkpoint_path = "converted/source_se_checkpoint.bin"
model = load_model(f'{ckpt_base}/en_default_se.pth')
buffer = convert_to_bytes(model)
save_bytes_to_file(buffer=buffer, file_path=source_se_checkpoint_path)
source_se_checkpoint = load_model_from_bytes(load_bytes_from_file(source_se_checkpoint_path))

reference_speaker_name = "demo_speaker1"
reference_speaker = f'resources/{reference_speaker_name}.mp3'
reference_speaker_name = "Morgan_Freeman"
reference_speaker = f'resources/{reference_speaker_name}.flac'
# se spectral envelope
# target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

target_se_checkpoint_path = f"target_se/{reference_speaker_name}_se.bin"
target_se = extract_se(reference_speaker, tone_color_converter, reference_speaker_name)
buffer = convert_to_bytes(target_se)
save_bytes_to_file(buffer=buffer, file_path=target_se_checkpoint_path)
target_se = load_model_from_bytes(load_bytes_from_file(target_se_checkpoint_path))

# target_se can be cached or pickled
# print(f"audio_name: [{audio_name}]\ntarget_se: [{target_se}]")

save_path = f'{output_dir}/output_en_default.wav'

# Run the base speaker tts
text = "the quick brown fox jumps over the lazy dog"
src_path = f'{output_dir}/tmp.wav'
speakers = ["friendly", "cheerful", "excited", "sad", "angry", "terrified", "shouting", "whispering"]

for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for speaker in speakers:
        # ???
        # base_speaker_path = f'{output_dir}/{speaker}_{tau}_tmp.wav'
        start_time = time.time()
        src_audio = base_speaker_tts.tts2(text, speaker=speaker, language='English', speed=1.0)
        end_time = time.time()
        tts2_duration_ms = (end_time - start_time) * 1000

        tau_str = str(tau).replace('.', '')
        save_path = f'{output_dir}/{speaker}_{tau_str}_out.wav'

        # ???
        start_time = time.time()
        audio, sample_rate = tone_color_converter.convert2(
            src_audio=src_audio,
            src_se=source_se_checkpoint,
            tgt_se=target_se,
            tau=tau)
        end_time = time.time()
        convert2_duration_ms = (end_time - start_time) * 1000

        soundfile.write(save_path, audio, sample_rate)

        print(f"tts2_duration_ms = [{tts2_duration_ms}]\nconvert2_duration_ms = [{convert2_duration_ms}]")
