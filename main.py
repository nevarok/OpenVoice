import os

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

base_speaker_tts = BaseSpeakerTTS(base_speaker_hps, device=device)
base_speaker_tts.load_checkpoint_dict(base_speaker_checkpoint)

tone_color_converter_checkpoint_path = "converted/tone_color_converter_checkpoint.bin"
model = load_model(f'{ckpt_converter}/checkpoint.pth')
buffer = convert_to_bytes(model)
save_bytes_to_file(buffer=buffer, file_path=tone_color_converter_checkpoint_path)
tone_color_converter_checkpoint = load_model_from_bytes(load_bytes_from_file(tone_color_converter_checkpoint_path))
tone_color_converter_hps = get_hparams_from_json(load_json_data(f'{ckpt_converter}/config.json'))

tone_color_converter = ToneColorConverter(tone_color_converter_hps, device=device)
tone_color_converter.load_checkpoint_dict(tone_color_converter_checkpoint)

os.makedirs(output_dir, exist_ok=True)

source_se_checkpoint_path = "converted/source_se_checkpoint.bin"
model = load_model(f'{ckpt_base}/en_default_se.pth')
buffer = convert_to_bytes(model)
save_bytes_to_file(buffer=buffer, file_path=source_se_checkpoint_path)
source_se_checkpoint = load_model_from_bytes(load_bytes_from_file(source_se_checkpoint_path))

reference_speaker_name = "demo_speaker1"
reference_speaker = f'resources/{reference_speaker_name}.mp3'

target_se_checkpoint_path = f"target_se/{reference_speaker_name}_se.bin"
target_se = extract_se(reference_speaker, tone_color_converter, reference_speaker_name)
buffer = convert_to_bytes(target_se)
save_bytes_to_file(buffer=buffer, file_path=target_se_checkpoint_path)
target_se = load_model_from_bytes(load_bytes_from_file(target_se_checkpoint_path))

save_path = f'{output_dir}/output_en_default.wav'

text = "the quick brown fox jumps over the lazy dog"
src_path = f'{output_dir}/tmp.wav'
speakers = ["default", "friendly", "cheerful", "excited", "sad", "angry", "terrified", "shouting", "whispering"]

for speaker in speakers:
    base_speaker_path = f'{output_dir}/{speaker}_tmp.wav'
    audio = base_speaker_tts.tts2(text, speaker=speaker, language='English', speed=1.0)

    save_path = f'{output_dir}/{speaker}_out.wav'

    audio = tone_color_converter.convert2(
        audio_src=audio,
        src_se=source_se_checkpoint,
        tgt_se=target_se)

    tone_color_converter.save_audio(audio, save_path)
