import subprocess


def text_to_speech_wav(text, filename):
    # Command to generate a WAV file from text
    subprocess.call(['espeak-ng', '-w ' + filename, text])


# Example usage
# text_to_speech_wav("Hello, this is a test", f"outputs/output.wav")