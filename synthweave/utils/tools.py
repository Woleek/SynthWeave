import json
import torchvision
import torchaudio


def read_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)
    
    
def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit='sec')
    return video, audio, info


def read_audio(path: str):
    return torchaudio.load(path)


def read_image(path: str):
    return torchvision.io.decode_image(path)