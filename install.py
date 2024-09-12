import launch
import os
import sys

# TODO: add pip dependency if need extra module only on extension

# if not launch.is_installed("aitextgen"):
#     launch.run_pip("install aitextgen==0.6.0", "requirements for MagicPrompt")

if not os.path.exists("extensions\DigitalHuman\DH_live"):
    git_cmd = ("cd extensions\DigitalHuman && git clone https://github.com/kleinlee/DH_live.git")
    os.system(git_cmd)

    checkpoint_cmd = "cd extensions\DigitalHuman\DH_live\checkpoint && copy /b render.pth.gz.001 + render.pth.gz.002 render.pth.gz && gzip -d render.pth.gz"
    os.system(checkpoint_cmd)

# if not os.path.exists("extensions\DigitalHuman\ChatTTS"):
#     git_cmd = ("cd extensions\DigitalHuman && git clone https://github.com/2noise/ChatTTS")
#     os.system(git_cmd)
#     requests_file = "extensions\DigitalHuman\ChatTTS\\requirements.txt"
#     # read
#     with open(requests_file, "r") as f:
#         lines = f.readlines()
#         data = []
#         for i in lines:
#             # 根据条件修改
#             if ('torchaudio' in i):
#                 i = i.replace('torchaudio', 'torchaudio==2.3.1')  # 修改abc为def
#             data.append(i)  # 记录每一行
#     # write
#     with open(requests_file, "w") as f:
#         for i in data:
#             f.writelines(i)
#
#     pip_cmd = "cd ..\\system\python && python -m pip install -r ..\..\webui\extensions\DigitalHuman\ChatTTS\\requirements.txt"
#     os.system(pip_cmd)

DigitalHuman_kv = {
# DH_live
    "beautifulsoup4": "beautifulsoup4==4.12.3",
    "dominate": "dominate==2.9.0",
    "fastapi": "fastapi==0.111.1",
    "kaldi_native_fbank": "kaldi_native_fbank==1.19.1",
    "librosa": "librosa==0.10.1",
    "mediapipe": "mediapipe==0.10.14",
    "opencv_contrib_python": "opencv_contrib_python==4.8.1.78",
    "opencv_python": "opencv_python==4.9.0.80",
    "pandas": "pandas==2.0.3",
    "Pillow": "Pillow==10.4.0",
    "Requests": "Requests==2.32.3",
    "scipy": "scipy==1.11.1",
    "sounddevice": "sounddevice==0.4.6",
    "thop": "thop==0.1.1.post2209072238",
    "tqdm": "tqdm==4.66.4",
    "uvicorn": "uvicorn==0.30.3",
    "visdom": "visdom==0.2.4",
    "wandb": "wandb==0.16.5",
    "tensorboard": "tensorboard==2.17.0",
# edge-tts
    "edge-tts": "edge-tts",
}

for k, v in DigitalHuman_kv.items():
    if not launch.is_installed(k):
        print(k, launch.is_installed(k))
        launch.run_pip("install " + v, "requirements for DigitalHuman")
