from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from pre_DH_live import DH_live_inference

DigitalHuman_path = os.path.dirname(os.path.abspath(__file__))

def qianwen(prompt, history):
    model_name = DigitalHuman_path + "/models/Qwen2-0.5B-Instruct"
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": "中文回答"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


import asyncio
import random

import edge_tts
from edge_tts import VoicesManager
import soundfile as sf
import numpy as np
import librosa
def resample_audio(driven_audio_path):
    # 读取上传的音频文件
    audio, sr = sf.read(driven_audio_path)
    # 如果是立体声，转换为单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # 重采样到16kHz
    resampled_audio = librosa.resample(audio,
                                       orig_sr=sr,
                                       target_sr=16000)
    # 将音频格式转换为16位深度
    resampled_audio = (resampled_audio * 32767).astype(np.int16)
    # 保存处理后的音频到wav文件
    output_file = driven_audio_path
    sf.write(output_file, resampled_audio, 16000, subtype='PCM_16')

    return output_file  # 返回保存的音频文件路径

async def amain(response, audio_name, audio_path) -> None:
    """Main function"""
    voices = await VoicesManager.create()
    voice = audio_name
    # Also supports Locales
    # voice = voices.find(Gender="Female", Locale="es-AR")

    communicate = edge_tts.Communicate(response, voice)
    await communicate.save(audio_path)


import gradio as gr
def human_final(prompt, history, audio_name, video_name):
    if audio_name=="空" or video_name=="空":
        gr.Info("音频模型 或 数字人模型 不能为空")
    Qwen_response = qianwen(prompt, history)
    print("Qwen 回答", Qwen_response)
    history.append((prompt, Qwen_response))
    audio_path = DigitalHuman_path + "/output/" + audio_name + ".wav"
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(amain(Qwen_response, audio_name, audio_path))
    finally:
        loop.close()
    audio_path_16 = resample_audio(audio_path)
    print("Edge 完成")
    human_video = DH_live_inference(audio_path_16, video_name=video_name)

    return '', history, human_video
