from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from pre_DH_live import DH_live_inference

DigitalHuman_path = os.path.dirname(os.path.abspath(__file__))

def qianwen(prompt, history, sys_prompt):
    model_name = DigitalHuman_path + "/models/Qwen2-0.5B-Instruct"
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": sys_prompt},
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



import torch
import time
import datetime
import re
import random
import tqdm
import ChatTTS
import torchaudio
import numpy as np
import wave
def clear_cuda_cache():
    """
    Clear CUDA cache
    :return:
    """
    torch.cuda.empty_cache()
def normalize_audio(audio):
    """
    Normalize audio array to be between -1 and 1
    :param audio: Input audio array
    :return: Normalized audio array
    """
    audio = np.clip(audio, -1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio
def save_audio(file_name, audio, rate=24000):
    """
    保存音频文件
    :param file_name:
    :param audio:
    :param rate:
    :return:
    """
    import os
    audio = (audio * 32767).astype(np.int16)

    # 检查默认目录
    if not os.path.exists("extensions/DigitalHuman/output"):
        os.makedirs("extensions/DigitalHuman/output")
    full_path = os.path.join("extensions/DigitalHuman/output", file_name)
    with wave.open(full_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    return full_path
def generate_audio(text_file, pt_file, seed=0, batch_size=4, speed=4,
                            temperature=0.1, top_P=0.7, top_K=20, cur_tqdm=None, skip_save=False,
                            skip_refine_text=False):
    print("33333333333333333333333333333333", DigitalHuman_path + "/models")
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    wav_filename = DigitalHuman_path + "/my_res.wav"
    rnd_spk_emb = torch.load(pt_file)
    params_infer_code = {
        'spk_emb': rnd_spk_emb,
        'prompt': f'[speed_{speed}]',
        'top_P': top_P,
        'top_K': top_K,
        'temperature': temperature
    }
    _params_infer_code = {**params_infer_code}
    wavs = chat.infer(text_file, params_infer_code=_params_infer_code)
    wavs = [normalize_audio(w) for w in wavs]  # 先对每段音频归一化
    combined_audio = np.concatenate(wavs, axis=1)

    save_audio(wav_filename, combined_audio)
    return wav_filename



import gradio as gr
def human_final(prompt, history, audio_name, video_name, system_prompt):
    if audio_name=="空" or video_name=="空":
        gr.Info("音频模型 或 数字人模型 不能为空")
    Qwen_response = qianwen(prompt, history, system_prompt)
    print("000000000000000000000000000000000000", Qwen_response)
    history.append((prompt, Qwen_response))
    audio_pt_path = DigitalHuman_path + "/output/" + audio_name + ".pt"
    print("11111111111111111111111111111111111", audio_pt_path)
    voice_inference = generate_audio(Qwen_response, audio_pt_path)

    # voice_inference = inference_audio(Qwen_response, audio_pt_path)
    human_video = DH_live_inference(voice_inference, video_name=video_name)
    # "{}/human_pre.pkl".format(os.path.dirname(video_out_path))

    return '', history, human_video
