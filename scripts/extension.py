import os
import sys
import gradio as gr
from modules.call_queue import wrap_queued_call
from modules import script_callbacks

import soundfile as sf
import numpy as np
import librosa

# DigitalHuman_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LivePortrait_dir = DigitalHuman_path + '/LivePortrait/'
# sys.path.append(LivePortrait_dir)
from pre_DH_live import DH_live_preparation, DH_live_inference
from pre_QwenChat import human_final
import shutil


DigitalHuman_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = DigitalHuman_path + "/output/"

def save_cosy_name(name, input_audio_path):
    if not name or name == "":
        gr.Info("音色名称不能为空")
        return False
    shutil.copyfile(DigitalHuman_path + "/output.pt", output_path + f"{name}.pt")
    gr.Info("音色保存成功,存放位置为 webui_forge/webui/extensions/DigitalHuman/output 目录")

def save_video_name(name):
    if not name or name == "":
        gr.Info("数字人名称不能为空")
        return False
    shutil.copyfile(DigitalHuman_path + "/human_pre.pkl",
                    output_path + f"{name}.pkl")
    shutil.copyfile(DigitalHuman_path + "/human_pre.mp4",
                    output_path + f"{name}.mp4")
    gr.Info("数字人保存成功,存放位置为 webui_forge/webui/extensions/DigitalHuman/output 目录")

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
    output_file = output_path + "processed_audio.wav"
    sf.write(output_file, resampled_audio, 16000, subtype='PCM_16')

    return output_file  # 返回保存的音频文件路径

reference_audio = ["空", "zh-CN-YunyangNeural", "zh-CN-XiaoxiaoNeural"]
# for name in os.listdir(output_path):
#     if os.path.splitext(name)[1] == '.pt':
#         reference_audio.append(name.replace(".pt", ""))
reference_video = ["空"]
for name in os.listdir(output_path):
    if os.path.splitext(name)[1] == '.pkl':
        reference_video.append(name.replace(".pkl", ""))
def change_audio_choices():
    # reference_audio = ["空"]
    # for name in os.listdir(output_path):
    #     if os.path.splitext(name)[1] == '.pt':
    #         reference_audio.append(name.replace(".pt", ""))
    reference_audio = ["空", "zh-CN-YunyangNeural", "zh-CN-XiaoxiaoNeural"]
    return {"choices": reference_audio, "__type__": "update"}
def change_video_choices():
    reference_video = ["空"]
    for name in os.listdir(output_path):
        if os.path.splitext(name)[1] == '.pkl':
            reference_video.append(name.replace(".pkl", ""))
    return {"choices": reference_video, "__type__": "update"}

def change_video(video_name):
    return output_path + "/" + video_name + ".mp4"

# UI部分
def on_ui_tabs():

    with gr.Blocks(analytics_enabled=False) as DigitalHuman_interface:
        gr.Markdown("<h2> <center>口唇驱动（DH_live）+ 交互问答（Qwen2）</center></h2>")

        with gr.Tabs():
            gr.Markdown("<div><h3>STEP1: &nbsp 使用 &nbsp DH_live &nbsp 实现数字人音频的口唇驱动 &nbsp 详细代码在 \
                        <a style='color:#9C9C9C' href='https://github.com/kleinlee/DH_live'> Github </h3><p></div>")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(variant='panel'):
                            gr.Markdown("1、输入视频进行预处理，对初始输入的视频，进行处理，去除口唇不清楚的帧")
                            with gr.Row(variant='panel'):
                                source_video = gr.Video(label="上传数字人原始视频")
                                with gr.Column():
                                    pre_video_btn = gr.Button('数字人预处理', variant='primary')
                                pre_video = gr.Video(label="预处理的视频", format="mp4")
                    with gr.Row():
                        with gr.Column(variant='panel'):
                            gr.Markdown("2、输入音频进行预处理，将音频处理成16kHz采样率，16位bit的单声道音频")
                            with gr.Row(variant='panel'):
                                driven_audio = gr.Audio(label="上传数字人音频", type="filepath")
                                with gr.Column():
                                    pre_audio_btn = gr.Button('获得所需的音频格式', variant='primary')
                                pre_audio = gr.Audio(label="得到所需的输入音频", format="wav", type="filepath")
                with gr.Column():
                    with gr.Row():
                        with gr.Column(variant='panel'):
                            gr.Markdown("3、获得测试数字人视频")
                            gen_video_btn = gr.Button('生成数字人', variant='primary')
                            gen_video = gr.Video(label="数字人的视频", format="mp4")
                    with gr.Row():
                        with gr.Column(variant='panel'):
                            new_video_name = gr.Textbox(label="4、输入新的数字人名称", placeholder="输入新的数字人名称.",
                                                       value='')
                            save_video_btn = gr.Button("保存刚刚推理的数字人模型")

        pre_video_btn.click(
            fn=wrap_queued_call(DH_live_preparation),
            inputs=[source_video],
            outputs=[pre_video]
        )
        pre_audio_btn.click(
            fn=wrap_queued_call(resample_audio),
            inputs=[driven_audio],
            outputs=[pre_audio])
        gen_video_btn.click(
            fn=DH_live_inference,
            inputs=[pre_audio, pre_video],
            outputs=[gen_video]
        )
        save_video_btn.click(
            fn=save_video_name,
            inputs=[new_video_name]
        )

        with gr.Tabs():
            gr.Markdown("<div><h3> STEP2: &nbsp 使用 &nbsp Qwen2 &nbsp 实现数字人交互问答并接入语音复刻和口唇驱动 &nbsp 详细代码在 \
                    <a style='color:#9C9C9C' href='https://github.com/QwenLM/Qwen2'> Github <h3><p></div>")
            with gr.Row():
                with gr.Column():
                    with gr.Row(variant='panel'):
                        audio_dropdown = gr.Dropdown(label="选择你想要的声音", choices=reference_audio,
                                                value="空", interactive=True)
                        refresh_audio_btn = gr.Button("刷新声音列表", variant='primary')
                        refresh_audio_btn.click(fn=change_audio_choices, inputs=[], outputs=[audio_dropdown])
                        video_dropdown = gr.Dropdown(label="选择你想要的数字人形象", choices=reference_video,
                                                value="空", interactive=True)
                        refresh_video_btn = gr.Button("刷新数字人列表", variant='primary')
                        refresh_video_btn.click(fn=change_video_choices, inputs=[], outputs=[video_dropdown])
                    with gr.Column(variant='panel'):
                        # system_prompt = gr.Textbox(value="", label="服务类型")
                        chatbot = gr.Chatbot(label='Qwen2-0.5B-Instruct', height=300)
                        query_box = gr.Textbox(label='输入你的问题', autofocus=True)
                        with gr.Row():
                            clear_btn = gr.ClearButton([query_box, chatbot], value='清空历史')
                            submit_human_btn = gr.Button(value='提交', variant='primary')
                with gr.Column(variant='panel'):
                    human_video = gr.Video(label="数字人", format="mp4", autoplay=True)

                video_dropdown.change(
                    fn=change_video,
                    inputs=[video_dropdown],
                    outputs=[human_video]
                )
                submit_human_btn.click(
                    fn=human_final,
                    # inputs=[query_box, chatbot, audio_dropdown, video_dropdown, system_prompt],
                    inputs=[query_box, chatbot, audio_dropdown, video_dropdown],
                    outputs=[query_box, chatbot, human_video]
                )

    return [(DigitalHuman_interface, "DigitalHuman", "DigitalHuman")]


# 调用函数来生成UI标签

script_callbacks.on_ui_tabs(on_ui_tabs)