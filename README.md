# SD webui forge 使用手册

# 支撑下载

## FFmpeg 下载 (windows)

1、官网下载 windows 版本

[Download FFmpeg](https://www.ffmpeg.org/download.html#build-windows)

2、把 FFmpeg 加到 环境变量中

![image.png](image.png)

## web ui forge 下载

项目地址 https://github.com/lllyasviel/stable-diffusion-webui-forge 官网下载 windows 一键启动

![image.png](image%201.png)

1. 解压缩并进入 `webui_forge_cu121_torch231` 目录中
2. 下载 [flux1-dev-bnb-nf4-v2.safetensors](https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/blob/main/flux1-dev-bnb-nf4-v2.safetensors) 放到 `webui_forge_cu121_torch231\webui\models\Stable-diffusion\` 目录下
3. 进入目录 `webui_forge_cu121_torch231\system` 删除 `python`文件夹。
4. 使用 conda 重新生成 python 库，在 `webui_forge_cu121_torch231\system` 目录下 cmd ，输入`conda create --prefix E:\3dhuman\webui_forge_cu121_torch231\system\python python=3.10.6`
    
    `conda activate E:\3dhuman\webui_forge_cu121_torch231\system\python`
    
    `conda install -c conda-forge pynini`
    
5. 修改 webui_forge\webui\requirements_versions.txt

einops==0.4.1   →     einops==0.6.1

1. 执行 `update.bat`
2. 执行 `run.bat`

## LivePortrait 插件安装

项目原地址 https://github.com/KwaiVGI/LivePortrait

### 下载LivePortrait sd插件

项目地址： https://github.com/dimitribarbot/sd-webui-live-portrait

1. 打开“Extensions”选项卡。 
2. 在选项卡中打开“Install from URL”选项卡。 
3. 输入 [https://github.com/dimitribarbot/sd-webui-live-portrait.git](https://github.com/dimitribarbot/sd-webui-live-portrait.git) 到“URL for extension's git repository”。 
4. 按“Install”按钮。 
5. 安装可能需要几分钟，因为 XPose 可能会被编译。最后，您将看到消息“Installed into stable-diffusion-webui\extensions\sd-webui-live-portrait. Use Installed tab to restart。
6.  转到“Installed”选项卡，单击“Check for updates”，然后单击“Apply and restart UI”。 （下次您还可以使用这些按钮来更新此扩展。）

在 SD webui forge 不能直接用sd-webui-live-portrait，有一些error(SD webui forge是gradio4+有一些改动)

- All model fields require a type annotation; if flag stitching retargeting input' is not meant to be a field, you maybe able to resolve this error by annotating it as a 'ClassVar' or updating model config ' ignored types '] ,
    
    ![image.png](image%202.png)
    
    `webui_forge_cu121_torch231\webui\extensions\sd-webui-live-portrait\scripts\api.py` 注销515、589
    
    ![image.png](image%203.png)
    
    ![image.png](image%204.png)
    
- ValueError: Invalid value for parameter 'type': file. Please choose from one of: ['filepath' , ' binary']
    
    ![image.png](image%205.png)
    
    `webui_forge_cu121_torch231\webui\extensions\sd-webui-live-portrait\scripts\main.py` 修改 350 行
    
    ![image.png](image%206.png)
    
- TypeError: save_pil_to_file() got an unexpected keyword argument 'name’
    
    ![image.png](image%207.png)
    
    gr.Image() 4.+  `gr.Image(type="filepath")` 有点问题 
    
    `webui_forge_cu121_torch231\webui\extensions\sd-webui-live-portrait\scripts\main.py` 
    
    366 行
    
    ![image.png](image%208.png)
    
    331 行
    
    ![image.png](image%209.png)
    
    346 行
    
    ![image.png](image%2010.png)
    
    661 行
    
    ![image.png](image%2011.png)
    
    上边这几行的 `source_image_input = gr.Image(type="filepath")`  都改成 `source_image_input = gr.Image()`
    
    ![image.png](image%2012.png)
    
    2222********************************************************************************
    
    ![image.png](image%2013.png)
    
    改成
    
    ```python
    from PIL import Image
    import numpy as np
    
        def gpu_wrapped_execute_video(*args, **kwargs):
            # print("Args:", args)
            tmp_args = list(args)
            if isinstance(tmp_args[0], np.ndarray):
                # print("tmp_args[0]:", isinstance(tmp_args[0], np.ndarray))
                source_image_input_path = 'extensions/sd-webui-live-portrait/source_image_input.jpg'
                tmp_img = Image.fromarray(tmp_args[0])
                tmp_img.save(source_image_input_path)
                tmp_args[0] = source_image_input_path
            if isinstance(tmp_args[4], np.ndarray):
                driving_image_input = 'extensions/sd-webui-live-portrait/driving_image_input.jpg'
                tmp_img = Image.fromarray(tmp_args[4])
                tmp_img.save(driving_image_input)
                tmp_args[4] = driving_image_input
            if isinstance(tmp_args[5], np.ndarray):
                driving_image_webcam_input = 'extensions/sd-webui-live-portrait/driving_image_webcam_input.jpg'
                tmp_img = Image.fromarray(tmp_args[5])
                tmp_img.save(driving_image_webcam_input)
                tmp_args[5] = driving_image_webcam_input
    
            pipeline = init_gradio_pipeline()
            return pipeline.execute_video(*tmp_args, **kwargs)
    ```
    
    ![image.png](image%2014.png)
    

### 下载模型

[https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main](https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main) 

model放置：(human)

`webui_forge_cu121_torch231\webui\models\liveportrait`

```bash
liveportrait
|__base_models
|____appearance_feature_extractor.safetensors
|____gitattributes
|____landmark_model.pth
|____motion_extractor.safetensors
|____spade_generator.safetensors
|____warping_module.safetensors
|__retargeting_models
|____stitching_retargeting_module.safetensors
|__landmark.onnx
```

`webui_forge_cu121_torch231\webui\models\insightface\models\buffalo_l\`

```bash
buffalo_l
|__2d106det.onnx
|__det_10g.onnx
```

# DigitalHuman 插件开发

位置：在 `webui_forge\webui\extensions`

## DH_live

 https://github.com/kleinlee/DH_live

在目录 `webui_forge\webui\extensions\DigitalHuman\` 下，打开 cmd，执行

```bash
git clone [https://github.com/kleinlee/DH_live.git](https://github.com/kleinlee/DH_live.git)
```

webui_forge_cu121_torch231\webui\extensions\DigitalHuman\DH_live\checkpoint

```bash
copy /b render.pth.gz.001 + render.pth.gz.002 render.pth.gz
gzip -d render.pth.gz
```

## CosyVoice

## Qwen
