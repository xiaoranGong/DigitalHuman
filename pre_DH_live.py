import uuid
import tqdm
import numpy as np
import cv2
import sys
import os
import math
import pickle
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

import shutil

def detect_face(frame):
    # 剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80的
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections or len(results.detections) > 1:
            return -1, None
        rect = results.detections[0].location_data.relative_bounding_box
        out_rect = [rect.xmin, rect.xmin + rect.width, rect.ymin, rect.ymin + rect.height]
        nose_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.NOSE_TIP)
        l_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.LEFT_EYE)
        r_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        # print(nose_, l_eye_, r_eye_)
        if nose_.x > l_eye_.x or nose_.x < r_eye_.x:
            return -2, out_rect

        h, w = frame.shape[:2]
        # print(frame.shape)
        if rect.xmin < 0 or rect.ymin < 0 or rect.xmin + rect.width > w or rect.ymin + rect.height > h:
            return -3, out_rect
        if rect.width * w < 100 or rect.height * h < 100:
            return -4, out_rect
    return 1, out_rect


def calc_face_interact(face0, face1):
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / ((x_max - x_min) * (y_max - y_min))
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / ((x_max - x_min) * (y_max - y_min))
    return min(tmp0, tmp1)


def detect_face_mesh(frame):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pts_3d = np.zeros([478, 3])
        if not results.multi_face_landmarks:
            print("****** WARNING! No face detected! ******")
        else:
            image_height, image_width = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                for index_, i in enumerate(face_landmarks.landmark):
                    x_px = min(math.floor(i.x * image_width), image_width - 1)
                    y_px = min(math.floor(i.y * image_height), image_height - 1)
                    z_px = min(math.floor(i.z * image_width), image_width - 1)
                    pts_3d[index_] = np.array([x_px, y_px, z_px])
        return pts_3d


def ExtractFromVideo(video_path, circle=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    dir_path = os.path.dirname(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度

    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    totalFrames = int(totalFrames)
    pts_3d = np.zeros([totalFrames, 478, 3])
    frame_index = 0
    face_rect_list = []
    mat_list = []
    model_name = os.path.basename(video_path)[:-4]
    # os.makedirs("../preparation/{}/image".format(model_name))
    for frame_index in tqdm.tqdm(range(totalFrames)):
        ret, frame = cap.read()  # 按帧读取视频
        # #到视频结尾时终止
        if ret is False:
            break
        # cv2.imwrite("../preparation/{}/image/{:0>6d}.png".format(model_name, frame_index), frame)
        tag_, rect = detect_face(frame)
        if frame_index == 0 and tag_ != 1:
            print("第一帧人脸检测异常，请剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80")
            pts_3d = -1
            break
        elif tag_ == -1:  # 有时候人脸检测会失败，就用上一帧的结果替代这一帧的结果
            rect = face_rect_list[-1]
        elif tag_ != 1:
            print("第{}帧人脸检测异常，请剔除掉多个人脸、大角度侧脸（鼻子不在两个眼之间）、部分人脸框在画面外、人脸像素低于80*80, tag: {}".format(frame_index, tag_))
            # exit()
        if len(face_rect_list) > 0:
            face_area_inter = calc_face_interact(face_rect_list[-1], rect)
            # print(frame_index, face_area_inter)
            if face_area_inter < 0.6:
                print("人脸区域变化幅度太大，请复查，超出值为{}, frame_num: {}".format(face_area_inter, frame_index))
                pts_3d = -2
                break
        face_rect_list.append(rect)
        x_min = rect[0] * vid_width
        y_min = rect[2] * vid_height
        x_max = rect[1] * vid_width
        y_max = rect[3] * vid_height
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        # x_min = int(max(0, x_mid - seq_w * 0.65))
        # y_min = int(max(0, y_mid - seq_h * 0.4))
        # x_max = int(min(vid_width, x_mid + seq_w * 0.65))
        # y_max = int(min(vid_height, y_mid + seq_h * 0.8))
        crop_size = int(max(seq_w * 1.35, seq_h * 1.35))
        x_min = int(max(0, x_mid - crop_size * 0.5))
        y_min = int(max(0, y_mid - crop_size * 0.45))
        x_max = int(min(vid_width, x_min + crop_size))
        y_max = int(min(vid_height, y_min + crop_size))
        frame_face = frame[y_min:y_max, x_min:x_max]
        #
        # print(y_min, y_max, x_min, x_max)

        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])
    cap.release()  # 释放视频对象
    return pts_3d


def DH_live_preparation(video_in):
    DigitalHuman_path = os.path.dirname(os.path.abspath(__file__)) # webui_forge\webui\extensions\DigitalHuman
    front_video_path = DigitalHuman_path + "/DH_live/data/front.mp4"
    back_video_path = DigitalHuman_path + "/DH_live/data/back.mp4"
    video_concat_path = DigitalHuman_path + "/DH_live/data/video_concat.txt"
    video_out_path = DigitalHuman_path + "/human_pre.mp4"
    if os.path.exists(front_video_path):
        os.remove(front_video_path)
        file = open(front_video_path, 'w')
        file.close()

    if os.path.exists(back_video_path):
        os.remove(back_video_path)
        file = open(back_video_path, 'w')
        file.close()
    if os.path.exists(video_out_path):
        os.remove(video_out_path)

    # 1 视频转换为25FPS, 并折叠循环拼接
    # ffmpeg_cmd = "ffmpeg -i {} -r 25 -ss 00:00:00 -t 00:02:00 -an -loglevel quiet -y {}".format(video_in_path, front_video_path)
    ffmpeg_cmd = "ffmpeg -i {} -r 25 -an -loglevel quiet -y {}".format(video_in, front_video_path)
    os.system(ffmpeg_cmd)

    cap = cv2.VideoCapture(front_video_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    ffmpeg_cmd = "ffmpeg -i {} -vf reverse -y {}".format(front_video_path, back_video_path)
    os.system(ffmpeg_cmd)

    # 检查文件是否存在
    # print("video_concat.txt exists:", os.path.exists(video_concat_path))
    # print("front.mp4 exists:", os.path.exists(video_file_path))
    ffmpeg_cmd = "ffmpeg -f concat -i {} -c:v copy -y {}".format(video_concat_path, video_out_path)

    os.system(ffmpeg_cmd)
    # exit()
    print("正向视频帧数：", frames)
    pts_3d = ExtractFromVideo(front_video_path)

    if type(pts_3d) is np.ndarray and len(pts_3d) == frames:
        print("关键点已提取")
    pts_3d = np.concatenate([pts_3d, pts_3d[::-1]], axis=0)
    Path_output_pkl = "{}/human_pre.pkl".format(os.path.dirname(video_out_path))
    with open(Path_output_pkl, "wb") as f:
        pickle.dump(pts_3d, f)

    cap = cv2.VideoCapture(video_out_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    print("循环视频帧数：", frames)
    return video_out_path

def DH_live_inference(audio_path, video_path='', video_name=''):

    DigitalHuman_path = os.path.dirname(os.path.abspath(__file__)) # webui_forge\webui\extensions\DigitalHuman
    DH_live_dir = DigitalHuman_path + "/DH_live/"
    sys.path.append(DH_live_dir)
    from talkingface.audio_model import AudioModel
    from talkingface.render_model import RenderModel

    audioModel = AudioModel()
    audioModel.loadModel(DH_live_dir + "/checkpoint/audio.pkl")
    renderModel = RenderModel()
    renderModel.loadModel(DH_live_dir + "/checkpoint/render.pth")
    # "{}/human_pre.pkl".format(os.path.dirname(video_out_path))

    if video_path != '':
        pkl_in_path = DigitalHuman_path + "/human_pre.pkl"
        video_in_path = DigitalHuman_path + "/human_pre.mp4"
        video_out_path = DigitalHuman_path + "/human_test.mp4"
        # print("使用 human_test.mp4 存的")
    elif video_name != '':
        pkl_in_path = DigitalHuman_path + "/output/{}.pkl".format(video_name)
        video_in_path = DigitalHuman_path + "/output/{}.mp4".format(video_name)
        video_out_path = DigitalHuman_path + "/output/{}tmp.mp4".format(video_name)
        # print("使用 存的", video_out_path)


    # video_path = video_path
    # human_pre_out_path = current_path + "/DH_live_output/human_pre_test.mp4"
    renderModel.reset_charactor(video_in_path, pkl_in_path)
    mouth_frame = audioModel.interface_wav(audio_path)

    cap_input = cv2.VideoCapture(video_in_path)

    vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    cap_input.release()

    task_id = str(uuid.uuid1())
    os.makedirs(DigitalHuman_path + "/output/{}".format(task_id), exist_ok=True)

    # 生成视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = DigitalHuman_path + "/output/{}/silence.mp4".format(task_id)

    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))
    for frame in tqdm.tqdm(mouth_frame):
        frame = renderModel.interface(frame)
        # cv2.imshow("s", frame)
        # cv2.waitKey(40)
        videoWriter.write(frame)
    try:
        # 确保视频帧都已写入并释放资源
        videoWriter.release()
        print("Video writer released successfully.")
    except Exception as e:
        print(f"Error releasing video writer: {e}")
    if os.path.exists(video_out_path):
        os.remove(video_out_path)
    os.system(
        "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p -loglevel quiet {}".format(save_path, audio_path, video_out_path)
    )
    # 删除临时文件
    shutil.rmtree(DigitalHuman_path + "/output/{}".format(task_id))
    print(f"Save path: {save_path}")
    print(f"Audio path: {audio_path}")
    print(f"output video name: {video_out_path}")
    return video_out_path

# if __name__ == "__main__":
#     main()
