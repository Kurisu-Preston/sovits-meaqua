import io

import gradio as gr
import librosa
import numpy as np
import soundfile
import torch
from inference.infer_tool import Svc
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

model_name = "pth/meaqua.pth"
config_name = "configs/config.json"

svc_model = Svc(model_name, config_name)
sid_map = {
    "aqua": 0,
    "mea": 1
}


def infer(sid, input_audio, vc_transform):


    # if audio_upload is not None:
    #     input_audio = audio_upload
    # elif audio_record is not None:
    #     input_audio = audio_record
    # else:
    #     return "你需要上传wav文件或使用网页内置的录音！", None

    if input_audio is None:
        return "请上传音频", None
    sampling_rate, audio = input_audio
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    out_wav_path.seek(0)

    sid = sid_map[sid]
    out_audio, out_sr = svc_model.infer(sid, vc_transform, out_wav_path)
    _audio = out_audio.cpu().numpy()
    return "Success", (48000, _audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""
                此为上传文件推理用GUI，需要录音变声功能请使用变声器功能\n
                目前仅提供mea和aqua两种音色，采样率48k，对显存要求较高\n
                代码参考：@innnky, @IceKyrin\n
                代码修改：@ChrisPreston（我）\n
                模型训练：@ChrisPreston（我）\n
                音源：aqua: Aqua Ch. 湊あくあ https://www.youtube.com/@MinatoAqua, mea: 神楽めあ / KaguraMea https://www.youtube.com/channel/UCWCc8tO-uUl_7SJXIKJACMw\n
                模型使用协议（重要）：\n
                1.请勿用于商业目的\n
                2.请勿用于会影响主播本人的行为（比如冒充本人发表争议言论）\n
                3.请勿用于血腥、暴力、性相关、政治相关内容\n
                4.请勿将该模型用于底模进行训练，如果需要训练用底模请使用移除无关权重的通用底模@innnky\n
                5.非个人使用场合请注明模型作者@ChrisPreston（我）以及sovits原项目作者@innnky/@Rcell\n
                6.允许用于个人娱乐场景下的游戏语音、直播活动，不得用于低创内容，用于直播前请与本人联系\n
                联系方式：电邮：kameiliduo0825@gmail.com, b站：https://space.bilibili.com/18801308\n
                免责声明：由于使用本模型造成的法律纠纷本人概不负责""")
            sid = gr.Dropdown(label="音色", choices=["aqua", "mea"], value="aqua")
            # record_input = gr.Audio(source="microphone", label="录制你的声音")
            upload_input = gr.Audio(source="upload", label="上传音频，过长容易爆显存，8G显存推荐35s以内")
            vc_transform = gr.Number(label="变调（整数，可以正负，升半音）", value=0)
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(infer, [sid,  upload_input, vc_transform], [vc_output1, vc_output2])

    app.launch()