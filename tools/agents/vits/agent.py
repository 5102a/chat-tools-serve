from scipy.io import wavfile
import logging
import os
import wave
import numpy as np
import torch
from torch import no_grad, LongTensor
from . import commons, utils, models
from .text import text_to_sequence

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def tts_fn(text, speaker, language, speed):
    if language is not None:
        text = language_marks[language] + text + language_marks[language]
    speaker_id = speaker_ids[speaker]
    stn_tst = get_text(text, hps, False)
    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    return "Success", (hps.data.sampling_rate, audio)


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data


class Vits:
    def __init__(self, model_path: str = "./model/G_39500.pth", config_path: str = "./model/config.json") -> None:
        global model, hps, speakers, speaker_ids
        p = os.path.dirname(__file__)
        hps = utils.get_hparams_from_file(
            os.path.abspath(os.path.join(p, config_path)))

        model = models.SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = model.eval()

        _ = utils.load_checkpoint(os.path.abspath(
            os.path.join(p, model_path)), model, None)
        speaker_ids = hps.speakers
        speakers = list(hps.speakers.keys())

    def run(self, query: str):
        text, output = tts_fn(query, speakers[0], '简体中文', 1)
        wavfile.write(
            './output.wav', output[0], convert_to_16_bit_wav(output[1]))
        return './output.wav'
