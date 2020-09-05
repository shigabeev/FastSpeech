import torch
import torch.nn as nn
import argparse
import numpy as np
import os

import hparams as hp
import audio
import utils
import text
import model as M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path), map_location=device)['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).to(device).long()
    src_pos = torch.from_numpy(src_pos).to(device).long()

    with torch.no_grad():
        _, mel = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

class TTS:

    def __init__(self, step=135000):
        self.vocoder = utils.get_melgan()
        self.model = get_DNN(step)

    def get_spec(self, string, alpha=1.0):
        text_norm = text.text_to_sequence(string, hp.text_cleaners)
        with torch.no_grad():
            mel, mel_cuda = synthesis(self.model, text_norm, alpha)
        return mel

    def inv_spec(self, mel):
        waveform = self.vocoder(mel.unsqueeze(0)).squeeze().detach().cpu().numpy()
        return waveform

    def run(self, string, alpha=1.0):
        mel = self.get_spec(string, alpha)
        waveform = self.inv_spec(mel)
        return waveform


if __name__ == "__main__":
    # Test

    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str)
    parser.add_argument('--step', type=int, default=135000)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    tts = TTS(args.step)
    waveform = tts.run(args.text, args.alpha)
    if not os.path.exists("results"):
        os.mkdir("results")
    audio.tools.save_audio(waveform, "results/" + str(hash(args.text)) + ".wav")

