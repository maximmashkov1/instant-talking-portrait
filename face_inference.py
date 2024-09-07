from face_utils import get_mel_spectrogram, animate_edge_movement, modeled_mouth_to_complete, resample_keypoints, animate_vertices, estimate_f0_and_power
from face_model.diffusion import *
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from time import time
class FaceKeypointModeling:
    def __init__(self,ddim=False):
        self.mouth_model = torch.load("./mouth_model/mouth.ckpt", map_location=torch.device('cpu')).eval().to('cpu')
        mean, std = self.mouth_model.normalization
        self.mouth_morph_targets = np.array([std[self.mouth_model.n_mels:]*target for target in self.mouth_model.get_morph_targets()])
        self.mouth_morph_mean = mean[self.mouth_model.n_mels:].numpy()

        self.unet = torch.load('face_model/face_unet.ckpt', map_location=torch.device('cpu'))
        self.unet.set_device('cpu')
        self.face_diffusion = self.unet.generate_ddim if ddim else self.unet.generate_ddpm

    def inference(self, audio, sr, fps, timesteps=None, face_smoothing=None):
        spectrogram = torch.tensor(get_mel_spectrogram(audio.copy(), sr, n_mels=self.mouth_model.n_mels).T)

        mean, std = self.mouth_model.normalization
        x = (spectrogram-mean[:self.mouth_model.n_mels])/std[:self.mouth_model.n_mels]
    
        x = x.unsqueeze(0)
        output = self.mouth_model.forward(x)
        output = output.squeeze(0).detach().cpu().numpy()
        

        ##########################################

        f0_power=torch.tensor(estimate_f0_and_power(audio.copy(), sr, target_sr=22050)).to(self.unet.device).unsqueeze(0)
        mean,std = self.unet.normalization
        f0_power_normalized = (f0_power-mean[:3])/std[:3]
        t=time()
        #mel=torch.tensor(get_mel_spectrogram(audio, n_mels=25).T).to(self.unet.device).unsqueeze(0)
        #mel=(mel-mean[:25])/std[:25]
        sequence=self.face_diffusion(f0_power_normalized.to(torch.float32)).squeeze(0).cpu().detach().numpy()
        print("Diffusion time: ",time()-t)
        print("SHAPE",sequence.shape)
        if face_smoothing != None:
            sequence[0], sequence[1] = gaussian_filter(sequence[0], sigma=face_smoothing), gaussian_filter(sequence[1], sigma=face_smoothing)

        sequence=torch.tensor(sequence.transpose(1,0).reshape(-1, 2))
        sequence = (mean[-2:]+(sequence*std[-2:])).numpy()

        if fps != self.mouth_model.original_fps:
            target_length = int(output.shape[0] * fps/self.mouth_model.original_fps)
            output = resample_keypoints(output, target_length)
            sequence = resample_keypoints(sequence, target_length)

        return output, sequence #N, 25 | N, 2



if __name__=='__main__':
    from model_mouth.mouth_dataset import SequenceDataset
    dataset = SequenceDataset("F:/datasets/AVSpeech/processed_25_fb_dataset.pickle", batch_size=512, val_size=1024, n_mels=25)
    animate_vertices(dataset.sequences[400][-2:].reshape(-1, 1, 2).cpu().detach().numpy(),30)

    """
    unet = torch.load('face_ddim.ckpt').to('cuda')
    mel = get_mel_spectrogram('obama.wav', n_mels=40).T
    mel=torch.tensor(mel).to('cuda').unsqueeze(0)

    sequence=unet.generate(mel).transpose(2,1).unsqueeze(0).cpu().detach().numpy()
    sequence=sequence.reshape(-1, 1, 2)
    animate_vertices(sequence,30)
    
    
    from model_mouth.model_mouth import SpeechToLip
    face_model = FaceKeypointModeling()
    
    blends = face_model.inference('audio.wav', 30)
    m_t = face_model.mouth_morph_targets
    base = np.load("model_mouth/base.npy")
    animation = []

    
    for frame in blends:
        morph=np.zeros(25)
        for i, blend in enumerate(frame):
            morph=morph+blend*m_t[i]
        animation.append(morph)
    
    animation = np.array(animation)
    animation += face_model.mouth_morph_mean 
    animation, jaw = modeled_mouth_to_complete(animation, base)
    animate_edge_movement(animation, 30)
    """