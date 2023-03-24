from huggingface_hub import snapshot_download

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import pathlib
import torch
import os, subprocess
from moviepy.editor import *

import torch

from PIL import Image
import numpy as np
from spectro import wav_bytes_from_spectrogram_image

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr

from TTS.api import TTS
device = "cuda"
MODEL_ID = "riffusion/riffusion-model-v1"
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe2 = pipe2.to(device)

def seedTorch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_audio(url, name):
    print(['yt-dlp', '--extract-audio', '--audio-format', 'wav', f'{url}', '--output', f'TTS/audio_samps/{name}.wav'])
    subprocess.run(['yt-dlp', '--extract-audio', '--audio-format', 'wav', f'{url}', '--output', f'TTS/audio_samps/{name}.wav'])
    
def make_video(vid_prompt):
    model_dir = pathlib.Path('/notebooks/modelscope-damo-text-to-video-synthesis')
    pipe = pipeline('text-to-video-synthesis', model_dir.as_posix(),output_video = 'outs/video.mp4')
    test_text = {
            'text': f'{vid_prompt}',
            'output_video_path' : 'outs/video.mp4'
        }
    output_video_path = pipe(test_text,output_video = 'outs/video.mp4')[OutputKeys.OUTPUT_VIDEO]
    return output_video_path

def make_speech(speech_prompt_input, name):
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
    tts.tts_to_file(f'{speech_prompt_input}', speaker_wav=f'/notebooks/TTS/audio_samps/{name}.wav', language="en", file_path="outs/speech.wav")
    return "outs/speech.wav"



spectro_from_wav = gr.Interface.load("spaces/fffiloni/audio-to-spectrogram")

def predict(music_prompt_input, negative_prompt, audio_input, duration):
    if audio_input == None :
        return classic(music_prompt_input, negative_prompt, duration)
    else :
        return style_transfer(music_prompt_input, negative_prompt, audio_input)

def classic(music_prompt_input, negative_prompt, duration):
    if duration == 5:
        width_duration=512
    else :
        width_duration = 512 + ((int(duration)-5) * 128)
    spec = pipe(music_prompt_input, negative_prompt=negative_prompt, height=512, width=width_duration).images[0]
    print(spec)
    wav = wav_bytes_from_spectrogram_image(spec)
    with open("outs/music.wav", "wb") as f:
        f.write(wav[0].getbuffer())
    return 'outs/music.wav'

def style_transfer(music_prompt_input, negative_prompt, audio_input):
    spec = spectro_from_wav(audio_input)
    print(spec)
    # Open the image
    im = Image.open(spec)
    
    
    # Open the image
    im = image_from_spectrogram(im, 1)
   
    
    new_spectro = pipe2(prompt=music_prompt_input, image=im, strength=0.5, guidance_scale=7).images
    wav = wav_bytes_from_spectrogram_image(new_spectro[0])
    with open("outs/music.wav", "wb") as f:
        f.write(wav[0].getbuffer())
    return 'outs/music.wav'

def image_from_spectrogram(
    spectrogram: np.ndarray, max_volume: float = 50, power_for_image: float = 0.25
) -> Image.Image:
    """
    Compute a spectrogram image from a spectrogram magnitude array.
    """
    # Apply the power curve
    data = np.power(spectrogram, power_for_image)

    # Rescale to 0-255
    data = data * 255 / max_volume

    # Invert
    data = 255 - data

    # Convert to a PIL image
    image = Image.fromarray(data.astype(np.uint8))

    # Flip Y
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Convert to RGB
    image = image.convert("RGB")

    return image

    
    
def run(music_prompt_input, speech_prompt_input, vid_prompt, negative_prompt, name, url, duration_input, YT, seed):
    print(f'seed: {seed}, the music prompt: {music_prompt_input}, the speech prompt: {speech_prompt_input}, the vid prompt: {vid_prompt}, neg: (negative_prompt),  name: {name}, duration: {duration_input}, url: {YT}')
    seedTorch(seed)
    vid = make_video(vid_prompt)
    get_audio(YT, name)
    speech_output = make_speech(speech_prompt_input, name)
    music_output= predict(music_prompt_input, negative_prompt, None, duration_input)
    # load the video
    video_clip = VideoFileClip(vid)
    # # load the audio
    music_clip = AudioFileClip('outs/music.wav')
    speech_clip = AudioFileClip('outs/speech.wav')

    new_audioclip = CompositeAudioClip([music_clip, speech_clip])
    video_clip.audio = new_audioclip
    video_clip.write_videofile("outs/final.mp4")
    return 'outs/music.wav', 'outs/speech.wav', 'outs/video.mp4', "outs/final.mp4"
    

with gr.Blocks(css="style.css") as demo:
    with gr.Row(elem_id="center"):
        gr.Markdown("# Full video generation pipeline with Gradient")
    with gr.Row(elem_id = 'description'):
        gr.Markdown(""" To run the pipeline, be sure to run `bash setup.sh` before spinning up this application. \n Fill in the cells below and click **Run the generator!** to see the output.""")
    
    with gr.Column(elem_id="col-container"):        
        music_prompt_input = gr.Textbox(placeholder="a happy childrens tv show theme", value = "a happy childrens tv show theme", label="Musical prompt", elem_id="music-in", interactive = True)
        speech_prompt_input = gr.Textbox(placeholder="oh what a beautiful day!",value = "oh what a beautiful day!", label="What the speaker should say", elem_id="speech-in")
        vid_prompt = gr.Textbox(placeholder="a disney princess dancing in the forest", value = "a disney princess dancing in the forest", label="What should the video be about?", elem_id="vid-in")
        with gr.Row():
            with gr.Column():
                negative_prompt = gr.Textbox(label="Negative prompt")
                name = gr.Textbox(placeholder="alice", value = "alice", label="What is the speakers name? (no spaces)", elem_id="prompt-in")
                url = gr.Textbox(placeholder="https://www.youtube.com/watch?v=Srn0xkXTSgs", value = "https://www.youtube.com/watch?v=Srn0xkXTSgs", label="A youtube video containing the speakers voice", elem_id="url-in", interactive = True, visible = False)
                YT = gr.Textbox(placeholder="https://www.youtube.com/watch?v=Srn0xkXTSgs", value = "https://www.youtube.com/watch?v=Srn0xkXTSgs", label="A youtube video containing the speakers voice", elem_id="url-in", interactive = True)
            with gr.Column():
                duration_input = gr.Slider(label="Duration in seconds", minimum=5, maximum=10, step=1, value=5, elem_id="duration-slider")

                seed = gr.Slider(minimum = 0, maximum = 5000, step = 1, label = 'seed')

            
        send_btn = gr.Button(value="Run the generator!", elem_id="submit-btn")
            
    with gr.Row(elem_id="col-container-2"):
        
        music_output = gr.Audio(type = 'filepath', label="Music output", elem_id="music-out")
        speech_output = gr.Audio(type='filepath', label="Speech output", elem_id="speech-out")
        video_output = gr.Video(type = 'filepath', label = 'Initial video output', elem_id = 'init-vid-out')
        final_video = gr.Video(type = 'filepath', label = 'Final merged video', elem_id = 'final-vid-out')
    with gr.Row():
        gr.Image('assets/logo.png').style(height = 53, width = 125, interactive = False)
        
    send_btn.click(run, inputs=[music_prompt_input, speech_prompt_input, vid_prompt, url, name, negative_prompt, duration_input, YT, seed], outputs=[music_output, speech_output, video_output, final_video])

demo.queue(max_size=250).launch(debug=True, share = True)


