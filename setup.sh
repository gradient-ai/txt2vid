apt-get update && apt-get install git-lfs -y
git clone https://github.com/modelscope/modelscope.git 
pip install open_clip_torch pytorch-lightning yt-dlp moviepy
git clone https://github.com/coqui-ai/TTS
pip install -r TTS/requirements.txt
pip install -r modelscope-text-to-video-synthesis/requirements.txt
pip install ffmpeg --upgrade
pip install modelscope==1.4.2
pip install -U huggingface_hub
pip install gradio
git-lfs clone https://huggingface.co/spaces/fffiloni/spectrogram-to-music
cd spectrogram-to-music
pip install -r requirements.txt
cd ..
git clone https://github.com/coqui-ai/TTS
cd TTS
pip install -e .[all,dev,notebooks]  # Select the relevant extras
pip install -U transformers
cd ..
git-lfs clone https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis
wget https://huggingface.co/spaces/fffiloni/spectrogram-to-music/raw/main/spectro.py
mkdir outs
