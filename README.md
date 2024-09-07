# Instant talking portrait

Simple, fast warp-based talking face from image and audio with no reference footage. Uses https://github.com/mikecokina/puppet-warp for image warping.
Inference time on cpu: around 0.2 seconds per phrase.


###Demo: https://www.youtube.com/embed/7d77oR1cpjY

### Installation:
1- Clone https://github.com/maximmashkov1/instant_talking_portrait
2- Install requirements.txt
3- Download depth_anything_v2_vitb.pth from https://huggingface.co/depth-anything/Depth-Anything-V2-Base/tree/main and place it inside Depth-Anything-V2/checkpoints
4- Start main.py

### Usage:
After defining a character, send audio to be pronounced as shown in example_send_message.py
