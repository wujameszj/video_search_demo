
from os.path import exists
from pathlib import Path

from numpy import arange
import matplotlib.pyplot as plt 
from torch import cat, load, save
from transformers import AutoProcessor as AP, AutoModel as AM

from decord import VideoReader, cpu
from tqdm import trange
import gradio as gr

from utils import compute_clip_sim, get_sample_frame_idx



def embed(videoreader, idx, model, processor):
    video_features = []
    for i in trange(0, len(idx)):
        frames = videoreader.get_batch(idx[i]).asnumpy()
        video_features.append(
            model.get_video_features(**processor(videos=list(frames), return_tensors="pt"))
        )
    return cat(video_features)



def search(video, query, sample_length=4):    
    model_name = "microsoft/xclip-base-patch16-zero-shot"
    processor, model = AP.from_pretrained(model_name), AM.from_pretrained(model_name)    
    
    vr = VideoReader(video, ctx=cpu(0))
    idx, range_per_sample = get_sample_frame_idx(vr, sample_length)
        
    save_path = f'feature_vectors/{Path(video).stem}_{len(idx)}samples.pt'
    if exists(save_path):
        video_features = load(save_path)
    else:
        video_features = embed(vr, idx, model, processor)
        save(video_features, save_path)

    text_features = model.get_text_features(**processor(text=[query], return_tensors="pt", padding=True))
    logits = compute_clip_sim(video_features, text_features)
    
    sample_start_time = arange(len(idx)) * range_per_sample / vr.get_avg_fps()
    fig = plt.figure(); plt.plot(sample_start_time, *logits); plt.grid()#; plt.savefig('temp.png')
    return fig, f'Content matching query found near {idx[logits.argmax()][0] // vr.get_avg_fps()}s'



demo = gr.Interface(
    search, 
    inputs=['video', 'text', 'number'], 
    outputs=['plot', 'text'], 
    examples=[
        ['videos/Tank.mp4', 'people walking', 4],
        ["videos/aerial.mp4", "moving cars", 16], 
        ["videos/aerial_small.mp4", 'moving cars', 32]
    ]
)



if __name__ == '__main__':
    demo.queue()
    demo.launch(server_name="0.0.0.0")
