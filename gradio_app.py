
from numpy import arange
import matplotlib.pyplot as plt 

from torch import cat, save
from transformers import AutoProcessor as AP, AutoModel as AM

from decord import VideoReader, cpu
from tqdm import trange
import gradio as gr

from utils import compute_clip_sim, get_sample_frame_idx



def search(video, query, sample_length=32):
    model_name = "microsoft/xclip-base-patch16-zero-shot"
    processor, model = AP.from_pretrained(model_name), AM.from_pretrained(model_name)
    
    vr = VideoReader(video, ctx=cpu(0))
    idx, range_per_sample = get_sample_frame_idx(vr, sample_length)

    video_features = []
    for i in trange(0, len(idx)):
        frames = vr.get_batch(idx[i]).asnumpy()
        video_features.append(
            model.get_video_features(**processor(videos=list(frames), return_tensors="pt"))
        )
    
    text_features = model.get_text_features(**processor(text=[query], return_tensors="pt", padding=True))
    logits = compute_clip_sim(cat(video_features), text_features)
    
    sample_start_time = arange(len(idx)) * range_per_sample / vr.get_avg_fps()
    fig = plt.figure(); plt.plot(sample_start_time, *logits); plt.grid()#; plt.savefig('temp.png')
    
#    return f'Content matching query found near {idx[logits.argmax()][0] // vr.get_avg_fps()}s'
    return fig, f'Content matching query found near {idx[logits.argmax()][0] // vr.get_avg_fps()}s'
#    return 'temp.png', f'Content matching query found near {idx[logits.argmax()][0] // vr.get_avg_fps()}s'



demo = gr.Interface(
    search, 
    inputs=['video', 'text'], 
    # inputs=[gr.Video(), gr.Textbox(), gr.Number()], 
#    outputs=["text"], 
    outputs=['plot', 'text'], 
#    outputs=[gr.outputs.Image(type='filepath'), "text"], 
    examples=[
        ["videos/aerial.mp4", "moving cars"], 
        ["videos/aerial_small.mp4", 'moving cars']
        # ["videos/aerial.mp4", "moving cars", 16], 
        # ["videos/aerial_small.mp4", 'moving cars', 32]
    ], 
#    cache_examples=True
)



if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0")
