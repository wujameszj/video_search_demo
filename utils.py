from torch import matmul, no_grad

def compute_clip_sim(image_embeds, text_embeds):
    with no_grad():
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = matmul(text_embeds, image_embeds.t()) # * model.logit_scale.exp()  # cosine similarity as logits
        return logits_per_text
    

    
def get_samples(videoreader, num_samples=32, sample_interval=15, start=None):#, end=None):
    ''' start and end are frame indices
        
#        Returns: an np.array of dimensions num_ x num 
    '''
    total_avail_frames = len(videoreader)
    print('total available frame:', total_avail_frames)
    
    if total_avail_frames < num_samples: raise Exception("Video too short")
    if total_avail_frames < num_samples*sample_interval: 
        print("Reducing sample interval to 1")
        sample_interval = 1
        
    start = start if start else 0
    end = start + num_samples*sample_interval
    
    indices = [*range(start, end, sample_interval)]
    print(indices)
    return videoreader.get_batch(indices).asnumpy()