
from math import ceil
from torch import matmul, no_grad



def compute_clip_sim(image_embeds, text_embeds):
    with no_grad():
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = matmul(text_embeds, image_embeds.t()) # * model.logit_scale.exp()  # cosine similarity as logits
        return logits_per_text
    

    
def get_samples(videoreader, num_samples=32, sample_interval=15, start=None):#, end=None):
    ''' start and end are frame indices. 
        sample_interval is in frames. 
    '''
    avail_frames = len(videoreader)
            
    start = start if start else 0
    end = start + num_samples*sample_interval

    if avail_frames < end:
        print("Reducing sample interval to 1")
        sample_interval = 1        
        end = start + num_samples*sample_interval
        if avail_frames < end:
            raise Exception("Video or duration requested too short")

    
#     if total_avail_frames < start + num_samples: 
#         raise Exception("Video or duration requested too short")
#     if total_avail_frames < end: 
#         print("Reducing sample interval to 1")
#         sample_interval = 1
    
    indices = [*range(start, end, sample_interval)]
    print(indices)
    return videoreader.get_batch(indices).asnumpy()



def get_sample_frame_idx(videoreader, sample_length=16, num_frames_per_sample=32):
    '''
    sample_length is in seconds
    return list of n lists of frame indices, where n is num_samples, AKA num_vectors
    '''
    range_per_sample = int(sample_length * videoreader.get_avg_fps())  # range in # of frames
    interval_per_frame_of_sample = ceil(range_per_sample / num_frames_per_sample)
    num_samples = int(len(videoreader) // range_per_sample)
    print('draws one frame every', interval_per_frame_of_sample, 'frames over', range_per_sample, 'frames')
    
    indices = []
    for i in range(0, num_samples):
        _indices = range(i*range_per_sample, (i+1)*range_per_sample, interval_per_frame_of_sample)
        indices.append([*_indices])
    return indices, range_per_sample