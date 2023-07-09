# video_search_demo

Given a text query, the app returns the video timestamp that is most likely to contain the content of interest.  
The graph on the top right shows relative probability of interest plotted against timestamp in seconds.  
The "sample length" option determines how many seconds of video will be examined at a time by the model.  A higher value will allow for faster search but at reduced precision and potentially accuracy.  

![screenshot](images/forest.png)  

![screenshot](images/aerial_cars.png)

![screenshot](images/tank_example.png)


## Try it 

1. Clone this repo and switch to project directory
1. Install dependencies: `pip install -r requirements.txt`
2. Launch app: `python gradio_app.py`  
_The first search after running the app will take a bit of time because the model has to be downloaded._  



## Bibtex

```
@article{XCLIP,
  title={Expanding Language-Image Pretrained Models for General Video Recognition},
  author={Ni, Bolin and Peng, Houwen and Chen, Minghao and Zhang, Songyang and Meng, Gaofeng and Fu, Jianlong and Xiang, Shiming and Ling, Haibin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Acknowledgements

X-CLIP implementation by [Microsoft](https://github.com/microsoft/VideoX/tree/master/X-CLIP); code taken from [Huggingface](https://huggingface.co/microsoft/xclip-base-patch32)  

Credits to [Armadas](https://www.youtube.com/watch?v=zCLOJ9j1k2Y) for the aerial video.


### Notes

Tested to work for 3.8 <= python <= 3.11   
Known issue:  random crashes with large video files
