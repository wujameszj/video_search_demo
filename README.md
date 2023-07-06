# video_search_demo

Given a text query, the app returns the video timestamp that is most likely to contain the content of interest.  
The graph on the top right shows relative probability of interest plotted against timestamp in seconds.  
The "sample length" option determines how many seconds of video will be examined at a time by the model.  A higher value will allow for faster search but at reduced precision and potentially accuracy.  

![screenshot](images/gradio_app.png)  

![screenshot](images/Tank%20example.png)


## Try it 

1. Clone this repo and switch to project directory
1. Install dependencies: `pip install -r environments.txt`
2. Launch app: `python gradio_app.py`
