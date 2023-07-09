FROM python:3.9

COPY . /

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
	apt-get install git-lfs -y && \
	git lfs install && \
	git clone https://huggingface.co/microsoft/xclip-base-patch16-zero-shot && \
	rm -r xclip-base-patch16-zero-shot/.git/

CMD ["python", "gradio_app.py"]
