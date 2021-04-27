FROM tensorflow/tensorflow:2.4.0

COPY . /h_to_d/
WORKDIR /h_to_d

# preventing error:
# ImportError: libGL.so.1: cannot open shared object file...
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


CMD ["flask", "run", "--host=0.0.0.0"]