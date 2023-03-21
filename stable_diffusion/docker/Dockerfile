ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${FROM_IMAGE_NAME}
# TODO(ahmadki): use upstream pytorch image

# tzdata
ENV TZ="Etc/UTC"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# apt dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  tzdata -y

# install LDM
COPY . /diffusion
RUN cd /diffusion && \
    pip install --no-cache-dir -r requirements.txt

# install xformers (--no-deps otherwise xformers would uninstall container pytorch)
# TODO: build xformers from source
# RUN pip install --no-cache-dir --no-deps xformers==0.0.16
