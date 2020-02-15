# Definition of Submission container

FROM duckietown/aido3-base-python3:daffy

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN rm -rf /workspace; mkdir /workspace

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY solution.py /workspace
COPY wrappers.py /workspace
COPY controller.py /workspace
COPY utils.py /workspace
COPY submissionModel.py /workspace
COPY models.py /workspace
COPY networks.py /workspace
COPY action_invariance.py /workspace
COPY latest_net_G.pth /workspace
COPY VanillaCNN_1579294019.6894116_lr_0.001_bs_16_dataset_sim_totepo_200final.pt /workspace
COPY ConvSkip.pth /workspace

# we make the workspace our working directory
WORKDIR /workspace

ENV DISABLE_CONTRACTS=1

RUN python -c "import solution; import wrappers; import models"

# let's see what you've got there...
CMD python solution.py
