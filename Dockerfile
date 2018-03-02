FROM sdp-docker-registry.kat.ac.za:5000/docker-base

MAINTAINER Ludwig Schwardt "ludwig@ska.ac.za"

COPY requirements.txt /tmp/install/requirements.txt
RUN install-requirements.py -d ~/docker-base/base-requirements.txt -r /tmp/install/requirements.txt

# Install the current package
COPY . /tmp/install/katsdpspectrometer
WORKDIR /tmp/install/katsdpspectrometer
RUN python ./setup.py clean && pip install --no-deps . && pip check
