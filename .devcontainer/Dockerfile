FROM python:alpine
# Patch
RUN apk upgrade --no-cache

# Install compiler requirements for python dependencies
RUN apk add --no-cache gcc python3-dev musl-dev linux-headers

# Install other userful tooks
RUN apk add --no-cache git

# Install python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
