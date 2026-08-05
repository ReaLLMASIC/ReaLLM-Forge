#!/bin/bash

# convert flac to wav
ffmpeg -i "${1}" "${1%%.flac}.wav"

