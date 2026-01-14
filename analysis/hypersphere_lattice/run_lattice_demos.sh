#!/bin/bash


for demo in ./lattice_demos/*; do
  echo "${demo}"
  bash "${demo}"
done
