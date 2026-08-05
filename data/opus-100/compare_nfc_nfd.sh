#!/bin/bash

# bash prepare_nfc_and_nfd_dataset_splits.sh input.txt

for tokenization in "char_bpe"; do
  for type in "nfc" "nfd"; do
    for vocab_size in "2000" "3000" "4000" "5000"; do
      python3 prepare.py -t p90_"$type".txt -v p10_"$type".txt --method "$tokenization" --vocab_size "$vocab_size" -s -S "$type"_"$vocab_size" -T
      mv char_bpe_vocab.json "${tokenization}_${type}_${vocab_size}/"
    done
  done
done
