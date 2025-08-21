# SlimPajama Dataset Scripts

These scripts facilitates the processing of the "SlimPajama" dataset, a
comprehensive and extensively deduplicated dataset designed for training large
language models (LLMs). SlimPajama represents a refined version of the 1.2T
token RedPajama dataset, reduced to 627B tokens by eliminating duplicates and
low-quality data.

## Dataset Overview

SlimPajama is the result of meticulous cleaning and deduplication of the
RedPajama dataset, focusing on maximizing data quality and computational
efficiency.

### Features

- **627B tokens**: A significant reduction from the original 1.21T tokens in RedPajama, focusing on quality over quantity.
- **Extensive Deduplication**: Leveraging MinHashLSH for efficient deduplication at the trillion-token scale.
- **Open Source**: Available under the Apache 2.0 license, with tools for dataset replication or preprocessing from scratch.

### Dataset Composition

SlimPajama consists of jsonl files structured with text content and metadata
indicating the original RedPajama data source set (e.g., CommonCrawl, GitHub,
Books).

## Getting Started

### Download and Process Dataset

1. **Clone the repository** containing the scripts for dataset processing from the Cerebras GitHub page.

```bash
bash get_dataset.sh
```

2. **Run tokenizatoin script**

```bash
python3 prepare.py input.txt --method tiktoken
```

### Utilization

After processing, SlimPajama is ready for use in training large language models.

## References

- [SlimPajama-627B Blogpost](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
- [SlimPajama Huggingface Page](https://huggingface.co/datasets/cerebras/SlimPajama-627B)

## Citation

To cite SlimPajama, the following is provided (as is shared on the project's Hugging Face page):

bibtex
```
@misc{cerebras2023slimpajama,
  author = {Soboleva, Daria and Al-Khateeb, Faisal and Myers, Robert and Steeves, Jacob R and Hestness, Joel and Dey, Nolan},
  title = {{SlimPajama: A 627B token cleaned and deduplicated version of RedPajama}},
  month = June,
  year = 2023,
  howpublished = {\url{https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama}},
  url = {https://huggingface.co/datasets/cerebras/SlimPajama-627B},
}
```

