# LLMs from scratch

This is an in-progress repository to gather training data for, train, and evaluate an LLM from scratch. The initial goal is to create an autoregressive transformer model that can generate English text that sounds _somewhat_ reasonable. I hope to document my process and share my results here. Note that I have no clue what I'm doing, so this is a learning experience for me as well.

For my first implementation, I'll skip gathering training data, and I'll instead use the [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext) from Hugging Face.  

I'm implementing this in Rust, but I plan to reimplement the relevant parts using CUDA for GPU acceleration.

## Project structure
- `src/` contains the Rust source code
- `scripts/` contains scripts to help with the project (probably will end up being mostly Jupyter notebooks for data analysis)

## References
- TODO