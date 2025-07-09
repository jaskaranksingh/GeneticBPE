# GeneticBPE

GeneticBPE is a biologically-informed tokenization framework that preserves structural motifs in biological sequences while maintaining compression efficiency. This implementation is based on the paper "GeneticBPE: Motif-Preserving Tokenization for Robust miRNA Modeling".

## Features

- Motif-aware tokenization for biological sequences
- Preservation of conserved regions and seed motifs
- Efficient compression while maintaining biological relevance
- Support for miRNA sequence processing
- Cross-species generalization capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from genetic_bpe import GeneticBPE

# Initialize tokenizer
tokenizer = GeneticBPE(
    vocab_size=512,
    motif_weight=2.5,
    penalty_weight=10.0
)

# Train tokenizer
tokenizer.train(corpus_path="path_to_sequences.txt")

# Tokenize sequences
tokens = tokenizer.encode("AUGCAUGCAUGC")
```

## Project Structure

```
genetic_bpe/
├── genetic_bpe/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── motif_bank.py
│   └── utils.py
├── tests/
│   └── test_tokenizer.py
├── examples/
│   └── basic_usage.py
├── requirements.txt
└── README.md
```

## License

MIT License

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{geneticbpe2024,
  title={GeneticBPE: Motif-Preserving Tokenization for Robust miRNA Modeling},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2024}
}
``` 