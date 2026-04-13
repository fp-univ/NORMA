# NORMA — NORmative Mineral Analysis

Python GUI for normative mineral quantification from bulk-rock oxide geochemistry, with equivalent dissolution depth calculation.

## Description

NORMA performs mass-balance inversion of bulk-rock oxide wt% data to estimate mineral assemblage proportions. Two fitting modes are available:
- **Best Fit**: unconstrained NNLS (non-negative least squares)
- **Normative**: greedy phase selection minimising RMSE

An optional second step computes equivalent dissolution depths from dissolved cation concentrations (mol).

## Requirements
numpy
matplotlib

Python standard library: `tkinter`, `re`, `math`, `csv`

Tested with Python 3.10+. Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

Launch from Spyder or any Python environment:

```bash
python code.py
```

The application opens maximised. No command-line arguments required.

## Input format

Bulk-rock oxide data can be entered manually or imported via CSV (use the built-in Export/Import function to generate a template).

## Authors

Flora Parrotin — flora.parrotin@univ-rennes.fr  
Code cleaning & refactoring: Claude (Anthropic)

## License

MIT
