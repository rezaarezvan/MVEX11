<div align="center">
<h3>
  claudeslens
</h3>

[![logo](docs/logo.png)](https://github.com/rezaarezvan/MVEX11)

</div>

# MVEX11
Bachelor's thesis in Mathematics - MVEX11

### Setup
```
python3 -m pip install -e .
```

To download the SODA10M dataset, run the following command:
```
python3 extra/soda.py
```

To download the pretrained weights, run the following command:
```
python3 extra/weights.py
```

### Testing
```
python3 -m pip install -e '.[testing]'
python3 -m pytest
```

### Citation

If you'd like to cite my thesis, you can use the following BibTeX:

```bibtex
@misc{claudeslens
      title={ClaudesLens: Uncertainty Quantification in Computer Vision Models}, 
      author={Mohamad Al Shaar and Nils Ekström and Gustav Gille and Reza Rezvan and Ivan Wely},
      year={2024},
      eprint={2406.13008},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.13008}, 
}
```
