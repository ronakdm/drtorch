# Diropt

Diropt is a library for distributionally robust optimization in PyTorch.

## Installation

Diropt requires PyTorch >= 1.6.0. Please go [here](https://pytorch.org/) for instructions 
on how to install PyTorch based on your platform and hardware.
Once Pytorch is installed you can install Diropt by running the following on the command line from the
root folder:
```
pip install -e .
```

Additional dependencies to run the example in `examples/train_fashion_mnist.ipynb` can be installed using `pip install -e .[examples]`. To build the docs, additional dependencies can be run using `pip install -e .[docs]`.

## Quickstart

A detailed quickstart example is present in the docs `docs/source/quickstart.ipynb`. At a glance, the functionality is a follows. First, we construct a function that inputs a vector of losses and returns a probability distribution over elements in this loss vector.
```
>>> from diropt import make_spectral_risk_measure, make_extremile_spectrum
>>> spectrum = make_superquantile_spectrum(batch_size, 2.0)
>>> compute_sample_weight = make_spectral_risk_measure(spectrum, penalty="chi2", shift_cost=1.0)
```
Assume that we have computed a vector of losses based on a model output in PyTorch. We can then use the function above and back propagate through the weighted sum of losses.
```
>>> x, y = get_batch()
>>> logits = model(x)
>>> losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
>>> with torch.no_grad():
>>>     weights = compute_sample_weight(losses.cpu().numpy()).to(device)
>>> loss = weights @ losses
>>> loss.backward()
```

## Documentation

The documentation is available [here](https://ronakdm.github.io/drtorch/).

## Contributing

If you find any bugs, please raise an issue on GitHub.
If you would like to contribute, please submit a pull request.
We encourage and highly value community contributions.

## Citation

If you find this package useful, or you use it in your research, please cite:

    @inproceedings{mehta2023stochastic,
      title={{Stochastic Optimization for Spectral Risk Measures}},
      author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Liu, Lang and Harchaoui, Zaid},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      pages={10112--10159},
      year={2023},
      organization={PMLR}
    }

## Acknowledgments

## License




