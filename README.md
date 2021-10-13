# PyFactorGraph

The primary purpose of this is so that people working in Python can avoid having
to re-write different data providers for each project and data set they work on.
The factor graph object and functions provided here allow for {reading, writing,
accessing} of measurements and variables.

We provide examples for working with the Extended Factor Graph format (`.fg`)
and pickled `FactorGraphData` objects (`.pickle`) in the `/examples` directory.

Also potentially useful, auto-documentation of this code can be found in the
`/docs` directory.

## Getting Started

Installing this package is quick and easy!

```bash
cd ~/PyFactorGraph
pip install .
```

Ta-da you should be ready to go!

## Contributing

If you want to contribute a new feature to this package please read this brief section.

### Code Standards

Any necessary coding standards are enforced through `pre-commit`. This will run
a series of `hooks` when attempting to commit code to this repo. Additionally,
we run a `pre-commit` hook to auto-generate the documentation of this library to
make sure it is always up to date.

To set up `pre-commit`

```bash
cd ~/PyFactorGraph
pip3 install pre-commit
pre-commit install
```

### Testing

If you want to develop this package and test it from an external package you can
also install via

```bash
cd ~/PyFactorGraph
pip install -e .
```

The `-e` flag will make sure that any changes you make here are automatically
translated to the external library you are working in.
