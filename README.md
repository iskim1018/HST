
# HST(HyperSphere Tree) Vector DB Indexing Algorithm

Welcome to the repository for the HST (Hierarchical Spherical Tree) vector DB indexing algorithm. This repository contains tools and initial experiments for the HST indexing algorithm. The primary script provided is `HST_test.py`, which allows you to test and experiment with the HST algorithm.

## Usage

The `HST_test.py` script provides various options to configure and run your experiments. Below is the help message outlining its usage and available options:

```plaintext
Usage: HST_test.py [<options>]
   <options>
   -h: help(this message)
   -q <query type>: rank:<rank>, nn:<dist>, pn:<dist_threshold>
   -c <max child in HS>
   -d <vector dimension>: vector dimension
   -n <# of vectors>: default 100
   -r <vector range format>: eg: -1,1 default (-1, 1)
   -s <path>: save HST
   -l <path>: load HST
   -S: setting seed for numpy random
   -m: distance metric, cosine, euclidean(default)
   -v <option>: verbose output, options: stvV
        s: summary, t: tree, v: vector, V: vector data
```

### Options

- **-h**: Display the help message.
- **-q <query type>**: Specify the query type. Options include:
  - `rank:<rank>`: Perform a ranking query.
  - `nn:<dist>`: Perform a nearest neighbor query.
  - `pn:<dist_threshold>`: Perform a proximity neighbor query.
- **-c <max child in HS>**: Set the maximum number of children in the hierarchical structure.
- **-d <vector dimension>**: Specify the vector dimension.
- **-n <# of vectors>**: Set the number of vectors (default is 100).
- **-r <vector range format>**: Set the vector range format (e.g., `-1,1` with default `(-1, 1)`).
- **-s <path>**: Save the HST to the specified path.
- **-l <path>**: Load the HST from the specified path.
- **-S**: Set the seed for numpy random.
- **-m**: Choose the distance metric, either `cosine` or `euclidean` (default is `euclidean`).
- **-v <option>**: Enable verbose output with the following options:
  - `s`: Summary
  - `t`: Tree
  - `v`: Vector
  - `V`: Vector data

## Getting Started

To get started with experimenting on the HST vector DB indexing algorithm, clone the repository and navigate to the project directory:

```sh
git clone https://github.com/cezanne/HST
cd HST
```

### Running the Script

You can run the `HST_test.py` script with various options to configure your experiment. For example:

```sh
python HST_test.py -d 128 -n 1000 -r -1,1 -m cosine
```

This command sets the vector dimension to 128, number of vectors to 1000, vector range from -1 to 1, and uses the cosine distance metric.

## Contributing

We welcome contributions to improve and extend this project. Please feel free to fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or feedback, please open an issue on GitHub or contact us at promptech@promptech.co.kr.

Thank you for using HST Vector DB Indexing Algorithm!
