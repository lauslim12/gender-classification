# Gender Classification

This is a small application made for classifying genders based on a given face. Technologies used are TensorFlow Lite, Google Colaboratory for the Jupyter Notebook, Anaconda, and Python.

## Requirements

* Anaconda
* Google Colaboratory
* Python
* TensorFlow Lite

## Project Structure

The project structure is really simple. There are a few things that you should keep in mind when using this repository.

* `models` folder is used to store the trained model and the labels file, generated from TensorFlow Lite.
* `notebooks` folder is used to store the Jupyter Notebook file. The file is then run in Google Colaboratory.
* `environment.yml` file is used to store the environment dependencies.
* `main.py` file is the application starting point.

## How to Use

This program is developed with Python and Jupyter Notebook. In order to use this program, do not forget to `git clone repository_link` first!

### Application

- First off, install the Anaconda environment first. I am running this on Windows, so there might be some different changes, but it should work fine.

```bash
  $ conda --version
  $ conda env create -f environment.yml
  $ conda activate gender-classification
```

- Alternatively, if you want to install the dependencies manually, you could opt out of Anaconda and install the following packages.

```bash
  $ pip install tensorflow
  $ pip install matplotlib
  $ apt-get install python-tk
```

- The reason that I installed `matplotlib` is because `matplotlib` does not naturally come with TensorFlow. After successfully installing all of the dependencies, switch to the repo directory.

```bash
  $ cd gender-classification
```

- Then, run the application by using the following command.

```bash
  $ python main.py
```

- The Tkinter application should start. Now, feel free to try uploading your photo by clicking on the 'Classify your Face' button, then investigate it if the gender is correct! The output will be in the form of `x% male, y% female`.

### Jupyter Notebook

- To use the Jupyter Notebook, simply clone this repository, and open the notebook in Google Colaboratory.

## Notes

The application is still imperfect and some errors or wrong classification may occur. The source code is formatted using `autopep8` and linted using `pylint`.