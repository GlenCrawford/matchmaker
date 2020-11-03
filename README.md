# Matchmaker

Matchmaker is a machine-learning project to perform similarity matching using a dataset of OkCupid dating profiles.

Given your basic details (such as your sex and orientation), and more interesting questions (such as your religion, education, use of drugs, etc), Matchmaker will return the best possible matches from the dataset of approximately 60,000 OkCupid profiles. The former questions are matched using logic rules (e.g. a straight woman's candidates will be filtered down to non-gay men). The latter questions (religion, pets, children, etc) are run through a [k-nearest neighbors model](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). This means that you and the profiles in the dataset will be plotted within an n-dimensional space (think a graph with many axes) along with the rest of the dating profiles, and your matches will be the ones nearest to you in the space, where "nearest" effectively means similarity. Hence the name of the algorithm: "Nearest Neighbors".

Weighting is applied appropriately so that differences on some dimensions will have more of an influence on similarity than differences on others. For example, a 28-year-old non-smoker would probably be more averse to dating a 27-year-old smoker than they would be to dating a 26-year-old non-smoker. And a vegan is probably more willing to date a vegetarian than they are to date an omnivore, since the former is "nearer" to the latter. Note that these weights are calibrated according to my own biases!

![Screenshot of matches displayed in the provided Django app.](docs/django_web_app.png?raw=true "Screenshot of matches displayed in the provided Django app.")

From a technical perspective, the entire thing is done in [Python](https://www.python.org/) with the [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/) and [scikit-learn](https://scikit-learn.org/) packages. The model is run in real-time; no pre-calculation or pre-training, so it will take 3 - 4 seconds to run (based on the performance on my laptop).

Note to self: Next time do something easier and less...controversial. The training data has, for example, 217 unique values for ethnicity and 45 for religion, including "radical agnostics", which I'm pretty sure is an oxymoron.

## Requirements

Developed with Python version 3.8.6.

See `requirements.txt` for packages and versions (and below to install).

## Setup

Clone the Git repo.

Install the dependencies:

```bash
$ pip install -r requirements.txt
```

The dataset has been included in the `data` directory as chunked files (don't worry, it's anonymized), so no need to get it yourself. See below for copyright and attribution.

## Run

### As a command-line tool

```bash
$ python matchmaker.py
```

Help output:

```bash
$ python matchmaker.py --help
usage: matchmaker.py [-h] [--matches MATCHES_TO_RETRIEVE] [--force-training]

K-Nearest Neighbors machine learning model to find the best matches within a set of OkCupid profiles.

optional arguments:
  -h, --help            show this help message and exit
  --matches MATCHES_TO_RETRIEVE
                        The number of matching profiles to find (default: 40).
  --force-training      Train the model even if a previously trained and saved model can be loaded and used (default: false).
```

As the command-line tool is mainly for development and testing use it is hard-coded with my profile data. That is easily editable in the script file itself; you will see the override line when you open it up.

### As a Django web app

Note that the [Django](https://www.djangoproject.com/) app is a very simple wrapper around the same model and logic that the CLI uses. It just provides an easy-to-use form and nice UI, and delegates the work to the main package outside of of the web app.

It is [deployed to Heroku](https://matchmaker-demo.herokuapp.com/) or you can boot the server yourself:

```bash
$ python manage.py runserver
```

And then open [http://localhost:8000/](http://localhost:8000/).

## Test

Unit testing is in place where appropriate, such as for data preprocessing, calculating match scores, etc. The tests are implemented with Python's [unittest](https://docs.python.org/3/library/unittest.html) standard library.

```bash
$ python -m unittest discover -v
```

## Dataset credits

The dataset used in this project was obtained from [Larxel on Kaggle](https://www.kaggle.com/andrewmvd/okcupid-profiles). In turn, that dataset was sourced from [Albert Y. Kim's GitHub repository](https://github.com/rudeboybert/JSE_OkCupid) which was created for the publication [OkCupid Profile Data for Introductory Statistics and Data Science Courses](http://www.amstat.org/publications/jse/v23n2/kim.pdf) (Journal of Statistics Education, July 2015, [Volume 23, Number 2](http://www.amstat.org/publications/jse/contents_2015.html)) by Albert Y. Kim and Adriana Escobedo-Land:

> We present a data set consisting of user proﬁle data for 59,946 San Francisco OkCupid users (a free online dating website) from June 2012. The data set includes typical user information, lifestyle variables, and text responses to 10 essay questions.
