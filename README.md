# imdb-quiz
This project is a Python-based application that generates a quiz based on IMDb movie data. The quiz includes questions of varying difficulty levels, ranging from basic facts about movies to complex relationships between actors, directors, and genres. The project is structured to process large datasets efficiently and provides a user-friendly interface for taking the quiz and viewing results.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Processing](#data-processing)
- [Running the Quiz](#running-the-quiz)
- [Viewing Results](#viewing-results)

## Project Overview

The project consists of several components:

- **Data Processing**: Efficiently processes large IMDb datasets to extract relevant information about movies, actors, and directors.
- **Quiz Generation**: Generates quiz questions of varying difficulty levels.
- **Quiz Interface**: Allows users to take the quiz and provides feedback.
- **Results Visualization**: Uses Streamlit to visualize quiz results.

## Setup

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/arashkarimi1999/imdb-quiz.git
   cd imdb-quiz
   ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download IMDb Dataset**

    Download the IMDb dataset from the [IMDb Datasets page](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset) and place the necessary TSV files in the `imdb-dataset` directory.

## Data Processing

The `preprocess.py` script processes the IMDb dataset to extract relevant information. It filters movies based on a vote threshold, extracts movie details, and processes credits for actors and directors.\
\
To run the data processing:
```bash
python preprocess.py
```
This will generate processed TSV files in the `selected-dataset` directory.

## Running the Quiz
The `main.py` script runs the quiz interface. It generates questions and prompts the user for answers, calculates scores, and saves the results.

To run the quiz:
```bash
python main.py
```
Follow the on-screen instructions to enter your name, select the number of questions, and answer the quiz questions.

## Viewing Results
The `app.py` script uses Streamlit to visualize quiz results. It displays total scores by player, average scores by difficulty, and cumulative scores by player.

To view the results:

Ensure you have the `quiz_results.csv` file generated by `main.py`.

Run the Streamlit app:
```bash
streamlit run app.py
```

This will open a web interface displaying the quiz results.
