import streamlit as st
import pandas as pd
import numpy as np


def total_score_by_player(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the total score for each player.

    Args:
        data (pd.DataFrame): The quiz results data.

    Returns:
        tuple: A tuple containing an array of player names and their corresponding total scores.
    """
    players = np.unique(data['PlayerName'].values)
    scores = np.array([data[data['PlayerName'] == player]['Score'].sum() for player in players])
    return players, scores

def average_score_by_difficulty(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the average score for each difficulty level.

    Args:
        data (pd.DataFrame): The quiz results data.

    Returns:
        tuple: A tuple containing an array of difficulty levels and their corresponding average scores.
    """
    difficulties = np.unique(data['Difficulty'].values)
    avg_scores = np.array([data[data['Difficulty'] == difficulty]['Score'].mean() for difficulty in difficulties])
    return difficulties, avg_scores

def cumulative_score_by_player(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Calculates the cumulative sum of scores for each player.

    Args:
        data (pd.DataFrame): The quiz results data.

    Returns:
        dict: A dictionary where the keys are player names and values are arrays of cumulative scores.
    """
    players = np.unique(data['PlayerName'].values)
    cumsum_data = {}
    
    for player in players:
        player_data = data[data['PlayerName'] == player].sort_values(by='Difficulty')
        cumsum_data[player] = np.cumsum(player_data['Score'].values)
    
    return cumsum_data

def main() -> None:
    """
    Main function to display the Streamlit app with visualizations of quiz results.
    """
    data = pd.read_csv('quiz_results.csv')

    players, total_scores = total_score_by_player(data)
    st.subheader('Total Score by Player')
    st.bar_chart(pd.Series(total_scores, index=players))

    difficulties, avg_scores = average_score_by_difficulty(data)
    st.subheader('Average Score by Difficulty')
    st.bar_chart(pd.Series(avg_scores, index=difficulties))

    cumsum_data = cumulative_score_by_player(data)
    st.subheader('Cumulative Score by Player')
    
    for player, cumsum_scores in cumsum_data.items():
        st.line_chart(pd.Series(cumsum_scores, name=player), x_label=f'{player} Questions', y_label='Cumulative Score')

    st.write("Raw Data:")
    st.dataframe(data)

if __name__ == "__main__":
    main()

