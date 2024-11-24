import os
from quiz import MovieQuizGenerator
from typing import List, Tuple

def display_question(question: dict, question_number: int) -> int:
    """
    Display a quiz question with options and prompt the user to select an answer.

    Args:
        question (dict): The question dictionary containing the question text and options.
        question_number (int): The current question number.

    Returns:
        int: The selected answer option (index 0-3).
    """
    print(f"\nQuestion {question_number} (Difficulty: {question['difficulty']}):")
    print(question['question'])
    print("\nOptions:")
    for i, option in enumerate(question['options'], 1):
        print(f"{i}. {option}")
    return int(input("\nYour answer (1/2/3/4): ")) - 1

def calculate_score(difficulty: str, is_correct: bool) -> int:
    """
    Calculate the score based on the question difficulty and whether the answer was correct.

    Args:
        difficulty (str): The difficulty of the question (easy, medium, or hard).
        is_correct (bool): Whether the player's answer was correct.

    Returns:
        int: The score for the question based on its difficulty and correctness.
    """
    scores = {'easy': 1, 'medium': 2, 'hard': 3}
    return scores[difficulty] if is_correct else 0

def save_results(player_name: str, scores: List[Tuple[int, str]]) -> None:
    """
    Save the quiz results to a CSV file.

    Args:
        player_name (str): The name of the player.
        scores (List[Tuple[int, str]]): A list of tuples where each tuple contains the score and difficulty of each question.
    
    Writes the results to a file named `quiz_results.csv`.
    """
    file_name = "quiz_results.csv"
    
    # Check if file exists and create header if not
    file_exists = os.path.isfile(file_name)
    
    # Open file in append mode
    with open(file_name, 'a' if file_exists else 'w') as f:
        if not file_exists:
            f.write("PlayerName,Score,Difficulty\n")
        # Append results
        for score, difficulty in scores:
            f.write(f"{player_name},{score},{difficulty}\n")
    
    print(f"\nResults saved to {file_name}.")

def main() -> None:
    """
    The main function that runs the quiz, prompts for answers, calculates scores, and saves the results.
    
    It generates questions of different difficulties, displays them, collects answers, and provides feedback to the user.
    The player's results are saved to a CSV file at the end.
    """
    player_name = input("Enter your name: ").strip()
    try:
        num_questions = int(input("Enter the number of questions: "))
    except ValueError:
        print("Invalid input. num_questions is set to default (9).")
        num_questions = 9

    # Initialize the quiz generator
    quiz = MovieQuizGenerator('./selected-dataset/movies.tsv', './selected-dataset/actors.tsv', './selected-dataset/directors.tsv')

    scores = []  # To store results in (Score, Difficulty) format
    
    # Generate questions and prompt the user for answers
    for i in range(num_questions):
        if i % 3 == 0:
            question = quiz.generate_easy_question()
        elif i % 3 == 1:
            question = quiz.generate_medium_question()
        else:
            question = quiz.generate_hard_question()

        try:
            player_answer = display_question(question=question, question_number=i)
            correct = question['options'][player_answer] == question['correct_answer']
            score = calculate_score(question['difficulty'], correct)
            print("Correct!" if correct else f"Wrong! The correct answer was {question['correct_answer']}.")
        except (ValueError, IndexError):
            score = 0
            print(f"Invalid choice! The correct answer was {question['correct_answer']}.")
        
        scores.append((score, question['difficulty']))
    
    # Save results
    save_results(player_name, scores)

if __name__ == "__main__":
    main()

