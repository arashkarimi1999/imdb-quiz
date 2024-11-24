import pandas as pd
import random
from typing import List, Dict

class MovieQuizGenerator:
    def __init__(self, movies_path: str, actors_path: str, directors_path: str):
        # Read TSV files
        self.movies_df = pd.read_csv(movies_path, sep='\t')
        self.actors_df = pd.read_csv(actors_path, sep='\t')
        self.directors_df = pd.read_csv(directors_path, sep='\t')
        
        # Convert string representations of lists to actual lists
        self.movies_df['genres'] = self.movies_df['genres'].apply(eval)
        self.movies_df['actors'] = self.movies_df['actors'].apply(eval)
        self.movies_df['directors'] = self.movies_df['directors'].apply(eval)
        self.actors_df['knownForTitles'] = self.actors_df['knownForTitles'].apply(eval)
        self.directors_df['knownForTitles'] = self.directors_df['knownForTitles'].apply(eval)

    def generate_easy_question(self) -> Dict:
        """Generate easy questions - Basic facts about movies"""
        question_types = [
            self._generate_year_question,
            self._generate_genre_question,
            self._generate_director_question
        ]
        return random.choice(question_types)()

    def generate_medium_question(self) -> Dict:
        """Generate medium questions - Relationships between movies and people"""
        question_types = [
            self._generate_actor_movie_question,
            self._generate_multi_genre_question,
            self._generate_director_year_question
        ]
        return random.choice(question_types)()

    def generate_hard_question(self) -> Dict:
        """Generate hard questions - Complex relationships and multiple facts"""
        question_types = [
            self._generate_actor_director_collaboration_question,
            self._generate_genre_year_combination_question,
            self._generate_multiple_movie_fact_question
        ]
        return random.choice(question_types)()

    def _generate_year_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        wrong_years = [int(movie['startYear'] + offset) for offset in [-2, -1, 1, 2]]
        wrong_years = random.sample(wrong_years, 3)
        
        return {
            'question': f"In which year was '{movie['primaryTitle']}' released?",
            'correct_answer': int(movie['startYear']),
            'options': sorted([int(movie['startYear'])] + wrong_years),
            'difficulty': 'easy'
        }

    def _generate_genre_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        all_genres = set(sum([m for m in self.movies_df['genres'].values], []))
        wrong_genres = random.sample([g for g in all_genres if g not in movie['genres']], 3)
        correct_genre = random.choice(movie['genres'])
        
        return {
            'question': f"Which of the following genres is associated with '{movie['primaryTitle']}'?",
            'correct_answer': correct_genre,
            'options': sorted([correct_genre] + wrong_genres),
            'difficulty': 'easy'
        }

    def _generate_director_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        director_id = movie['directors'][0]
        director_name = self.directors_df[self.directors_df['nconst'] == director_id]['primaryName'].iloc[0]
        wrong_directors = self.directors_df[self.directors_df['nconst'] != director_id]['primaryName'].sample(3).tolist()
        
        return {
            'question': f"Who directed '{movie['primaryTitle']}'?",
            'correct_answer': director_name,
            'options': sorted([director_name] + wrong_directors),
            'difficulty': 'easy'
        }

    def _generate_actor_movie_question(self) -> Dict:
        actor = self.actors_df.sample(1).iloc[0]
        actor_movies = [
            self.movies_df[self.movies_df['tconst'] == movie_id]['primaryTitle'].iloc[0]
            for movie_id in actor['knownForTitles']
            if not self.movies_df[self.movies_df['tconst'] == movie_id].empty
        ]
        if actor_movies:
            correct_movie = random.choice(actor_movies)
            wrong_movies = self.movies_df[~self.movies_df['primaryTitle'].isin(actor_movies)]['primaryTitle'].sample(3).tolist()
            
            return {
                'question': f"In which of these movies did {actor['primaryName']} appear?",
                'correct_answer': correct_movie,
                'options': sorted([correct_movie] + wrong_movies),
                'difficulty': 'medium'
            }
        return self.generate_medium_question()

    def _generate_multi_genre_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        genres = movie['genres']
        if len(genres) >= 2:
            genre_pairs = [genres[i:i+2] for i in range(len(genres)-1)]
            correct_pair = random.choice(genre_pairs)
            all_genres = set(sum([m for m in self.movies_df['genres'].values], []))
            wrong_pairs = []
            while len(wrong_pairs) < 3:
                wrong_pair = random.sample(list(all_genres), 2)
                if wrong_pair not in wrong_pairs and wrong_pair != correct_pair:
                    wrong_pairs.append(wrong_pair)
            
            return {
                'question': f"Which pair of genres describes '{movie['primaryTitle']}'?",
                'correct_answer': f"{correct_pair[0]} and {correct_pair[1]}",
                'options': sorted([f"{pair[0]} and {pair[1]}" for pair in [correct_pair] + wrong_pairs]),
                'difficulty': 'medium'
            }
        return self.generate_medium_question()

    def _generate_director_year_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        director_id = movie['directors'][0]
        director_name = self.directors_df[self.directors_df['nconst'] == director_id]['primaryName'].iloc[0]
        year = int(movie['startYear'])
        
        wrong_combinations = [
            f"{director_name}, {year + 2}",
            f"{self.directors_df.sample(1).iloc[0]['primaryName']}, {year}",
            f"{self.directors_df.sample(1).iloc[0]['primaryName']}, {year - 1}"
        ]
        
        return {
            'question': f"Which director-year combination is correct for '{movie['primaryTitle']}'?",
            'correct_answer': f"{director_name}, {year}",
            'options': sorted([f"{director_name}, {year}"] + wrong_combinations),
            'difficulty': 'medium'
        }

    def _generate_actor_director_collaboration_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        director_id = movie['directors'][0]
        director_name = self.directors_df[self.directors_df['nconst'] == director_id]['primaryName'].iloc[0]
        actor_id = random.choice(movie['actors'])
        actor_name = self.actors_df[self.actors_df['nconst'] == actor_id]['primaryName'].iloc[0]
        
        wrong_combinations = [
            f"{self.actors_df.sample(1).iloc[0]['primaryName']} with {director_name}",
            f"{actor_name} with {self.directors_df.sample(1).iloc[0]['primaryName']}",
            f"{self.actors_df.sample(1).iloc[0]['primaryName']} with {self.directors_df.sample(1).iloc[0]['primaryName']}"
        ]
        
        return {
            'question': f"Which actor-director collaboration occurred in '{movie['primaryTitle']}'?",
            'correct_answer': f"{actor_name} with {director_name}",
            'options': sorted([f"{actor_name} with {director_name}"] + wrong_combinations),
            'difficulty': 'hard'
        }

    def _generate_genre_year_combination_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        genre = random.choice(movie['genres'])
        year = int(movie['startYear'])
        
        wrong_combinations = [
            f"{genre}, {year + 2}",
            f"{random.choice(sum([m for m in self.movies_df['genres'].values], []))}, {year}",
            f"{random.choice(sum([m for m in self.movies_df['genres'].values], []))}, {year - 1}"
        ]
        
        return {
            'question': f"Which genre-year combination is correct for '{movie['primaryTitle']}'?",
            'correct_answer': f"{genre}, {year}",
            'options': sorted([f"{genre}, {year}"] + wrong_combinations),
            'difficulty': 'hard'
        }

    def _generate_multiple_movie_fact_question(self) -> Dict:
        movie = self.movies_df.sample(1).iloc[0]
        director_id = movie['directors'][0]
        director_name = self.directors_df[self.directors_df['nconst'] == director_id]['primaryName'].iloc[0]
        genre = random.choice(movie['genres'])
        year = int(movie['startYear'])
        
        correct_facts = f"{director_name} directed this {genre} film in {year}"
        wrong_facts = [
            f"{director_name} directed this {random.choice(sum([m for m in self.movies_df['genres'].values], []))} film in {year + 1}",
            f"{self.directors_df.sample(1).iloc[0]['primaryName']} directed this {genre} film in {year}",
            f"{self.directors_df.sample(1).iloc[0]['primaryName']} directed this {random.choice(sum([m for m in self.movies_df['genres'].values], []))} film in {year - 1}"
        ]
        
        return {
            'question': f"Which statement is correct about '{movie['primaryTitle']}'?",
            'correct_answer': correct_facts,
            'options': sorted([correct_facts] + wrong_facts),
            'difficulty': 'hard'
        }

# Example usage
def generate_quiz(num_questions: int = 9) -> List[Dict]:
    quiz = MovieQuizGenerator('./selected-dataset/movies.tsv', './selected-dataset/actors.tsv', './selected-dataset/directors.tsv')
    questions = []
    
    # Generate equal numbers of each difficulty
    for _ in range(num_questions // 3):
        questions.extend([
            quiz.generate_easy_question(),
            quiz.generate_medium_question(),
            quiz.generate_hard_question()
        ])
    
    random.shuffle(questions)
    return questions

if __name__ == "__main__":
    quiz_questions = generate_quiz()
    
    # Print the quiz
    for i, q in enumerate(quiz_questions, 1):
        print(f"\nQuestion {i} (Difficulty: {q['difficulty']}):")
        print(q['question'])
        print("\nOptions:")
        for j, option in enumerate(q['options'], 1):
            print(f"{j}. {option}")
        print(f"\nCorrect answer: {q['correct_answer']}")
        print("-" * 80)

