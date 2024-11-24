from dataclasses import dataclass
from typing import Set, List, Dict, Iterator
import pandas as pd
import os
from abc import ABC, abstractmethod

@dataclass
class IMDBConfig:
    """Configuration class for IMDB data processing.
    
    Attributes:
        base_path (str): Root directory containing IMDB dataset files
        output_path (str): Directory where processed files will be saved
        vote_threshold (int): Minimum number of votes required for a movie to be included
        chunk_size (int): Number of rows to process at once for memory efficiency
    """
    base_path: str
    output_path: str
    vote_threshold: int
    chunk_size: int = 100000

class FileHandler:
    """Handles file operations for reading and writing data.
    
    This class provides utility methods for file operations, including
    directory creation and chunked reading of TSV files.
    """
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Creates a directory if it doesn't exist.
        
        Args:
            path (str): Path to the directory to be created
        """
        if not os.path.exists(path):
            os.makedirs(path)
    
    @staticmethod
    def read_tsv_chunks(filepath: str, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Reads a TSV file in chunks to manage memory usage.
        
        Args:
            filepath (str): Path to the TSV file
            chunk_size (int): Number of rows to read in each chunk
        
        Returns:
            Iterator[pd.DataFrame]: Iterator yielding DataFrame chunks
        
        Note:
            Uses '\\N' as NA value which is common in IMDB dataset
        """
        return pd.read_csv(
            filepath, 
            sep='\t', 
            low_memory=False, 
            na_values='\\N', 
            chunksize=chunk_size
        )

class DataProcessor(ABC):
    """Abstract base class for data processors.
    
    Provides a common interface for all data processors and handles
    configuration and file handling initialization.
    
    Attributes:
        config (IMDBConfig): Configuration object
        file_handler (FileHandler): Utility for file operations
    """
    def __init__(self, config: IMDBConfig):
        """
        Args:
            config (IMDBConfig): Configuration object with processing parameters
        """
        self.config = config
        self.file_handler = FileHandler()

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Abstract method to process data.
        
        Returns:
            pd.DataFrame: Processed data
            
        Note:
            Must be implemented by concrete classes
        """
        pass

class MovieQualifier(DataProcessor):
    """Processes movie qualifications based on vote threshold.
    
    Filters movies based on the minimum vote threshold specified in the config.
    """
    def process(self) -> Set[str]:
        """Process movie ratings to find qualifying movies.
        
        Returns:
            Set[str]: Set of movie IDs (tconst) that meet the vote threshold
            
        Note:
            Uses chunked processing to handle large datasets efficiently
        """
        qualifying_movies = set()
        ratings_file = f"{self.config.base_path}/title.ratings.tsv"
        
        for chunk in self.file_handler.read_tsv_chunks(ratings_file, self.config.chunk_size):
            qualifying = chunk[chunk['numVotes'] >= self.config.vote_threshold]['tconst']
            qualifying_movies.update(qualifying.tolist())
        
        return qualifying_movies

class MovieProcessor(DataProcessor):
    """Processes basic movie data from the IMDB dataset.
    
    Extracts and processes movie information including titles, years, and genres.
    """
    def process(self, qualifying_movies: Set[str]) -> pd.DataFrame:
        """Process basic movie data for qualifying movies.
        
        Args:
            qualifying_movies (Set[str]): Set of movie IDs that meet the vote threshold
            
        Returns:
            pd.DataFrame: Processed movie data with columns:
                - tconst: Movie ID
                - primaryTitle: Movie title
                - startYear: Release year
                - genres: List of genres
        """
        movies = []
        basics_file = f"{self.config.base_path}/title.basics.tsv"

        for chunk in self.file_handler.read_tsv_chunks(basics_file, self.config.chunk_size):
            movie_chunk = chunk[
                (chunk['titleType'] == 'movie') & 
                (chunk['tconst'].isin(qualifying_movies))
            ][['tconst', 'primaryTitle', 'startYear', 'genres']].copy()
            
            movie_chunk.loc[:, 'genres'] = movie_chunk['genres'].str.split(',')
            movies.append(movie_chunk)

        return pd.concat(movies, ignore_index=True)

class CreditProcessor(DataProcessor):
    """Processes movie credits including actors and directors.
    
    Handles the extraction and processing of cast and crew information.
    
    Attributes:
        actors (List[pd.DataFrame]): List of DataFrames containing actor information
        directors (List[pd.DataFrame]): List of DataFrames containing director information
    """
    def __init__(self, config: IMDBConfig):
        """
        Args:
            config (IMDBConfig): Configuration object with processing parameters
        """
        super().__init__(config)
        self.actors: List[pd.DataFrame] = []
        self.directors: List[pd.DataFrame] = []

    def process(self, movie_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process credit information for qualifying movies.
        
        Args:
            movie_df (pd.DataFrame): DataFrame containing movie information
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed actor and director DataFrames
        """
        self._initialize_credit_lists(movie_df)
        self._process_credits(movie_df)
        return self._finalize_credits()

    def _initialize_credit_lists(self, movie_df: pd.DataFrame) -> None:
        """Initialize empty credit lists in the movie DataFrame.
        
        Args:
            movie_df (pd.DataFrame): DataFrame to initialize with credit lists
        """
        movie_df['actors'] = [[] for _ in range(len(movie_df))]
        movie_df['directors'] = [[] for _ in range(len(movie_df))]

    def _process_credits(self, movie_df: pd.DataFrame) -> None:
        """Process credits from principals file.
        
        Args:
            movie_df (pd.DataFrame): DataFrame containing movie information
            
        Note:
            Updates self.actors and self.directors lists with processed data
        """
        principals_file = f"{self.config.base_path}/title.principals.tsv"
        
        for chunk in self.file_handler.read_tsv_chunks(principals_file, self.config.chunk_size):
            relevant_chunk = chunk[chunk['tconst'].isin(movie_df['tconst'])]
            if len(relevant_chunk) == 0:
                continue

            self._process_actors(relevant_chunk, movie_df)
            self._process_directors(relevant_chunk, movie_df)

    def _process_actors(self, chunk: pd.DataFrame, movie_df: pd.DataFrame) -> None:
        """Process actor credits from a chunk of data.
        
        Args:
            chunk (pd.DataFrame): Chunk of principals data
            movie_df (pd.DataFrame): DataFrame containing movie information
        """
        actors_chunk = chunk[chunk['category'] == 'actor'][['tconst', 'nconst']]
        if not actors_chunk.empty:
            self.actors.append(actors_chunk)
            self._update_credits(movie_df, actors_chunk, 'actors')

    def _process_directors(self, chunk: pd.DataFrame, movie_df: pd.DataFrame) -> None:
        """Process director credits from a chunk of data.
        
        Args:
            chunk (pd.DataFrame): Chunk of principals data
            movie_df (pd.DataFrame): DataFrame containing movie information
        """
        directors_chunk = chunk[chunk['category'] == 'director'][['tconst', 'nconst']]
        if not directors_chunk.empty:
            self.directors.append(directors_chunk)
            self._update_credits(movie_df, directors_chunk, 'directors')

    def _update_credits(self, movie_df: pd.DataFrame, credit_chunk: pd.DataFrame, credit_type: str) -> None:
        """Update credit lists in the movie DataFrame.
        
        Args:
            movie_df (pd.DataFrame): DataFrame to update
            credit_chunk (pd.DataFrame): Chunk of credit data to process
            credit_type (str): Type of credit ('actors' or 'directors')
        """
        credits_grouped = credit_chunk.groupby('tconst')['nconst'].apply(list).reset_index(name='new_credits')
        movie_df_merged = pd.merge(movie_df, credits_grouped, how='left', on="tconst")
        movie_df[credit_type] = movie_df_merged.apply(
            lambda row: row[credit_type] + (row['new_credits'] if isinstance(row['new_credits'], list) else []),
            axis=1
        )

    def _finalize_credits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Finalize credit processing and return results.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Final actor and director DataFrames
        """
        return (
            pd.concat(self.actors, ignore_index=True),
            pd.concat(self.directors, ignore_index=True)
        )

class NameProcessor(DataProcessor):
    """Processes names data for actors and directors.
    
    Handles the extraction and processing of personal information for cast and crew.
    """
    def process(self, actors_df: pd.DataFrame, directors_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process name information for actors and directors.
        
        Args:
            actors_df (pd.DataFrame): DataFrame containing actor IDs
            directors_df (pd.DataFrame): DataFrame containing director IDs
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Processed actor and director data with names
        """
        names_file = f"{self.config.base_path}/name.basics.tsv"
        
        actors_df = self._initialize_name_columns(actors_df)
        directors_df = self._initialize_name_columns(directors_df)

        for chunk in self.file_handler.read_tsv_chunks(names_file, self.config.chunk_size):
            actors_df = self._process_names_chunk(chunk, actors_df)
            directors_df = self._process_names_chunk(chunk, directors_df)

        return self._finalize_names(actors_df, directors_df)

    def _initialize_name_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialize name-related columns in DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to initialize
            
        Returns:
            pd.DataFrame: DataFrame with initialized name columns
        """
        df['primaryName'] = None
        df['knownForTitles'] = None
        return df

    def _process_names_chunk(self, chunk: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of names data.
        
        Args:
            chunk (pd.DataFrame): Chunk of names data
            df (pd.DataFrame): DataFrame to update with names
            
        Returns:
            pd.DataFrame: Updated DataFrame with processed names
        """
        relevant_chunk = chunk[chunk['nconst'].isin(df['nconst'])]
        if relevant_chunk.empty:
            return df

        return self._merge_names_data(df, relevant_chunk[['nconst', 'primaryName', 'knownForTitles']])

    def _merge_names_data(self, df: pd.DataFrame, names_chunk: pd.DataFrame) -> pd.DataFrame:
        """Merge names data into existing DataFrame.
        
        Args:
            df (pd.DataFrame): Existing DataFrame
            names_chunk (pd.DataFrame): Chunk of names data to merge
            
        Returns:
            pd.DataFrame: Merged DataFrame with updated name information
        """
        df = df.merge(
            names_chunk.rename(columns={
                'primaryName': 'new_primaryName',
                'knownForTitles': 'new_knownForTitles'
            }),
            on='nconst',
            how='left'
        )
        
        df['primaryName'] = df.apply(
            lambda row: row['new_primaryName'] if pd.isna(row['primaryName']) else row['primaryName'],
            axis=1
        )
        df['knownForTitles'] = df.apply(
            lambda row: row['new_knownForTitles'] if pd.isna(row['knownForTitles']) else row['knownForTitles'],
            axis=1
        )
        
        return df.drop(['new_primaryName', 'new_knownForTitles'], axis=1)

    def _finalize_names(self, actors_df: pd.DataFrame, directors_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Finalize name processing and prepare final datasets.
        
        Args:
            actors_df (pd.DataFrame): Processed actor DataFrame
            directors_df (pd.DataFrame): Processed director DataFrame
            
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Final actor and director DataFrames
        """
        actors_data = self._prepare_final_data(actors_df)
        directors_data = self._prepare_final_data(directors_df)
        return actors_data, directors_data

    def _prepare_final_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final dataset by cleaning and formatting data.
        
        Args:
            df (pd.DataFrame): DataFrame to prepare
            
        Returns:
            pd.DataFrame: Prepared DataFrame
        """
        df = df.drop('tconst', axis=1)
        df['knownForTitles'] = df['knownForTitles'].apply(
            lambda x: x.split(',') if isinstance(x, str) else []
        )
        return df

class IMDBDataProcessor:
    """Main class orchestrating the IMDB data processing pipeline.
    
    Coordinates the entire data processing workflow including:
    - Movie qualification
    - Basic movie data processing
    - Credit processing
    - Name processing
    - Result saving
    
    Attributes:
        config (IMDBConfig): Configuration for the processing pipeline
        file_handler (FileHandler): Utility for file operations
    """
    def __init__(self, config: IMDBConfig):
        """
        Args:
            config (IMDBConfig): Configuration object with processing parameters
        """
        self.config = config
        self.file_handler = FileHandler()
        self.file_handler.ensure_directory(config.output_path)

    def process(self) -> None:
        """Execute the complete data processing pipeline.
        
        Coordinates all processing steps and saves results to specified output directory.
        """
        # Step 1: Get qualifying movies
        print(f"Finding movies with at least {self.config.vote_threshold} votes...")
        movie_qualifier = MovieQualifier(self.config)
        qualifying_movies = movie_qualifier.process()
        print(f"Found {len(qualifying_movies)} movies meeting vote threshold")

        # Step 2: Process basic movie data
        print("\nExtracting movie data...")
        movie_processor = MovieProcessor(self.config)
        movie_df = movie_processor.process(qualifying_movies)

        # Step 3: Process credits
        print("\nExtracting cast and directors...")
        credit_processor = CreditProcessor(self.config)
        actors_df, directors_df = credit_processor.process(movie_df)

        # Step 4: Process names
        print("\nExtracting name data...")
        name_processor = NameProcessor(self.config)
        actors_data, directors_data = name_processor.process(actors_df, directors_df)

        # Step 5: Save results
        self._save_results(movie_df, actors_data, directors_data)
        self._print_statistics(movie_df, actors_data, directors_data)

    def _save_results(self, movie_df: pd.DataFrame, actors_data: pd.DataFrame, directors_data: pd.DataFrame) -> None:
        print("\nSaving datasets...")
        movie_df.to_csv(f"{self.config.output_path}/movies.tsv", sep='\t', index=False)
        actors_data.to_csv(f"{self.config.output_path}/actors.tsv", sep='\t', index=False)
        directors_data.to_csv(f"{self.config.output_path}/directors.tsv", sep='\t', index=False)

    def _print_statistics(self, movie_df: pd.DataFrame, actors_data: pd.DataFrame, directors_data: pd.DataFrame) -> None:
        print("\nDataset Statistics:")
        print(f"Total movies: {len(movie_df)}")
        print(f"Total actors: {len(actors_data)}")
        print(f"Total directors: {len(directors_data)}")
        print("\nProcessing complete! Datasets have been saved to the output directory.")

def main():
    config = IMDBConfig(
        base_path="./imdb-dataset",
        output_path="./selected-dataset",
        vote_threshold=50000
    )
    processor = IMDBDataProcessor(config)
    processor.process()

if __name__ == "__main__":
    main()

