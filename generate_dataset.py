from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import re

from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


def random_index_excluding_token(string: str, token: str = "<N>") -> int | None:
    """
    Returns a random index in the given string that does not point to a given `token.

    Parameters:
    - string (str): The input string containing `token`s.
    - token (str, optional): The token. Default: "<N>".

    Returns:
    - int: A random valid index not pointing to the `token`, or None if no valid index exists.
    """
    token_length = len(token)
    n = len(string)

    # Identify invalid index ranges (occupied by "<N>")
    invalid_ranges = []
    start = 0

    while start < n:
        # Find occurrences of "<N>"
        start = string.find(token, start)
        if start == -1:
            break
        # Mark the indices occupied by "<N>"
        invalid_ranges.append(range(start, start + token_length))
        start += token_length

    # Create a set of all valid indices
    valid_indices = set(range(1, n))
    for invalid_range in invalid_ranges:
        valid_indices -= set(invalid_range)

    # Convert to a sorted list of valid indices
    valid_indices = sorted(valid_indices)

    # If no valid indices exist, return None
    if not valid_indices:
        return None

    # Return a random choice from valid indices
    return random.choice(valid_indices)


def process_text(
    text: str,
    digit_token: str = "<N>"
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Returns the count of all processing strings in a text. This function:
    - Preprocesses the text so that it only contains hangul characters and whitespace.
    - Replaces all numbers with a `digit_token`.
    - Traverses each word in the text and:
      - Randomly splits and counts words longer than one character into two.
      - Counts adjacent words.

    Parameters:
    - text (str): The input text to process.
    - digit_token (str, optional): The token to replace all numbers with. Default: "<N>"

    Returns:
    - tuple[dict[str, int], dict[str, int]]: two dicts with (string: str, count: int) key, value pairs that count the
      number of strings that contain 1) the same word and 2) different words.
    """
    # Remove all characters enclosed in parentheses or brackets
    text = re.sub(r"\(.+?\)", "", text)
    # Remove all characters that are not hangul, digits, whitespace, or newlines
    text = re.sub(r"[^\u3131-\uD79D\d\s\n]", "", text)
    # Replace newlines with whitespace
    text = re.sub(r"\n", " ", text)
    # Remove multiple consecutive whitespaces
    text = re.sub(r"\s{2,}", " ", text)
    # Replace all numbers with `<N>` token
    text = re.sub(r"\d+", digit_token, text)

    # Words split into two
    local_same = []
    # Adjacent words
    local_different = []

    # Iterate over each word in the text
    words = text.split()
    for i in range(len(words)):
        length = len(words[i])
        # If the current word is longer than one character
        if length > 1:
            # Randomly split the word into two
            rand_index = random_index_excluding_token(words[i])
            split_str = f"{words[i][:rand_index]} {words[i][rand_index:]}"
            local_same.append(split_str)
        # If the current word is not the last word
        if i < len(words) - 1:
            local_different.append(f"{words[i]} {words[i + 1]}")

    local_same_counts = Counter(local_same)
    local_different_counts = Counter(local_different)
    return local_same_counts, local_different_counts


def main(
    dataset_path: str = "wikimedia/wikipedia",
    dataset_name: str = "20231101.ko"
):
    ds = load_dataset(dataset_path, dataset_name)

    same = defaultdict(int)  # Includes strings until they occur 100 times
    same_passed = dict()  # Includes strings that occur at least 100 times
    different = defaultdict(int)
    different_passed = dict()

    # Define number of processes
    num_processes = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []

        # Process text in each entry in the dataset
        iterator = tqdm(
            enumerate(ds["train"]),
            total=len(ds["train"]),
            desc="Loading data"
        )
        for i, entry in iterator:
            futures.append(executor.submit(process_text, entry["text"]))

        # Count the total occurrences of strings across the whole dataset
        iterator = tqdm(
            as_completed(futures),
            total=len(ds["train"]),
            desc="Processing data"
        )
        for future in iterator:
            local_same_counts, local_different_counts = future.result()

            # Count the `same word` strings
            for string, count in local_same_counts.items():

                # Try counting this string
                try:
                    same_passed[string] += count

                # KeyError is raised when the string has not yet occurred at least 100 times
                except KeyError:
                    same[string] += count

                    # If the string has occurred at least 100 times
                    if same[string] > 100:
                        same_passed[string] = same[string] - 100

            # Count the `different word` strings
            for string, count in local_different_counts.items():
                try:
                    different_passed[string] += count
                except KeyError:
                    different[string] += count
                    if different[string] > 100:
                        different_passed[string] = different[string] - 100

    # Prepare data for JSON output
    n_same = len(same_passed)
    n_diff = len(different_passed)
    n = n_same + n_diff

    # Initialize an empty dataframe
    df = pd.DataFrame({
        "text": pd.Series([""] * n),
        "label": pd.Series(np.zeros(n), dtype=int),
        "count": pd.Series(np.zeros(n), dtype=int),
    })

    # Add the `same word` strings to the dataframe
    iterator = tqdm(
        enumerate(same_passed.items()),
        total=n_same,
        desc="Processing same words"
    )
    for i, (string, count) in iterator:
        df.iloc[i] = [string, 0, count]

    # Add the `different word` strings to the dataframe
    iterator = tqdm(
        enumerate(different_passed.items()),
        total=n_diff,
        desc="Processing different words"
    )
    for i, (string, count) in iterator:
        df.iloc[i + n_same] = [string, 1, count]

    df.to_csv("training_data.csv", index=False)


if __name__ == "__main__":
    main()
