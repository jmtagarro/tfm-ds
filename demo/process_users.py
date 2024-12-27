import os
import json
import pandas as pd
from multiprocessing import Pool, cpu_count

# Paths
RATINGS_PATH = '../data/ml-20m-psm/ratings.csv'
USER_GROUPS_PATH = 'user_groups.json'

# User groups
user_groups = {"lessthan50": [], "from51to150": [], "from151to300": [], "morethan300": [], "all": []}

# Function to categorize users
def categorize_user(row):
    user_id, count = row
    if count < 50:
        return ("lessthan50", user_id)
    elif 50 <= count <= 150:
        return ("from51to150", user_id)
    elif 150 < count <= 300:
        return ("from151to300", user_id)
    elif count > 300:
        return ("morethan300", user_id)
    return None

# Process ratings to categorize users by number of ratings
def process_users():
    global user_groups

    print("Processing ratings to generate user groups...")
    ratings = pd.read_csv(RATINGS_PATH, sep=',', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'], skiprows=0)

    # Ensure all values are correctly parsed
    ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
    ratings = ratings.dropna(subset=['userId'])
    ratings['userId'] = ratings['userId'].astype(int)

    # Count ratings per user
    rating_counts = ratings['userId'].value_counts().reset_index()
    rating_counts.columns = ['userId', 'count']

    # Use multiprocessing to categorize users
    with Pool(cpu_count()) as pool:
        results = pool.map(categorize_user, rating_counts.itertuples(index=False, name=None))

    # Aggregate results into user groups
    for result in results:
        if result:
            group, user_id = result
            user_groups[group].append(user_id)
    
    user_groups['all'] = rating_counts['userId'].tolist()

    # Save user groups to file
    with open(USER_GROUPS_PATH, 'w') as f:
        json.dump(user_groups, f)

    print("User groups saved to file.")

if __name__ == '__main__':
    process_users()

