import json
import pandas as pd
from collections import defaultdict
from sklearn.metrics import DistanceMetric

# --------------------------------------------------
# 1. File path
# --------------------------------------------------
file_path = "/Users/martinsokorie/Desktop/INST414 EX 5/imdb_movies_2000to2022.prolific.json"

# --------------------------------------------------
# 2. Load JSON Lines file properly
# --------------------------------------------------
data = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print("Loaded movies:", len(data))

# --------------------------------------------------
# 3. Build Actor × Genre count matrix
# --------------------------------------------------
actor_genre_counts = defaultdict(lambda: defaultdict(int))
actor_names = {}

for movie in data:
    genres = movie.get("genres", [])
    actors = movie.get("actors", [])

    for actor in actors:
        # IMPORTANT FIX: actor is a list [id, name]
        actor_id = actor[0]
        actor_name = actor[1]

        actor_names[actor_id] = actor_name

        for genre in genres:
            actor_genre_counts[actor_id][genre] += 1

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(actor_genre_counts, orient="index").fillna(0)
df = df.astype(int)

print("Feature matrix shape:", df.shape)

# --------------------------------------------------
# 4. Select Query Actor (Chris Hemsworth)
# --------------------------------------------------
query_actor_id = "nm1165110"

if query_actor_id not in df.index:
    raise ValueError("Chris Hemsworth (nm1165110) not found in dataset.")

query_vector = df.loc[[query_actor_id]].values

# --------------------------------------------------
# 5. Compute Euclidean Distances
# --------------------------------------------------
dist = DistanceMetric.get_metric("euclidean")

all_vectors = df.values
distances = dist.pairwise(query_vector, all_vectors)[0]

distance_df = pd.DataFrame({
    "actor_id": df.index,
    "distance": distances
})

# Remove query actor from results
distance_df = distance_df[distance_df["actor_id"] != query_actor_id]

# Sort by smallest distance
top_10 = distance_df.sort_values("distance").head(10)

# Add actor names
top_10["actor_name"] = top_10["actor_id"].map(actor_names)

# --------------------------------------------------
# 6. Print Top 10 Most Similar Actors
# --------------------------------------------------
print("\nTop 10 actors most similar to Chris Hemsworth:\n")

for _, row in top_10.iterrows():
    print(f"{row['actor_name']} ({row['actor_id']}) — Distance: {row['distance']:.4f}")