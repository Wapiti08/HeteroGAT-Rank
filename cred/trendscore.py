'''
 # @ Create Time: 2025-07-01 12:49:28
 # @ Modified time: 2025-07-01 12:49:31
 # @ Description: calculate trend score based on semantic similarity match

score = feature_weight × similarity × trend_count

 '''

import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_trend_score(top_features_with_score, trend_dict, verbose=True):
    ''' compute semantic trend score for top features

    args:
        top_features_with_scores (list of tuples): [(feature, score),...]
        trend_dict (dict): category -> {trend_key: count}
        verbose(bool): whether to print each category score.
    
    returns:
        total_score (float), category_scores (dict)

    '''
    features = [f for f, _ in top_features_with_score]
    scores = [s for _, s in top_features_with_score]

    # check whether feature scores are the same
    if all(np.isclose(s, scores[0]) for s in scores):
        weights = [5.5] * len(scores)
    else:
        weights = list(range(10,0,-1))
    
    feature_embeds = model.encode(features, convert_to_tensor=True)

    category_scores = {}

    for category, trend_items in trend_dict.items():
        score = 0.0
        trend_keys = list(trend_items.index)
        trend_counts = trend_items.values
        
        trend_embeds = model.encode(trend_keys, convert_to_tensor=True)
        sim_matrix = util.cos_sim(feature_embeds, trend_embeds).cpu().numpy()

        for i, feature in enumerate(features):
            for j, trend_key in enumerate(trend_keys):
                similarity = sim_matrix[i][j]
                weighted_score = weights[i] * similarity * trend_counts[j]
                score += weighted_score

        category_scores[category] = score

        if verbose:
            print(f"[{category}] trend score: {score:.2f}")

    total_score = sum(category_scores.values())
    if verbose:
        print(f"\n🔹 Total trend score: {total_score:.2f}")
    return total_score, category_scores


def compute_desc_cluster_score(top_features_with_score, clustered_df, verbose=True, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)

    features = [f for f, _ in top_features_with_score]
    scores = [s for _, s in top_features_with_score]

    # unify the ranking scores
    if all(np.isclose(s, scores[0]) for s in scores):
        weights = [5.5] * len(scores)
    else:
        weights = list(range(10, 10 - len(features), -1))

    # get the desc in clusters
    cluster_desc_map = (
        clustered_df
        .groupby('desc_cluster')['desc']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    # get the size of cluster
    cluster_size_map = clustered_df['desc_cluster'].value_counts().to_dict()

    cluster_ids = list(cluster_desc_map.keys())
    trend_keys = [cluster_desc_map[cl] for cl in cluster_ids]
    trend_embeds = model.encode(trend_keys, convert_to_tensor=True)
    
    feature_embeds = model.encode(features, convert_to_tensor=True)

    sim_matrix = util.cos_sim(feature_embeds, trend_embeds).cpu().numpy()

    # weighted scoring
    cluster_scores = defaultdict(float)
    for i, feature in enumerate(features):
        for j, cluster_id in enumerate(cluster_ids):
            similarity = sim_matrix[i][j]
            weighted_score = weights[i] * similarity * cluster_size_map[cluster_id]
            cluster_scores[cluster_id] += weighted_score

    total_score = sum(cluster_scores.values())

    if verbose:
        for cid, score in cluster_scores.items():
            print(f"[Cluster {cid}] score: {score:.2f} (desc: \"{cluster_desc_map[cid]}\")")
        print(f"\n🔹 Total trend score: {total_score:.2f}")

    return total_score, dict(cluster_scores)

if __name__ == "__main__":

    top_features_with_scores = [
    ("Readline/readline-i.ri", 1.0000), ("libxml/xmlstring.h", 1.0000),
    ("ClassMethods/commands-i.ri", 1.0000), ("bundler/plugin", 1.0000),
    ("Color/set_color-i.ri", 1.0000), ("HiddenCommand/cdesc-HiddenCommand.ri", 1.0000),
    ("Thor/Base", 1.0000), ("templates/newgem", 1.0000),
    ("Actions/inject_into_class-i.ri", 1.0000), ("source/git", 1.0000)
    ]

    trend_dict = {
    "location": {
        "entrypoint/download": 12026,
        "setup": 5462,
        "socket or other communication protocols modules": 1674
    },
    "function": {
        "install powerful malware": 12026,
        "spyware and information stealing": 5425,
        "build communication tunnel with C2": 1674
        }
    }

    compute_trend_score(top_features_with_scores, trend_dict)

