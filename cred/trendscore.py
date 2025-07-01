'''
 # @ Create Time: 2025-07-01 12:49:28
 # @ Modified time: 2025-07-01 12:49:31
 # @ Description: calculate trend score based on semantic similarity match

score = feature_weight × similarity × trend_count

 '''

import numpy as np
from sentence_transformers import SentenceTransformer, util

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