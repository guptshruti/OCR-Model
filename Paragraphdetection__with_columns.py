def group_words(prediction_groups, hor_threshold, ver_threshold):
    words_data = [(word, box) for word, box in prediction_groups[0]]
    centers = np.array([np.mean(box, axis=0) for _, box in words_data])

    # Adjust DBSCAN parameters as needed
    clustering = DBSCAN(eps=min(hor_threshold, ver_threshold), min_samples=1).fit(centers)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(words_data[idx])

    sorted_clusters = [sorted(cluster, key=lambda word: np.mean(word[1], axis=0)[1]) for cluster in clusters.values()]
    sorted_clusters.sort(key=lambda cluster: np.mean([np.mean(word[1], axis=0)[1] for word in cluster]))

    refined_clusters = []
    for cluster in sorted_clusters:
        if len(cluster) > 1:
            # Enhanced gap detection with context
            paragraph_groups = []
            current_paragraph = [cluster[0]]
            for i in range(1, len(cluster)):
                prev_y = np.mean(cluster[i - 1][1], axis=0)[1]
                curr_y = np.mean(cluster[i][1], axis=0)[1]
                vertical_gap = abs(curr_y - prev_y)

                # Use a relative gap threshold
                if vertical_gap > ver_threshold * 1.5:  # Test with different multipliers
                    paragraph_groups.append(current_paragraph)
                    current_paragraph = [cluster[i]]
                else:
                    current_paragraph.append(cluster[i])

            paragraph_groups.append(current_paragraph)
            refined_clusters.extend(paragraph_groups)
        else:
            refined_clusters.append(cluster)

    return refine_column_clusters(refined_clusters, hor_threshold)
