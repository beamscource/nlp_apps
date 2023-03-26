def visualize_topic_model(topic_model, run_dir, tag):
    try:
        similarity_matrix = topic_model.visualize_heatmap()
        similarity_matrix.write_html(f"./{run_dir}/_{tag}_similarity_matrix.html")
    except:
        print('Cannot plot the similarity matrix.')
    try:
        topic_clusters = topic_model.visualize_topics()
        topic_clusters.write_html(f"./{run_dir}/_{tag}_topic_clusters.html")
    except:
        print('Cannot plot topic clusters.')

    try:
        top_keywords_per_topic = topic_model.visualize_barchart(top_n_topics=30,\
                                                                n_words=10)
        top_keywords_per_topic.write_html(f"./{run_dir}/_{tag}_top_keywords_per_topic.html")
    except:
        print('Cannot plot top keywords.')