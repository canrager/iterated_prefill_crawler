import os
import json
from typing import Dict, Any
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from core.project_config import RESULT_DIR

def create_oval_mask(width: int, height: int) -> np.ndarray:
    """
    Create an oval mask for the wordcloud.
    
    Args:
        width: Width of the mask
        height: Height of the mask
        
    Returns:
        A 2D numpy array where 0 represents the mask area (where words will be placed) and 255 represents the background
    """
    mask = np.ones((height, width), dtype=np.uint8) * 255  # Start with white background
    center_x = width // 2
    center_y = height // 2
    a = width // 2  # semi-major axis
    b = height // 2  # semi-minor axis
    
    for y in range(height):
        for x in range(width):
            if ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1:
                mask[y, x] = 0  # Black for the oval area where words will be placed
    return mask

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    Color function that uses the normalized score to determine the color.
    Returns RGB color as a tuple of integers.
    """
    # Get the normalized score from the word_scores dictionary
    normalized_score = kwargs.get('word_scores', {}).get(word, 0)
    # Use the colormap to get the color based on the normalized score
    color = kwargs.get('colormap')(normalized_score)
    # Convert to RGB integers (0-255)
    return tuple(int(x * 255 *0.9) for x in color[:3])

def generate_wordcloud_from_ranking(
    run_title: str,
    result_dir: str = RESULT_DIR,
    method: str = "elo",
    width: int = 1600,
    height: int = 1000,
    background_color: str = "rgba(255, 255, 255, 0)",
    colormap: str = "winter",
    min_font_size: int = 10,
    max_font_size: int = 120,
    relative_scaling: float = 0.5,
) -> None:
    """
    Generate a word cloud from ranking results where word sizes and colors are proportional to their scores.
    
    Args:
        run_title: Title of the run to generate word cloud for
        result_dir: Directory containing the ranking results
        method: Ranking method to use (default: "elo")
        width: Width of the word cloud image
        height: Height of the word cloud image
        background_color: Background color of the word cloud (default: transparent)
        colormap: Matplotlib colormap to use for words
        min_font_size: Minimum font size for words
        max_font_size: Maximum font size for words
        relative_scaling: How much the size of words should be based on frequency (0-1)
    """
    # Try to load ranking results
    ranking_file = os.path.join(result_dir, f"topics_clustered_ranked_{run_title}.json")
    if not os.path.exists(ranking_file):
        print(f"No ranking file found at {ranking_file}")
        return
        
    with open(ranking_file, 'r') as f:
        clusters = json.load(f)
        
    # Create word frequency dictionary where frequency is the score
    word_scores = {}
    for cluster_str, cluster_data in clusters.items():
        word = cluster_str.strip()
        score = cluster_data["ranking"][method]["rank_score"]
        word_scores[word] = score
        
    # Normalize scores to be between 0 and 1
    if word_scores:
        min_score = min(word_scores.values())
        max_score = max(word_scores.values())
        score_range = max_score - min_score
        if score_range > 0:
            for word in word_scores:
                normalized_score = (word_scores[word] - min_score) / score_range
                word_scores[word] = normalized_score
    
    # Create oval mask
    mask = create_oval_mask(width, height)
    
    # Create colormap function
    cmap = plt.get_cmap(colormap)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        relative_scaling=relative_scaling,
        normalize_plurals=False,
        mask=mask,
        mode="RGBA",
        color_func=lambda word, font_size, position, orientation, random_state=None, **kwargs: color_func(
            word, font_size, position, orientation, random_state, 
            word_scores=word_scores, colormap=cmap
        ),
    ).generate_from_frequencies(word_scores)
    
    # Create and save the plot
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save the word cloud
    output_file = os.path.join(result_dir, f"wordcloud_{method}_{run_title}.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Word cloud saved to {output_file}") 