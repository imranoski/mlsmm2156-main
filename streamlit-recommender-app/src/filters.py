def filter_by_year(recommendations, start_year, end_year):
    """Filter recommendations by year range."""
    return [rec for rec in recommendations if start_year <= rec['year'] <= end_year]

def filter_by_tags(recommendations, tags):
    """Filter recommendations by tags."""
    if not tags:
        return recommendations
    return [rec for rec in recommendations if any(tag in rec['tags'] for tag in tags)]

def filter_recommendations(recommendations, start_year=None, end_year=None, tags=None):
    """Filter recommendations based on year range and tags."""
    filtered_recommendations = recommendations
    if start_year is not None and end_year is not None:
        filtered_recommendations = filter_by_year(filtered_recommendations, start_year, end_year)
    if tags is not None:
        filtered_recommendations = filter_by_tags(filtered_recommendations, tags)
    return filtered_recommendations