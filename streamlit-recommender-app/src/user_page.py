from streamlit import st
import pandas as pd
from recommender import get_recommendations  # Assuming this function exists in recommender.py
from filters import filter_recommendations  # Assuming this function exists in filters.py

def user_page():
    st.title("User Recommendation Page")

    user_id = st.text_input("Enter your User ID:")
    
    if st.button("Get Recommendations"):
        if user_id:
            recommendations = get_recommendations(user_id)
            if recommendations:
                st.write("Recommendations:")
                
                # Filters
                year_range = st.slider("Select Year Range", 1900, 2023, (2000, 2023))
                tags = st.multiselect("Select Tags", options=["Action", "Comedy", "Drama", "Horror", "Romance"])

                # Filter recommendations based on user input
                filtered_recommendations = filter_recommendations(recommendations, year_range, tags)

                # Display filtered recommendations
                for rec in filtered_recommendations:
                    st.write(f"Title: {rec['title']}, Year: {rec['year']}, Tags: {', '.join(rec['tags'])}")
            else:
                st.write("No recommendations found for this User ID.")
        else:
            st.write("Please enter a valid User ID.")

if __name__ == "__main__":
    user_page()