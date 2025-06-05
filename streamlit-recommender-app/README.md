# Streamlit Recommender System

This project is a Streamlit application for a recommendation system that allows users to obtain personalized recommendations based on their preferences. The application consists of two main pages: an admin page for modifying model parameters and a user page for inputting user IDs to receive recommendations.

## Project Structure

```
streamlit-recommender-app
├── src
│   ├── admin_page.py       # Streamlit code for the admin page
│   ├── user_page.py        # Streamlit code for the user page
│   ├── recommender.py       # Recommendation logic
│   ├── filters.py           # Functions for filtering recommendations
│   └── __init__.py         # Marks the directory as a Python package
├── data                     # Directory for datasets
│   └── (place your datasets here)
├── requirements.txt         # Dependencies for the application
├── README.md                # Documentation for the project
└── streamlit_app.py         # Entry point for the Streamlit application
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd streamlit-recommender-app
   ```

2. **Install dependencies**:
   Make sure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Place your datasets**:
   Add your datasets to the `data` directory.

4. **Run the application**:
   Start the Streamlit application by running:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

- **Admin Page**: Modify model parameters and settings for the recommendation system.
- **User Page**: Input your user ID to obtain a list of recommendations. You can filter the recommendations by year range and tags.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.