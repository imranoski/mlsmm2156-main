from admin_page import admin_page
from user_page import user_page
import streamlit as st

def main():
    st.title("Recommendation System")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", ["Admin Page", "User Page"])
    
    if page == "Admin Page":
        admin_page()
    else:
        user_page()

if __name__ == "__main__":
    main()