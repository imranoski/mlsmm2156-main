import streamlit as st
from src.admin_page import admin_page
from src.user_page import user_page

def main():
    st.title("Recommendation System")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Admin", "User"])

    if page == "Admin":
        admin_page()
    elif page == "User":
        user_page()

if __name__ == "__main__":
    main()