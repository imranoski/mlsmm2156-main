import streamlit as st

def admin_page():
    st.title("Admin Page")

    st.header("Modify Model Parameters")

    # Example of model parameters
    n_factors = st.number_input("Number of Factors", min_value=1, value=100)
    k_neighbors = st.number_input("Number of Neighbors", min_value=1, value=3)
    min_k = st.number_input("Minimum K", min_value=1, value=1)

    # Save button
    if st.button("Save Parameters"):
        # Here you would typically save the parameters to a config file or database
        st.success("Parameters saved successfully!")

    st.header("Model Settings")

    # Example of model selection
    model_options = ["ModelBaseline2", "ModelBaseline3", "ModelBaseline4", "ModelBaseline5"]
    selected_model = st.selectbox("Select Model", model_options)

    # Example of additional settings
    enable_logging = st.checkbox("Enable Logging", value=True)

    if st.button("Update Model Settings"):
        # Here you would typically update the model settings
        st.success("Model settings updated successfully!")

if __name__ == "__main__":
    admin_page()