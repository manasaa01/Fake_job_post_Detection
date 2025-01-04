import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from numpy import sqrt, argmax

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Page configurations
st.set_page_config(page_title="Job Verification", layout="wide")

# Create a sidebar for navigation
def create_navbar():
    st.sidebar.title("Navigation")
    pages = ["Home", "Predict", "Compare", "Understanding", "Contact"]
    return st.sidebar.radio("Go to", pages)

# Function for home page
def home_page():
    st.title("Job Verify")
    
    # Create navbar
    st.markdown("""
    <style>
    .nav-button {
        background-color: #f2f2f2;
        border-radius: 5px;
        margin: 5px;
        padding: 10px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .nav-button:hover {
        background-color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2>Job Verify</h2>
            <div>
                <button class="nav-button" onclick="window.location.href='/Predict'">Predict</button>
                <button class="nav-button" onclick="window.location.href='/Compare'">Compare</button>
                <button class="nav-button" onclick="window.location.href='/Understanding'">About</button>
                <button class="nav-button" onclick="toggleDarkMode()">Toggle Dark Mode</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Predicting fake job postings...**")
    st.button("Start Verify")
    
    # Flashing text effect
    st.markdown("""
    <style>
    .flashing {
        animation: flashing 1s infinite;
    }
    @keyframes flashing {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="flashing">This is a flash text!</p>', unsafe_allow_html=True)

# Function for predict page
def show_predict_page():
    st.title("Predict If Job is Real or Fake")
    
    text = st.text_area('**Enter Job Description**')
    ok = st.button("Predict")

    if ok:
        st.write("**Input Text**")
        st.write(text)

        text = spacy_process(text)
        st.write("**After Text-Preprocessing**")
        st.write(text)

        data = {'text': [text]}
        df = pd.DataFrame(data)

        model_data = load_model()
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]

        x = vectorizer.transform(df.loc[:, 'text'])
        temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

        prediction = model.predict(temp)

        if prediction[0] == 1:
            st.markdown("<h1 style='color:red;'>Job is Fake! ❌</h1>", unsafe_allow_html=True)
        elif prediction[0] == 0:
            st.markdown("<h1 style='color:green;'>Job is Real! ✅</h1>", unsafe_allow_html=True)

# Function for compare page
def compare_model_page():
    st.title("Compare Models")
    st.write("### Model Comparison")
    
    # Center align elements with border
    with st.container():
        st.markdown("<div style='border: 2px solid #ddd; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        
        button = False
        df = pd.read_csv('clean_df.csv')
        
        # Vectorizer Configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            gram = st.selectbox("**Select Grams**", ("Mono-Gram", "Bi-Gram", "Tri-Gram"))
            gram = (1, 1) if gram == "Mono-Gram" else (2, 2) if gram == "Bi-Gram" else (3, 3)

        with col2:
            no_features = st.slider('**Select Max-Features**', 1, 1000, 100)

        with col3:
            vec = st.selectbox("**Select Vectorizer**", ("Count", "TF-IDF"))

        model = st.selectbox("**Select Model**", ("Logistic Regression", "Random Forest", "Support Vector Machine"))

        # Data Configuration
        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider('**Select Test Size**', 10, 100, 30) / 100

        with col2:
            over_sample = st.selectbox('**Do Over-Sampling**', ['Yes', 'No'])
            over_sample = True if over_sample == 'Yes' else False

        st.write("#### 3. Model Configuration")
        if model == "Logistic Regression":
            col1, col2 = st.columns(2)
            with col1:
                penalty = st.selectbox("**Select Penalty**", ("l1", "l2", "elasticnet"))
                random_state = st.slider('**Select Random State**', 1, 1000, 42)
            with col2:
                solver = st.selectbox("**Select Solver**", ("liblinear", "newton-cg", "newton-cholesky", "sag", "saga"))
                n_jobs = st.slider('**Select N-Jobs**', 1, 1000, 42)

            model = LogisticRegression(
                penalty=penalty,
                solver=solver,
                random_state=random_state,
                n_jobs=n_jobs
            )
            
            train = st.button("Train")

            if train:
                st.write("#### 4. Model Evaluation")
                trainer(df, test_size, over_sample, vectorizer, model)
                button = st.button('Save Logistic Regression as Pickle')

        # Similar structure for Random Forest and SVC...

        st.markdown("</div>", unsafe_allow_html=True)

# Function for understanding page
def understanding_page():
    st.title("Understanding the Data")
    # Align graphs horizontally
    st.write("#### Data Visualizations")
    
    # Add your graphs here
    # e.g., sns.countplot, plt.show(), st.pyplot()

# Function for contact page
def contact_page():
    st.title("Contact Us")
    st.write("Share your feedback!")
    
    email = st.text_input("Your Email:")
    review = st.text_area("Your Review:")
    
    if st.button("Post"):
        st.success("Feedback submitted!")

# Load model function
def load_model():
    with open('notebook_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Run app
page = create_navbar()

if page == "Home":
    home_page()
elif page == "Predict":
    show_predict_page()
elif page == "Compare":
    compare_model_page()
elif page == "Understanding":
    understanding_page()
elif page == "Contact":
    contact_page()
