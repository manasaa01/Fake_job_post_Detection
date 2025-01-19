import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from numpy import sqrt, argmax
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
from spacy.cli import download
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Download Spacy model if not already present
download("en_core_web_sm")

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Function to load model from pickle
def load_model():
    with open('app_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
st.set_page_config(layout="wide")

     

# Home Page
# Function to set the theme based on user choice

# Sidebar for navigation

# Custom CSS for the animated gradient text
st.markdown(
    """
    <style>
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .animated-title {
        font-size: 3em; /* Adjust size as needed */
        font-weight: bold;
        text-align: center;
        background: linear-gradient(-45deg, #ff7eb3, #ff758c, #ff7b5c, #ff9e00);
        background-size: 400% 400%;
        color: transparent;
        -webkit-background-clip: text;
        animation: gradient 5s ease infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Adding the animated title
st.markdown('<h1 class="animated-title"> ⛉ DETECT FAKE JOBS</h1>', unsafe_allow_html=True)


# Theme toggle functionality
with st.container():
    selected = option_menu(
        menu_title=None,  # None to remove the menu title
        options=['Home', 'Explore', 'Models', 'Checkout', 'About'],
        icons=['house-fill', 'globe','code-slash', 'pen',],
        orientation='horizontal'
    )
# Main content for Home Page

import streamlit as st

if selected == 'Home':
    # Set the title of the app
    st.title("Welcome to the Future of Job Safety")

    # Create two columns
    col1, col2 = st.columns([2, 1])  # Left column is wider

    with col1:
        # Add information on the left side
        st.markdown("""
           ***Protect yourself from job scams with our AI-powered detection system***
            In today's digital world, job seekers are often faced with misleading job postings that can lead
            to scams and fraudulent activities. Our tool helps you identify fake job listings instantly - 
            making your job hunt safer and more secure.Job hunting made safer 
                    - Powered by machine learning to protect you from fraud.
                    """)
        
        # Add a checkout button with animation
        st.markdown(
            """
            <style>
            .animated-button {
                display: inline-block;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                background-color: #FF5733;
                border: none;
                border-radius: 5px;
                text-align: center;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.3s;
            }
            .animated-button:hover {
                background-color: #C70039;
                text-color: #ffffff;
                transform: scale(1.1);
            }
            </style>
            <button class="animated-button">Get started</button>
            """, unsafe_allow_html=True
        )

    with col2:
        # Add the scrolling message above the image
        st.markdown(
            """
            <div style="overflow: hidden; white-space: nowrap; width: 100%;">
                <marquee behavior="scroll" direction="left" scrollamount="5" style="color: #FF5733; font-size: 20px;">
                    Beware of scams! Use our tool to predict fake job postings!
                </marquee>
            </div>
            """, unsafe_allow_html=True
        )

        # Add an image below the scrolling message
        st.image("download-removebg-preview.jpg", use_column_width=True, caption="Identify fake job postings effortlessly!")
# Add logic for other pages if necessary
# Explore Page
if selected == "Explore":
    # Loading Dataset into DataFrame
    st.write("#### 1. About Dataset")
    df = pd.read_csv("fake_job_postings.csv")
    st.dataframe(df.head())

    rows = df.shape[0]
    cols = df.shape[1]
    st.write("This dataset has", rows, "rows and ", cols, "columns.")

    st.write("#### 2. Exploratory Data Analysis")
    st.write("##### 2.1 Missing Values")
    fig = sns.set(rc={'figure.figsize': (8, 5)})
    fig, ax = plt.subplots()
    plt.title("Heat Map for Missing Values")
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
    st.pyplot(fig)

    # Filling Na with Blank Spaces
    df.fillna('', inplace=True)

    st.write("##### 2.2 Comparing Number of Fraudulent and Non-Fraudulent Job Postings")
    fig = sns.set(rc={'figure.figsize': (10, 3)})
    fig, ax = plt.subplots()
    plt.title("Number of Fradulent Vs Non-Fraudlent Jobs")
    sns.countplot(y='fraudulent', data=df)
    st.pyplot(fig)

    not_fraudulent = df.groupby('fraudulent')['fraudulent'].count()[0]
    fraudulent = df.groupby('fraudulent')['fraudulent'].count()[1]
    st.write(f"{not_fraudulent} jobs are NOT Fraudulent and {fraudulent} jobs are Fraudulent.")

    st.write("##### 2.3 Experience-wise Count")
    exp = dict(df.required_experience.value_counts())
    del exp['']  # Remove empty keys

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    sns.set_theme(style="whitegrid")
    plt.bar(exp.keys(),exp.values())
    plt.title('No. of Jobs with Experience')
    plt.xlabel('Experience')
    plt.ylabel('No. of jobs')
    plt.xticks(rotation=30)
    st.pyplot(fig)
    st.write("##### 2.4 Countrywise Job Count")

    # First Spliting location Column to extract Country Code
    def split(location):
        l = location.split(',')
        return l[0]

    df['country'] = df.location.apply(split)

    countr = dict(df.country.value_counts()[:14])
    del countr['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    plt.title('Country-wise Job Posting')
    plt.bar(countr.keys(), countr.values())
    plt.ylabel('No. of jobs')
    plt.xlabel('Countries')
    st.pyplot(fig)

    st.write("##### 2.5 Education Job Count")

    edu = dict(df.required_education.value_counts()[:7])
    del edu['']

    fig = sns.set(rc={'figure.figsize': (10, 5)})
    fig, ax = plt.subplots()
    plt.title('Job Posting based on Education')
    plt.bar(edu.keys(), edu.values())
    plt.ylabel('No. of jobs')
    plt.xlabel('Education')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write("##### 2.6 Top 10 Titles of Jobs Posted which were NOT fraudulent")

    dic = dict(df[df.fraudulent==0].title.value_counts()[:10])
    dic_df = pd.DataFrame.from_dict(dic, orient ='index')
    dic_df.columns = ["Number of Jobs"]
    st.dataframe(dic_df)

    st.write("##### 2.7 Top 10 Titles of Jobs Posted which were fraudulent")

    dic = dict(df[df.fraudulent==1].title.value_counts()[:10])
    dic_df = pd.DataFrame.from_dict(dic, orient ='index')
    dic_df.columns = ["Number of Jobs"]
    st.dataframe(dic_df)


    # More plots and analysis...
    # You can continue adding similar sections as in your original code for more insights
# functions

# Creating a Dataframe with word-vectors in TF-IDF form and Target values

def final_df(df, is_train, vectorizer, column):

    # TF-IDF form
    if is_train:
        x = vectorizer.fit_transform(df.loc[:,column])
    else:
        x = vectorizer.transform(df.loc[:,column])

    # TF-IDF form to Dataframe
    temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

    # Droping the text column
    df.drop(df.loc[:,column].name, axis = 1, inplace=True)

    # Returning TF-IDF form with target
    return pd.concat([temp, df], axis=1)


# Training the model with various combination and returns y_test and y_pred

def train_model(df, input, target, test_size, over_sample, vectorizer, model):

    X = df.drop(target, axis=1)
    y = df[target]
    print("Splitted Data into X and Y.")

    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Splitted Data into Train and Test.")
    
    # Training Preprocessing
    X_train = final_df(X_train, True, vectorizer, input)
    X_train.dropna(inplace=True)
    print("Vectorized Training Data.")

    if over_sample:
        sm = SMOTE(random_state = 2)
        X_train, Y_train = sm.fit_resample(X_train, Y_train.ravel())
        print("Oversampling Done for Training Data.")

    # Testing Preprocessing
    x_test = final_df(x_test, False, vectorizer, input)
    x_test.dropna(inplace=True)
    print("Vectorized Testing Data.")

    # fitting the model
    model = model.fit(X_train, Y_train)
    print("Model Fitted Successfully.")

    # calculating y_pred
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)

    return model, x_test, y_test, y_pred_prob

def evaluate(y_test, y_pred, y_pred_prob):
    roc_auc = round(roc_auc_score(y_test, y_pred_prob[:, 1]), 2)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1], pos_label=1)
    
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    
    # locate the index of the largest g-mean
    ix = argmax(gmeans)

    y_pred = (y_pred > thresholds[ix])

    accuracy = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**ROC-AUC Score** \t\t: {roc_auc*100} %")
        st.write('**Best Threshold** \t\t: %.3f' % (thresholds[ix]))
    with col2:
        st.write('**G-Mean** \t\t\t: %.3f' % (gmeans[ix]))
        st.write(f"**Model Accuracy** : {round(accuracy,2,)*100} %")

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

def trainer(df, test_size, over_sample, vectorizer, model):
    model, x_test, y_test, y_pred_prob = train_model(
        df=df, 
        input='text', 
        target='fraudulent', 
        test_size=test_size,
        over_sample=over_sample, 
        vectorizer=vectorizer, 
        model=model)

    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)

    evaluate(y_test, y_pred, y_pred_prob)


nlp = spacy.load('en_core_web_sm')

# Text Preprocessing with varoius combination

def spacy_process(text):
  # Converts to lowercase
  text = text.strip().lower()

  # passing text to spacy's nlp object
  doc = nlp(text)
    
  # Lemmatization
  lemma_list = []
  for token in doc:
    lemma_list.append(token.lemma_)
  
  # Filter the stopword
  filtered_sentence =[] 
  for word in lemma_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
      filtered_sentence.append(word)
    
  # Remove punctuation
  punctuations="?:!.,;$\'-_"
  for word in filtered_sentence:
    if word in punctuations:
      filtered_sentence.remove(word)

  return " ".join(filtered_sentence)

# For Loading the Pickle File
def load_model():
    with open('notebook_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
# Compare Models Page


if selected == "Models":

    button = False

    

    df = pd.read_csv('clean_df.csv')

    st.write("#### 1. Vectorizer Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        gram = st.selectbox("**Select Grams**", ("Mono-Gram", "Bi-Gram", "Tri-Gram"))
        
        if gram == "Mono-Gram":
            gram = (1,1)
        elif gram == "Bi-Gram":
            gram = (2,2)
        elif gram == "Tri-Gram":
            gram = (3,3)

    with col2:
        no_features = st.slider('**Select Max-Features**', 1, 1000, 100)

    with col3:
        vec = st.selectbox("**Select Vectorizer**", ("Count", "TF-IDF"))

    if vec == "Count":
        vectorizer = CountVectorizer(ngram_range=gram, max_features = no_features)
    elif vec == "TF-IDF":
        vectorizer = TfidfVectorizer(ngram_range=gram, max_features = no_features)

    model = st.selectbox("**Select Model**", ("Logistic Regression","Random Forest","Support Vector Machine"))

    st.write("#### 2. Data Configuration")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider('**Select Test Size**', 10, 100, 30)
        test_size = test_size/100

    with col2:
        over_sample = st.selectbox('**Do Over-Sampling**', ['Yes', 'No'])
        if over_sample == 'Yes':
            over_sample = True
        elif over_sample == 'No':
            over_sample = False

    st.write("#### 3. Model Configuration")

    if model == "Logistic Regression":

        col1, col2 = st.columns(2)

        with col1:
            penalty = st.selectbox("**Select Penalty**", ("l1","l2","elasticnet"))
            random_state = st.slider('**Select Random State**', 1, 1000, 42)
        with col2:
            solver = st.selectbox("**Select Solver**", ("liblinear","newton-cg", "newton-cholesky", "sag", "saga"))
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

    elif model == "Random Forest":
        col1, col2 = st.columns(2)

        with col1:
            criterion = st.selectbox("**Select Criterion**", ("gini","entropy","elasticnet"))
            n_estimators = st.slider('**Select N-Estimatorse**', 1, 1000, 100)
            n_jobs = st.slider('**Select N-Jobs**', 1, 1000, 10)
        with col2:
            max_features = st.selectbox("**Select Max-Features**", ("sqrt","log2"))
            max_depth = st.slider('**Select Max-Depth**', 1, 50, 10)
            random_state = st.slider('**Select Random-State**', 1, 1000, 42)

        model = RandomForestClassifier(
            criterion=criterion,
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

        train = st.button("Train")

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Random Forest as Pickle')

    elif model == "Support Vector Machine":
        col1, col2 = st.columns(2)

        with col1:
            kernel = st.selectbox("**Select Kernel**", ("linear","poly","rbf", "sigmoid"))

        with col2:
            random_state = st.slider('**Select Random State**', 1, 1000, 42)

        model = SVC(
            kernel=kernel,
            random_state=random_state,
            probability=True
        )

        train = st.button("Train")
        

        if train:
            st.write("#### 4. Model Evaluation")
            trainer(df, test_size, over_sample, vectorizer, model)
            button = st.button('Save Support Vector Machine as Pickle')

    if button:
        data = {"model": model}
        with open(r'app.pkl', 'wb') as file:
            pickle.dump(data, file)

    # Configuration for model training and selection

# prediction 

# Function to create a styled text area
if selected == "Checkout":
    st.markdown("""
        **Add the description and check the results**
    """)

    text = st.text_area('**Enter Job Description**')

    ok = st.button("Predict")

    if ok:
        st.write("**Input Text**")
        st.write(text)

        text = spacy_process(text)
        st.write("**After Text-Preprocessing**")
        st.write(text)

        data = {
            'text': [text]
        }

        df = pd.DataFrame(data)

        data = load_model()  # Assuming this returns a dict with 'model' and 'vectorizer'
        model = data["model"]
        vectorizer = data["vectorizer"]

        # Transform the text into vectorized form
        x = vectorizer.transform(df.loc[:, 'text'])
        temp = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())

        # Make a prediction
        prediction = model.predict(temp)

        # Check prediction result and display appropriate message
        if prediction[0] == 1:
            st.markdown(unsafe_allow_html=True,
                body="""
                    <style>
                        .fake-job {
                            color: red;
                            font-size: 50px;
                            animation: shake 0.5s;
                        }

                        @keyframes shake {
                            0% { transform: translate(0, 0); }
                            25% { transform: translate(-5px, 0); }
                            50% { transform: translate(5px, 0); }
                            75% { transform: translate(-5px, 0); }
                            100% { transform: translate(0, 0); }
                        }
                    </style>
                    <span class='fake-job'><strong><h4>❌ Job is Fake! :</h4></strong></span>
                    """)
        elif prediction[0] == 0:
            st.markdown(unsafe_allow_html=True,
                body="""
                    <style>
                        .real-job {
                            color: green;
                            font-size: 50px;
                            animation: bounce 0.5s;
                        }

                        @keyframes bounce {
                            0%, 100% { transform: translateY(0); }
                            50% { transform: translateY(-10px); }
                        }
                    </style>
                    <span class='real-job'><strong><h3>✅ Job is Real! :</h3></strong></span>
                    """
            )

# About Page Function
if selected =="About" :
    st.title("About Us")
    st.write("""
        This application helps in detecting fake job postings using machine learning models. 
        Our mission is to provide users with a reliable tool to differentiate between genuine 
        and fraudulent job postings, ensuring a safer job search experience.
    """)

    st.header("Submit Your Review")

    # Text area for review
    review = st.text_area("Write your review here:", placeholder="Share your experience...")

   
    # Submit button for the review
    if st.button("Submit Review"):
        if review :
            st.success(f"Thank you for your review! You rated us {rating} stars.")
        elif not review:
            st.warning("Please write a review before submitting.")
        else:
            st.warning("Please rate before submitting.")


