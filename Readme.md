# Fake Job Posting Detection

A machine learning project to detect fraudulent job postings using the Employment Scam Aegean Dataset (EMSCAD) from Kaggle. The system uses Logistic Regression and Random Forest models to classify job postings as real or fraudulent, with a Streamlit web interface for easy interaction.

## Dataset
The project uses the Employment Scam Aegean Dataset (EMSCAD) from Kaggle, which contains:
- Real and fraudulent job postings
- Features like job description, company profile, requirements
- Binary classification labels (fraudulent/real)

## Technologies Used
- **Development Environment**:
  - Jupyter Notebook for analysis and model development
- **Data Processing**: 
  - Pandas for data manipulation
  - NumPy for numerical operations
- **Machine Learning**: 
  - Scikit-learn's Logistic Regression
  - Random Forest Classifier
- **User Interface**: 
  - Streamlit
  - Custom CSS for styling

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Jupyter Notebook/JupyterLab

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-job-detection.git
cd fake-job-detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the EMSCAD dataset from Kaggle and place it in the `data` directory.

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```



## Jupyter Notebooks
The project includes several Jupyter notebooks for different stages of development:

1. **Data_Exploration.ipynb**
   - Initial data analysis
   - Feature distribution analysis
   - Missing value detection
   - Data visualization

2. **Data_Preprocessing.ipynb**
   - Data cleaning
   - Text preprocessing
   - Feature engineering
   - Data transformation

3. **Model_Training.ipynb**
   - Model selection
   - Hyperparameter tuning
   - Training process
   - Cross-validation

4. **Model_Evaluation.ipynb**
   - Performance metrics
   - Model comparison
   - Feature importance analysis
   - Error analysis

## Usage

### Running the Notebooks
1. Start Jupyter Notebook server:
```bash
jupyter notebook
```
2. Navigate to the `notebooks` directory
3. Open notebooks in sequential order

### Running the Web Application
```bash
cd src
streamlit run app.py
```
## Features
- Comprehensive data analysis using Jupyter Notebooks
- Data preprocessing and cleaning using Pandas
- Text feature extraction
- Model training with Logistic Regression and Random Forest
- Interactive web interface using Streamlit
- Real-time prediction capabilities
- Custom CSS styling for better user experience

## Web Interface
The Streamlit interface provides:
- Input fields for job posting details
- Real-time prediction results
- Probability scores for fraud detection
- Visual representation of results
- User-friendly design with custom CSS

## Acknowledgments
- EMSCAD dataset providers on Kaggle
- Streamlit community for UI components
- scikit-learn documentation and community
