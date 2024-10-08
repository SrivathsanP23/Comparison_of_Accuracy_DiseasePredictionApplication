import os
import pickle
import numpy
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import r2_score

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the models
svm_model = pickle.load(open('parkinsons_model_svm.sav', 'rb'))
logistic_model = pickle.load(open('smartdiseaseprediction_logreg.sav', 'rb'))
ran_cls = pickle.load(open('randomforestmodel.sav','rb'))
ens_cls=pickle.load(open('smartdiseaseprediction_ensemble.sav','rb'))
# Load your dataset
# Assuming the dataset is in CSV format, replace 'your_dataset.csv' with your actual dataset file
data_path = 'parkinsons.csv'
data = pd.read_csv(data_path)

# Replace 'feature_columns' with the actual feature columns and 'target_column' with the actual target column
feature_columns = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 
                   'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
                   'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
target_column = 'status'  # Replace with your actual target column name

X = data[feature_columns]
Y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

# Apply SMOTE to the scaled training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scale, y_train)
log_reg = LogisticRegression(C=0.1, class_weight='balanced')
svm = SVC(C=10, probability=True, class_weight='balanced')
rf = RandomForestClassifier(max_depth=10, n_estimators=100, class_weight='balanced')

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('svc', svm), ('rf', rf)], voting='soft',weights=[1,2,1])

# Train the model
voting_clf.fit(X_train_smote, y_train_smote)

# Calculate accuracies for SVM
svm_train_predictions = svm_model.predict(X_train_smote)
svm_train_accuracy = accuracy_score(y_train_smote, svm_train_predictions)
svm_test_predictions = svm_model.predict(X_test_scale)
svm_test_accuracy = accuracy_score(y_test, svm_test_predictions)
svm_r = r2_score(y_test,svm_test_predictions)

# Calculate accuracies for Logistic Regression
logistic_train_predictions = logistic_model.predict(X_train_smote)
logistic_train_accuracy = accuracy_score(y_train_smote, logistic_train_predictions)
logistic_test_predictions = logistic_model.predict(X_test_scale)
logistic_test_accuracy = accuracy_score(y_test, logistic_test_predictions)
log_r=r2_score(y_test,logistic_test_predictions)
#calculate accuracies for RandomForest Classifier
randomforest_train_prediction = ran_cls.predict(X_train_smote)
randomforest_train_accuracy = accuracy_score(y_train_smote,randomforest_train_prediction)
randomforest_test_prediction = ran_cls.predict(X_test_scale)
randomforest_test_accuracy = accuracy_score(y_test,randomforest_test_prediction)
ranf_r=r2_score(y_test,randomforest_test_prediction)

#calculate for ensemble
ensemble_train_prediction = voting_clf.predict(X_train_smote)
ensemble_train_accuracy = accuracy_score(y_train_smote,ensemble_train_prediction)
ensemble_test_prediction = voting_clf.predict(X_test_scale)
ensemble_test_accuracy = accuracy_score(y_test,ensemble_test_prediction)
ens_r=r2_score(y_test,ensemble_test_prediction)

# Page title
st.title("Smart Disease Prediction System")

# Tab-based navigation
tab1, tab2 ,tab3, tab4= st.tabs(["SVM Model", "Logistic Regression Model","RandomForest Model","Ensemble"])

# Initialize a dictionary to store user inputs
user_input = {}

# Function to get user inputs
def get_user_inputs(prefix):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        user_input['fo'] = st.slider('MDVP-Fo(Hz)', min_value=0.0, max_value=300.0, step=0.1, key=f'{prefix}_fo')

    with col2:
        user_input['fhi'] = st.slider('MDVP-Fhi(Hz)', min_value=0.0, max_value=300.0, step=0.1, key=f'{prefix}_fhi')

    with col3:
        user_input['flo'] = st.slider('MDVP-Flo(Hz)', min_value=0.0, max_value=300.0, step=0.1, key=f'{prefix}_flo')

    with col4:
        user_input['Jitter_percent'] = st.slider('MDVP-Jitter(%)', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_Jitter_percent')

    with col5:
        user_input['Jitter_Abs'] = st.slider('MDVP-Jitter(Abs)', min_value=0.0, max_value=0.1, step=0.0001, key=f'{prefix}_Jitter_Abs')

    with col1:
        user_input['RAP'] = st.slider('MDVP-RAP', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_RAP')

    with col2:
        user_input['PPQ'] = st.slider('MDVP-PPQ', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_PPQ')

    with col3:
        user_input['DDP'] = st.slider('Jitter-DDP', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_DDP')

    with col4:
        user_input['Shimmer'] = st.slider('MDVP-Shimmer', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_Shimmer')

    with col5:
        user_input['Shimmer_dB'] = st.slider('MDVP-Shimmer(dB)', min_value=0.0, max_value=10.0, step=0.1, key=f'{prefix}_Shimmer_dB')

    with col1:
        user_input['APQ3'] = st.slider('Shimmer-APQ3', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_APQ3')

    with col2:
        user_input['APQ5'] = st.slider('Shimmer-APQ5', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_APQ5')

    with col3:
        user_input['APQ'] = st.slider('MDVP-APQ', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_APQ')

    with col4:
        user_input['DDA'] = st.slider('Shimmer-DDA', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_DDA')

    with col5:
        user_input['NHR'] = st.slider('NHR', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_NHR')

    with col1:
        user_input['HNR'] = st.slider('HNR', min_value=0.0, max_value=50.0, step=0.1, key=f'{prefix}_HNR')

    with col2:
        user_input['RPDE'] = st.slider('RPDE', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_RPDE')

    with col3:
        user_input['DFA'] = st.slider('DFA', min_value=0.0, max_value=2.0, step=0.01, key=f'{prefix}_DFA')

    with col4:
        user_input['spread1'] = st.slider('spread1', min_value=-10.0, max_value=10.0, step=0.01, key=f'{prefix}_spread1')

    with col5:
        user_input['spread2'] = st.slider('spread2', min_value=-10.0, max_value=10.0, step=0.01, key=f'{prefix}_spread2')

    with col1:
        user_input['D2'] = st.slider('D2', min_value=0.0, max_value=10.0, step=0.01, key=f'{prefix}_D2')

    with col2:
        user_input['PPE'] = st.slider('PPE', min_value=0.0, max_value=1.0, step=0.001, key=f'{prefix}_PPE')

# SVM Model Tab
with tab1:
    st.header("SVM Model")
    
    #st.write(f"Test Accuracy: {svm_test_accuracy * 100:.2f}%")
    get_user_inputs(prefix='svm')

    # Code for Prediction
    if st.button("Parkinson's Test Result (SVM)"):
        # Prepare the input for the model
        input_values = list(user_input.values())
        inpt = numpy.asarray(input_values)
        inpt_reshape = inpt.reshape(1,-1)
        std_data=scaler.transform(inpt_reshape)

        # Make the prediction
        parkinsons_prediction = svm_model.predict(std_data)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            st.write(f"Training Accuracy: {svm_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {svm_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {svm_r :.4f}")
            
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.write(f"Training Accuracy: {svm_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {svm_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {svm_r :.4f}")
            

        st.success(parkinsons_diagnosis)

# Logistic Regression Model Tab
with tab2:
    st.header("Logistic Regression Model")
    #st.write(f"Test Accuracy: {logistic_test_accuracy * 100:.2f}%")
    get_user_inputs(prefix='logistic')

    # Code for Prediction
    if st.button("Parkinson's Test Result (Logistic Regression)"):
        # Prepare the input for the model
        input_values = list(user_input.values())
        inpt = numpy.asarray(input_values)
        inpt_reshape = inpt.reshape(1,-1)
        std_data=scaler.transform(inpt_reshape)

        # Make the prediction
        parkinsons_prediction = logistic_model.predict(std_data)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            st.write(f"Training Accuracy: {logistic_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {logistic_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {log_r :.4f}")

        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.write(f"Training Accuracy: {logistic_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {logistic_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {log_r :.4f}")


        st.success(parkinsons_diagnosis)
with tab3:
    st.header("RandomForest Model")
    #st.write(f"Test Accuracy: {logistic_test_accuracy * 100:.2f}%")
    get_user_inputs(prefix='randomforest')

    # Code for Prediction
    if st.button("Parkinson's Test Result (Randomforest)"):
        # Prepare the input for the model
        input_values = list(user_input.values())
        inpt = numpy.asarray(input_values)
        inpt_reshape = inpt.reshape(1,-1)
        std_data=scaler.transform(inpt_reshape)

        # Make the prediction
        parkinsons_prediction = ran_cls.predict(std_data)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            st.write(f"Training Accuracy: {randomforest_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {randomforest_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {ranf_r :.4f}")
            
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.write(f"Training Accuracy: {randomforest_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {randomforest_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {ranf_r :.4f}")


        st.success(parkinsons_diagnosis)
    
with tab4:
    st.header("Ensemble Model")
    #st.write(f"Test Accuracy: {logistic_test_accuracy * 100:.2f}%")
    get_user_inputs(prefix='ensemble')

    # Code for Prediction
    if st.button("Parkinson's Test Result (Ensemble)"):
        # Prepare the input for the model
        input_values = list(user_input.values())
        inpt = numpy.asarray(input_values)
        inpt_reshape = inpt.reshape(1,-1)
        std_data=scaler.transform(inpt_reshape)

        # Make the prediction
        parkinsons_prediction = voting_clf.predict(std_data)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            st.write(f"Training Accuracy: {ensemble_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {ensemble_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {ens_r :.4f}")
            
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.write(f"Training Accuracy: {ensemble_train_accuracy * 100:.3f}%")
            st.write(f"Testing Accuracy: {ensemble_test_accuracy * 100:.3f}%")
            st.write(f"r2 score: {ens_r :.4f}")


        st.success(parkinsons_diagnosis)
    
