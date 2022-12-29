import streamlit as st
import pandas as pd
import numpy as np 
import pickle # to load a saved model
import base64 # to open .gif files in streamlit
import sklearn


# Set the navigation
# Home page: show data selected, preparation and modelling
# Prediction page: allow input of features, and click predict button to show 'Good'/'Bad' wine


app_mode=st.sidebar.selectbox('Select Page',['Home', 'Prediction']) # two pages

# if 'Home' page selected
if app_mode=='Home':

	tab1, tab2, tab3, tab4, tab5 = st.tabs(["Business Objective", "Data", "Classification ML Models"
		, "Pickle the Best Model", "Deployment"])
	with tab1:
		st.image("Presentation1.gif")

	with tab2:
		st.markdown("Get the python libraries")
		st.image("Data_Libraries.png")
		st.write("***")
		st.markdown("Clean or dirty?")
		st.image("Data_NoNulls.png")
		st.write("***")
		st.markdown("Data types?")
		st.image("Data_DataTypes.png")
		st.write("***")
		st.markdown("Quality (y) vs. Features (X)")
		st.image("Data_CorrMatrix.png")
		st.write("***")
		st.markdown("Distribution of Quality (y)")
		st.image("Data_Quality.png")
		st.write("***")
		st.markdown("Binarization of Quality (y)")
		st.image("Data_Binarization.png")

	with tab3:
		st.markdown("Store features as X, target variable as y")
		st.markdown("Perform train_test_split")
		st.image("ML_Xy_train_test_split.png")
		st.write("***")
		st.markdown("Experimenting with Five Classifcication Models")
		st.image("ML_fiveModels.png")
		st.write("***")
		st.markdown("Accuracy with SMOTE (Oversampling)")
		st.image("ML_SMOTE.png")
		st.write("***")
		st.markdown("Accuracy without SMOTE (Oversampling)")
		st.image("ML_NoSMOTE.png")

	with tab4:
		st.markdown("Scoreboard: Accuracy, Precision, Recall, F1")
		st.image("Pickle_ScoreBoard.png")
		st.write("***")
		st.markdown("Visualizing the Score Board")
		st.image("Pickle_ScoreBoard2.png")
		st.write("***")
		st.markdown("Accuracy Score Board")
		st.image("Pickle_AccuracyScoreBoard.png")

	with tab5:
		st.markdown("RF Model. Full Dataset. Save the Model")
		st.image("Deployment_pickleRF.png")
		st.write("***")
		st.text("Deploy via Streamlit")
		st.text("In command prompt, simply type streamlit run <filename>.py")

# 'Prediction page' selected
elif app_mode=='Prediction':

	st.subheader('Welcome to Red Wine Quality Prediction App')
	st.sidebar.header("Input details about the wine: ")
	## input all the features
	FixedAcidity=st.sidebar.slider("Fixed Acidity",0.0,2.0,15.0)
	VolatileAcidity=st.sidebar.slider("Volatile Acidity",0.0,2.0,10.0)
	CitricAcid=st.sidebar.slider("CitricAcid",0.0,2.0,10.0)
	ResidualSugar=st.sidebar.slider("Residual Sugar",0.0,2.0,10.0)
	Chlorides=st.sidebar.slider("Chlorides",0.0,2.0,10.0)
	FreeSulfurDioxide=st.sidebar.slider("Free Sulfur Dioxide",0.0,2.0,20.0)
	TotalSulfurDioxide=st.sidebar.slider("Total Sulfur Dioxide",0.0,2.0,40.0)
	Density=st.sidebar.slider("Density",0.0,2.0,5.0)
	pH=st.sidebar.slider("pH",0.0,2.0,7.0)
	Sulphates=st.sidebar.slider("Sulphates",0.0,2.0,3.0)
	Alcohol=st.sidebar.slider("Alcohol",0.0,2.0,40.0)


	data1 = {
	'Fixed Acidity': FixedAcidity,
	'VolatileAcidity':VolatileAcidity,
	'CitricAcid':CitricAcid,
	'ResidualSugar':ResidualSugar,
	'Chlorides':Chlorides,
	'FreeSulfurDioxide':FreeSulfurDioxide,
	'TotalSulfurDioxide':TotalSulfurDioxide,
	'density':Density,
	'pH':pH,
	'Sulphates':Sulphates,
	'alcohol':Alcohol
	}

	feature_list=[
		FixedAcidity,VolatileAcidity,CitricAcid,ResidualSugar,
		Chlorides,FreeSulfurDioxide,TotalSulfurDioxide,Density,pH,Sulphates,Alcohol]

	single_sample=np.array(feature_list).reshape(1,-1)

# Upon click the 'Predict' button
	if st.button("Predict"):

		loaded_model=pickle.load(open('rf.pkl','rb'))
		prediction=loaded_model.predict(single_sample)
		if prediction[0]==0:
			st.error("Bad!")
		elif prediction[0]==1:
			st.success("Good!")




