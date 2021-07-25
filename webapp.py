import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import contractions
import re
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import streamlit as st
from tempfile import NamedTemporaryFile



st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)








def preprocess(text, stem=True, stop_w = False):

    text = contractions.fix(text)

    text = text.replace("\n", " ")

    # Remove link,user and special characters
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[0-9]|[^A-Za-z0-9]+"
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    text = re.sub(r" s ", " ", text)

    tokens = []
    for token in text.split():

      if stop_w:

        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)

      else:

            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)


    return " ".join(tokens)






loaded_model = joblib.load(r"C:\Users\Admin\Desktop\xray-streamit\email-cat\model1.sav")
# loading pickled vectorizer
cv_fit = joblib.load(r"C:\Users\Admin\Desktop\xray-streamit\email-cat\cv1.pkl")

st.write("""
# Email Classifier
by Hardik :)
""")

uploaded_file = st.file_uploader("Excel File",type="csv")

if uploaded_file is not None:
	tdf = pd.read_csv(uploaded_file )
	tdf['clean_text'] = tdf['Email Replies'].apply(lambda x: preprocess(x,False,False))
	sample_input = cv_fit.transform(tdf['clean_text'])
	sample_output = loaded_model.predict(sample_input)
	result_df_t = tdf[['Email Replies','Category']].copy()
	result_df_t['pred_cat_num'] = sample_output.copy()
	result_df_t['pred_cat_label'] = result_df_t['pred_cat_num'].copy()
	result_df_t['pred_cat_label'].replace({0:'Meeting Accepted',1:'Question/Queries',2:'Out of office',3:'Unsubscribe'},inplace=True)

	st.dataframe(result_df_t)
	
else:
	st.warning("you need to upload a excel file.")


