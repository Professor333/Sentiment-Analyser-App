import streamlit as st

st.title("Customer Review Sentiment Analyser")
st.markdown("This app analyse the sentiment of customer reviews to gain insights into their opinion.")


from openai import OpenAI
import pandas as pd

#open ai key input
openai_apikey=st.sidebar.text_input('enter your open ai key',
                                   type="password")

def classify_sentiment_openai(review_text):
    client = OpenAI(api_key=openai_apikey)
    prompt = f'''
        Classify the following customer review. 
        State your answer
        as a single word, "positive", 
        "negative" or "neutral":

        {review_text}
        '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content


#user_input = st.text_input("Enter a customer review ")
#st.write("The current user review is:", user_input)

#reading csv file from users
#df=pd.read_csv('restaurant_reviews.csv')
#st.write(df)

uploaded_file=st.file_uploader('upload a review file that you want to analyse',
                               type=['csv'])

# once user uploaded file, then read file

if uploaded_file is not None:
    # read file
    reviews_df=pd.read_csv(uploaded_file)

    #check if data has text column 
    text_columns=reviews_df.select_dtypes(include='object').columns

if len(text_columns)==0:
    st.error('No text column found with customer review files. Check again!')

# show a drop down for columns
review_column=st.selectbox('Select the column with text review',
                           text_columns
)

# Analysing the sentiments

reviews_df['Sentiment']=reviews_df[review_column].apply(classify_sentiment_openai)


#sentiment distribution
reviews_df['Sentiment']=reviews_df['Sentiment'].str.title()
sentiment_count=reviews_df['Sentiment'].value_counts()
#st.write(reviews_df)
#st.write(sentiment_count)


#create 3 columns to display 3 metrics

col1,col2,col3=st.columns(3)

with col1:
    positive_count=sentiment_count.get('Positive',0)
    st.metric('Positive',positive_count,
              f"{positive_count /len(reviews_df) *100: .2f}%")

with col2:
    negative_count=sentiment_count.get('Negative',0)
    st.metric('Negative',negative_count,f"{negative_count/len(reviews_df) *100: .2f}%")

with col3:
    neutral_count=sentiment_count.get('Neutral',0)
    st.metric('Neutral',neutral_count,f"{neutral_count /len(reviews_df) *100: .2f}%")

import plotly.express as px
fig=px.pie(values=sentiment_count.values,
           names=sentiment_count.index,
           title='Sentiment Distribution'
                )
st.plotly_chart(fig)







# Example usage
#st.title('Sentiment')
#st.write(classify_sentiment_openai(user_input))