import streamlit as st
import pandas as pd
from predict import predict
from vnstock import Vnstock
from vnstock.explorer.tcbs import Company
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_candlestick(df):
    try:
        fig = make_subplots(rows= 2, cols=1, shared_xaxes=True, vertical_spacing=0.3,
                            row_heights=[0.8, 0.2],
                            subplot_titles=(f"Candlestick Chart", "Volume Chart"))
        fig.add_trace(go.Candlestick(x=df['time'],
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='Candlestick'), row=1, col=1)
        if 'volume' in df.columns:
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], name='volume', marker_color='grey'), row=2, col=1)
        fig.update_layout(
            xaxis_rangeslider_visible=True,
            width=1200,
            height=1000,
            xaxis=dict(
                type="date",
                range=[df['time'].min(), df['time'].max()],
            )
        )

        st.plotly_chart(fig, use_container_width=False)

    except Exception as e:
        st.error(f"Error plotting chart: {e}")

SYMBOLS = ['ACB', 'BID', 'VCB', 'MBB']
st.set_page_config(page_title="Sentiment Analysis", page_icon=":guardsman:", layout="wide")
st.title("Sentiment Analysis")
st.write("This is a simple sentiment analysis application using machine learning.")
tabs = st.tabs(["Chart","News"])

with tabs[0]:
    st.header("Input data")
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        symbol = st.selectbox("Select stock symbol:", SYMBOLS, index=0)
        min = '2023-01-01'
    with col2:
        pass
    if st.button("Load data"):
        cols1, cols2 = st.columns([0.7, 0.3])
        with cols1: 
            st.write("### History data")
            stock = Vnstock().stock(symbol= symbol, source='VCI')
            cp = stock.quote.history(symbol=symbol, start= '2023-01-01', end= str(pd.Timestamp.now().date()), interval='1D')
            plot_candlestick(cp)
        with cols2:
            st.write("### Company news")
            compiled = Company(symbol=symbol)
            df = compiled.news()
            st.dataframe(df[['publish_date', 'title']])

with tabs[1]:
    cols1, cols2 = st.columns([0.5, 0.5])
    with cols1:
        text = st.text_input("Enter your text here:", key="input_text")
    with cols2:
        uploaded_file = st.file_uploader("Input file", type=['csv', 'xlsx', 'txt'])
    st.session_state.predict_text_button = st.button("Predict", key="predict_button")
    if st.session_state.predict_text_button:
        if uploaded_file is not None:
            st.success(f"Đã tải lên: {uploaded_file.name}")        
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=0)
                df['label'] = df['title'].apply(lambda x: predict(x))
                st.write("Predict label for news in input file:")
                st.write(df)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, header=0)
                df['label'] = df['title'].apply(lambda x: predict(x))
                st.write("Predict label for news in input file:")
                st.write(df)
            elif uploaded_file.name.endswith('.txt'):
                content = uploaded_file.read().decode('utf-8')
                predict = predict(content)
                st.text_area("Nội dung file", content, height=300)
                st.write("Predict label for news in input file:")
                st.write(predict)
        if text:
            sentiment = predict(text)
            st.write("Predicted Sentiment:")
            st.write(f"{text} - {sentiment}")