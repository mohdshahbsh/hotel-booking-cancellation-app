import pandas as pd
import numpy as np
from joblib import load, dump
import streamlit as st
from streamlit import session_state as ss
from sklearn.preprocessing import StandardScaler

def main():
    # main page
    st.title('Hotel Booking App')
    welcome = ''' 
              <font color='#BDD5EA' size=5>Welcome, 
              <p> This app was designed to help hoteliers to assess the risk of booking cancellations using 
              machine learning algorithms. Please input the necessary information on the sidebar to the left and press
              the <font color='red'>Run</font> button to generate the bookings cancellation probability.</font>
              <br>
              <br>
              <center><font color='#FF3366' size=12> Probability of Cancellation </font></center>
              '''
    st.markdown(welcome, unsafe_allow_html=True)
    # sidebar
    st.sidebar.header('Enter Booking Information')
    st.sidebar.number_input('Booking Lead Time', min_value=0, key='lead_time')
    deposit_type = {'No Deposit': 0,  # deposit type encode
                    'Refundable': 1,
                    'Non Refund': 3}
    st.sidebar.selectbox('Deposit Type', options=deposit_type.keys(), key='deposit_type')
    st.sidebar.number_input('Average Daily Rate (ADR)', key='adr')
    st.sidebar.number_input('Number of Special Request',
                            min_value=0,
                            max_value=5,
                            step=1,
                            key='total_of_special_requests')
    st.sidebar.number_input('Previous Cancellations', min_value=0, step=1, key='previous_cancellations')
    market_segment = {'Direct': 0,  # market segment encode
                      'Corporate': 1,
                      'Online TA': 2,
                      'Offline TA/TO': 3,
                      'Complementary': 4,
                      'Groups': 5,
                      'Undefined': 6,
                      'Aviation': 7
                      }
    st.sidebar.selectbox('Customer Market Segment', options=market_segment.keys(), key='market_segment')
    st.sidebar.number_input('Agent Code Number', min_value=0, step=1, key='agent')
    customer_type = {'Transient': 0,
                     'Contract': 1,
                     'Transient-Party': 2,
                     'Group': 3
                     }
    st.sidebar.selectbox('Customer Type', options=customer_type.keys(), key='customer_type')
    # calculate probability
    st.sidebar.button('Run', key='run')
    if ss.run:
        # create array of input data
        input_data = np.array([
            deposit_type[ss.deposit_type],
            ss.lead_time,
            ss.adr,
            ss.total_of_special_requests,
            ss.previous_cancellations,
            market_segment[ss.market_segment],
            ss.agent,
            customer_type[ss.customer_type]
        ]).reshape(1,8)
        # pre-process input data
        data = pd.read_csv('cleaned_features.csv', index_col=0)
        scaler = StandardScaler()
        scaler.fit(data)
        input_norm = scaler.transform(input_data)
        # load model
        model = load('hotel_app_logistic_regression_model.joblib')
        # make prediction
        result = model.predict_proba(input_norm)
        proba = np.round(result[0][1]*100,2)
        # display prediction
        if proba < 50:
            color = '#7ABD91'
        else:
            color = '#FF6962'
        st.markdown(f'''
        <center><font color={color} size=12> {proba}% </font></center>
        ''', unsafe_allow_html=True)
        if proba < 50:
            st.balloons()
        else:
            st.snow()


if __name__ == '__main__':
    main()
