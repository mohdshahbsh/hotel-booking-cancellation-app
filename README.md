## Impact of COVID

[APP DEMO](https://mohdshahbsh-hotel-booking-cancellation-app-hotel-app-ym27gr.streamlitapp.com/)

In the last 2 years, the tourism industry was dealt a heavy economic blow with the onset of COVID-19. In the span of 3 months, the virus was able to bring the global hotel occupancy rate down to 13.3% since May 2020, which is an 82.3% drop from the same time the previous year.

<img src="https://i.imgur.com/Yw0f4DJ.png" width="600px" alt="https://www.statista.com/statistics/206825/hotel-occupancy-rate-by-region/">

However, with global vaccination rates inching closer to 70%, most governments around the world has loosen COVID-19 restrictions in an effort to boost the country’s economic recovery. As tourist begin to flock into the various countries around the world, the hotel industry is once again inundated with online bookings from all over the world.

<img src="https://i.imgur.com/SuZsq2b.jpg" width="600px" alt="https://www.nytimes.com/interactive/2021/world/covid-vaccinations-tracker.html" >

Most of them are done through online travel agencies (OTA) which makes up over 90% of all the bookings that a hotel receives. However, a study in 2019 by [Esther Hertzfield](https://www.hotelmanagement.net/tech/study-cancelation-rate-at-40-as-otas-push-free-change-policy) reported that on average 40% of these bookings are cancelled.

Prior to COVID-19, the impact of these cancellations were damaging, and they are usually brought on by aggressive marketing strategies from competing hoteliers who wish to poach customers away from their prior bookings by offering cheaper prices through online marketing. Given that most online travel agencies (OTA) offer free cancellations to incentivise sales, customers can make cancellations for any prior bookings at any time with virtually no consequences. The impact of risk free booking are; 

1. It makes it much more difficult for hotels to create a demand forecast  
2. A cancelled booking is ultimately lost revenue

Using machine learning algorithms hoteliers can minimize the loss of revenue by being able to predict them beforehand. Using the information from an online booking as the input variable, the hotelier will be able to assess the probability of a booking being cancelled. Such insight would allow the hotelier to better allocate their marketing resources towards these high risk bookings. 

Incentives such as providing discounted rates, cashback, or extra amenities can be targeted towards those customers to lower the cancellation rate. In addition, hoteliers will also be able to conduct a much more accurate demand forecasting by taking the high risk bookings into account.

The dataset used for this project was obtained from a real hotel located in Portugal. It was made available by [Antonio et al.](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) as a way to provide more real world data of hotels to improve research and education in the industry.

The following 8 attributes were chosen to be the independent variables for the machine learning model. They were chosen based on their feature importance score and the generalized nature of the attribute.

<img src="https://i.imgur.com/EsDvxD6.png" width="600px" caption="Feature importance of hotel booking dataset attributes" >
