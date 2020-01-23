![img](https://i.ibb.co/fXTsPPr/powerpic.jpg)

------

## **Capstone project: The Power of Data**



The aim of this project is to explore different machine learning methods to predict UK hourly day-ahead electricity prices.

The dataset was created from data publicly available (i.e ENTSOE, Nordpool). We used historical data for the United Kingdom as well as the neighbouring countries, France and the Netherlands. It contains important features such as generation and demand historical forecasts, clearing prices and national holidays.

More variables were incorporated after performing EDA and features presenting no significant correlation with the target variable (i.e UK prices) were removed.

A selection of machine learning regression models was tested, optimised and ranked according to different scoring metrics.

Findings and overall performance of the best model will be discussed and analysed.

------

Table of contents:
<!-- MDTOC maxdepth:6 firsth1:1 numbering:0 flatten:0 bullets:1 updateOnSave:1 -->

   - [**Capstone project: The Power of Data**](#Capstone-project-The-Power-of-Data)   
      - [Problem Statement](#Problem-Statement)   
      - [Target Overview](#Target-Overview)   
      - [Accuracy Metric](#Accuracy-Metric)   
      - [Data](#Data)   
      - [Approach](#Approach)   
      - [Findings](#Findings)   
      - [Conclusion](#Conclusion)   

<!-- /MDTOC -->



------

### Problem Statement
The dynamics of the electricity prices have increased in complexity in the past decades. Characteristics influencing prices include continous balance between demand and supply, time dependence, weather conditions and the influence of neighbouring countries. These characteristics are difficult to predict and versatile, resulting with incertainty and highly volatile electricity prices.

![img](https://lh6.googleusercontent.com/a5WLemCyfziJwO9dOBJ6mb167PmSq8ovXnuFA4NR0CiRb59yLmVPf83krKtCUN-sGc7_UsEBKEd0uBWwPwglHgsjdlZgqiKlZUqnH4mYRfrzvZN7p5vnr0yfT3fPCcWr7gc3GFuOfw4)

The increased penetration of renewable energy in the recent years has worsened this behaviour, due to its dependence on weather conditions. Concerns around its effects on prices and grid stability have arisen, increasing the focus on  improving forecasting accuracy.  

While statistics methods are commonly used and generally show a good performance, they might be limited when it comes to non-linear and high frequency data i.e hourly prices with rapid variations.

Machine learning methods can address these limitations.


### Target Overview

The dependent variables of this project are the hourly day-ahead prices in the UK market.

The Market Clearing Price (MCP) is established in an auction conducted once a day.

It is set at the intersection between the **supply curve** (aggregated supply bids) and the **demand curve** (aggregated demand bids). Bids with negative prices are allowed.

The price of power consumed over a given period is determined by the **most expensive ‘marginal’ resource** in the mix. The marginal resource has historically been gas, creating a close correlation with gas.



![img](https://lh4.googleusercontent.com/QZv4VosGtpRYrbUFcaloOf8yyoI46wbMliBIOjwGeNT5QL5dWx6fKG0MCOXyeGTF7fp-xWUp0IPTdn1nUDCdFy_sUr28C7-rFV4PdMVdCRPd2HsrDY35pVeeabrBO8p7Xb1Z7uhVgp0)



### Accuracy Metric

In order to assess the models' accuracy, we will be scoring the models against the **R**esidual **M**ean **S**quared **E**rror (RMSE).

![img](https://i.ibb.co/jrJ2fJ8/Code-Cogs-Eqn.gif)


### Data

Free historical data from the European Network of Transmission System Operators (ENTSOE) and Nordpool was used to create the dataset. Four years of historical forecasts and clearing prices were downloaded (2016 to 2019).

Since the data had different currencies and granularities, the challenge was to convert EUR into GBP and transform all the data to an hourly granularity. The time zone difference was also taken into account.

It's important to highlight the fact that key features such as Gas and Coal data couldn't be obtained.

| Name             | Description                                               |
| ---------------- | --------------------------------------------------------- |
| date             | datetime yyyy-mm-dd hh:mm:ss                              |
| uk_time          | Hour UK time zone                                         |
| cet_time         | Hour CET                                                  |
| da_uk_price      | Day-ahead UK hourly prices (£/MWh)                        |
| da_fr_price      | Day-ahead French hourly prices (£/MWh)                    |
| da_nl_price      | Day-ahead Dutch hourly prices (£/MWh)                     |
| da_load_fr       | French demand day-ahead hourly forecasts                  |
| da_load_nl       | Dutch demand day-ahead hourly forecasts                   |
| da_load_uk       | UK demand day-ahead hourly forecasts                      |
| fr_generation    | French aggregated generation day-ahead hourly forecasts   |
| nl_generation    | Dutch aggregated generation day-ahead hourly forecasts    |
| uk_generation    | UK aggregated generation day-ahead hourly forecasts       |
| fr_solar         | French solar generation day-ahead hourly forecasts        |
| fr_wind_onshore  | French onshore wind generation day-ahead hourly forecasts |
| nl_solar         | Dutch solar generation day-ahead hourly forecasts         |
| nl_wind_offshore | Dutch offshore wind generation day-ahead hourly forecasts |
| nl_wind_onshore  | Dutch onshore wind generation day-ahead hourly forecasts  |
| uk_solar         | UK solar generation day-ahead hourly forecasts            |
| uk_wind_offshore | UK offshore wind generation day-ahead hourly forecasts    |
| uk_wind_onshore  | UK onshore wind generation day-ahead hourly forecasts     |
| uk_bh            | UK bank holiday (True if BH else False)                   |
| nl_bh            | NL bank holiday (True if BH else False)                   |
| fr_bh            | FR bank holiday (True if BH else False)                   |
|                  |                                                           |

### Approach

Several machine learning models were selected for this project according to their compatibility with the dataset.

High performance with:

* Multiple features

* High Frequency

* Seasonality

* Medium size dataset



**Random Forest**

A supervised learning algorithm using ensemble learning method for regression. A Random Forest uses multiple decision trees and a technique called bagging that involves training each decision tree on a different data sample where sampling is done with replacement. All predictions are then averaged. This technique provides a way to reduce overfitting and works well with strong and complex models.



**Extreme Gradient Boosting (XGBoost)**

Boosting is a different approach than bagging. It takes a weak base learner and tries to make it a stronger learner by retraining it on the misclassified samples.

XGBoost is a decision-tree-based ensemble algorithm that uses a gradient boosting framework.

It is one of the fastest implementations of gradient boosted trees. Instead of considering the potential loss for all possible splits to create a new branch, it looks at the distribution of features across all data points in a leaf. It then uses this information to reduce the search space of possible feature splits.



**Support Vector Regression**

Support Vector Regression (SVR) is a method using a Support Vector Machines approach for regression problems. It constructs a hyperplane in a multiple dimensional space that will help predict continuous values (wx + b = 0).

SVR performs linear regression in feature space (high dimensional) but unlike in least square regression, the error function is the ε-insensitive loss function.

The SVR  goal is to make sure the errors do not exceed the threshold. A margin of tolerance (epsilon) is set and errors situated within a certain distance of the true value will be ignored. The best fit line is the hyperplane containing the maximum number of points.

The slack variables (ξi) measure the cost of the errors and equal zero for all points that are inside the threshold.



![https://www.researchgate.net/figure/Schematic-of-the-one-dimensional-support-vector-regression-SVR-model-Only-the-points_fig5_320916953
](https://miro.medium.com/max/1438/1*rs0EfF8RPVpgA-EfgAq85g.jpeg)

https://www.researchgate.net/figure/Schematic-of-the-one-dimensional-support-vector-regression-SVR-model-Only-the-points_fig5_320916953



**Multilayer Perceptron**

A multilayer perceptron is one of the most common neural network architectures.

In brief, Neural networks attempt to iteratively train a set of weights that, when used together, return the most accurate predictions for a set of inputs. The model is trained using a loss function, which our model will attempt to minimise over iterations.

We start with an input layer of features that are passed into neurons in the hidden layers. Each neuron is a perceptron, like a bunch of small linear regressions. The information is passed from one layer to the next until it hits the output layer. The output layer does one calculation to output a prediction for the outcome.

By increasing the volume of data in the network, it adjusts the weights based on the output of the loss function, until we reach a highly trained model and specific weights (backpropagation).



![image-20200121130705378](https://i.ibb.co/3rN6HgS/neuralnet.png)


### Findings

Preliminary findings from the Explanatory Data Analysis highlighted insignificant correlation between the UK prices and fundamental variables such as supply and demand. The correlation with the prices in the neighbouring countries wasn't particularly strong.

![image-20200121140646449](https://i.ibb.co/ySYBgWH/correlation.png)

Since there is a lot of seasonality within the traget variable, we looked at the autocorrelation with different lags. The highest scores were for past prices at a similar hour.

![image-20200121141144642](https://i.ibb.co/gJhF320/autocorrelation.png)

The relationship bewteen UK prices and the neighbours was investigated. The correlations between the UK prices at T and the French/Dutch prices at different lags were assessed.

Lags with the highest correlation scores were added as additional features.

Features selection was performed and all the above models were optimised.

The below graph shows the most important features, obtained by using a Recursive Feature Elimination (RFE) with a Random Forest estimator.

![image-20200121153941449](https://i.ibb.co/dkYRwh2/feature-importance.png)

Out of all the models selected, the Random Forest model performed best. It has the highest RMSE and R2 scores and a decent cross validation score.

![image-20200121154613300](https://i.ibb.co/vJ1rHM4/newplot.png)

The model is also more accurate than the baseline - hourly averaged price.



![image-20200121155833858](https://i.ibb.co/Jzf0Mh5/predictions.png)



### Conclusion

Optimising the model helped minimising the RMSE scores. However, a RMSE of 5.69 remains quite significant given the scale of the target variable with a mean of 45.86 £/MWh.

High spikes failed to be accurately captured, which was part of our goal.

There are limitations to consider regarding the data. Only free data was used and the quality cannot be fully trusted. ENSTOE gathers data submitted by all market participants, but the validity of this data can only be verified to some extend. Updates are not performed when a deadline has been missed or when there has been a revision, resulting in missing or incorrect data.  Moreover, important features were missing as the data couldn't be obtained for free.

The above considerations must be taken into account in the judgment of the model's performance. Using more reliable data and adding variables such as gas and coal prices would probably improve the accuracy of the model.

Other machine learning techniques can also be explored.
