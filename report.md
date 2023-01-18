# Comparison studies on predicting time series using Traditional Statistical Methods and LSTM



### Abstract

With the rapid development of artificial intelligence, recent years have seen various machine learning methods applied to time series modeling and forecast. LSTM is just one of those machine learning methods. We wonder whether LSTM performs better than out traditional statsictical methods like ARIMA on different datasets. So we apply the two methods to three datasets with different characteristics to make a comparison.

*Keywords:* Time series forecasting, LSTM, ARMA-GARCH model.

### 1.Introduction

​	Developing efficient models to predict time series is undoubtedly a significant part of time-series analysis. While there are many traditional approaches to build models for time series and do predictions based on the specific model, ARIMA, GARCH, to name but a few, recent years have seen professionals turn to Main Learning and Deep Learning, trying to make even more accurate predictions. LSTM (Long Short-Term Memory) is one of the machine-learning methods that can be applied to fit a given time-series and make predictions. LSTM has gained great popularity for this model considers both short-term and long-run implication. On top of that, artificial neural nnetworks are good nonlinear function approximators, so they provide a natural approach to deal with time series that are suspected to have nonlinear dependence on inputs. Although LSTM-network enjoys the advantage of a more flexible estimating process, whether LSTM outperforms traditional methods when it comes to modeling time-series with different traits is a fascinating question.

​	When we use traditional statistsical methods to analysis time-series, chances are that we first plot the series, figuring out some simple characteristics such as seasonality or evident rising trend, then we use specific methods to capture those traits, turning them into a mathematic form. But when applying machine-learning method, we don't have to pay too much attention to those features due to the model's flexibility. 

​	In this article, we try to use both traditional models like ARIMA-GARCH and LSTM to fit real-world data with different traits. To be specific, we use three datasets. The first one is the common stuied "flights" contained in Python seaborn library, which serves as a representative of time-series with both rising trend in the long-term and seasonality. The second dataset shows the daily hit of the USTC-secondary-class platform and we will show in section 4 that this time series looks quite gentle without evident trend. The third dataset contains the GDP and population data of China in the past decades and we can see a steep rising trend without seasonality from the plot. The above three datasets cover the most common time-series in our daily life and thus studying them is of great importance.

​	The rest of the article is organized as follows. Section 2 introduces the LSTM method and the applied algorithm. We also present  the GARCH model in this section. In Section 3 we employ both the traditional models like ARMA-GARCH and LSTM to three different datasets and analysis the results in detail. At last, we give a brief discussion in Section 4 to end this article. 

​	

### 2.Methodology

#### 2.1 Long Short Term Memory Network (LSTM)

​	Imagine that you are reading an essay, it's human nature that we understand each word based on our understanding of previous words in the sentence, which means our thoughts have persistence. However, traditional neural networks can't do this. To address this issue, RNN (recurrent neural network) was proposed. This kind of networks has loops, allowing information to persist. In cases where the gap between the relevant information and the place that it's needed is small, (for example, when we try to predict the last word in a single sentence with complete meaning, it's likely that we don't need any further context), traditional RNN can give satisfied results. Unfortunately, when such gap grows and log-term information should  be taken into consideration, RNNs become unable to connect the information. Thus why LSTM was proposed.

​	LSTM is a special kind of RNN, capable of learning both long-term and short-term dependencies. It's such valuable characteristic that make LSTM stands out and becomes a popular way to predict time-series. LSTM has the form of a chain of repeating modules if neural network and each single module is composed of four neural network layer interacting in a special. Figure 1 below shows the basic structure of LSTM.

![Figure1: the basic structure of LSTM ](./1.png) 

​	The key to LSTM is the **cell state**, which is the horizontal line running through the top of the diagram, representing long-term memory. It runs straight down the entire chain with some linear interactions. And the line at the bottom of the diagram is the **hidden state** representing short-term memories.

​	The first step in a LSTM is to decide what information we're going th throw away from the cell state. And this decision is made by a sigmoid layer called the "forget gate layer". It use $h_{t-1}$ (the short-term memory at time $t-1$) and $x_t$ (the newly-input information at time t) to output a number between 0-1, determining what percentage of the Long-Term Memory is remembered. Figure 2(a) shows this procedure.The next steps (shown in Figure 2(b) and 2(c) ) is to create a potential long-term memory based on the short-term memory at time $t-1$ and the input information. The next stage (shown in Figure 2(d)) updates the short-term memory using the newly-updated long-term memory as input.

<img src="./2a.png" style="zoom:50%;" />

<img src="./2b.png" style="zoom:50%;" />

<img src="./2c.png" style="zoom:50%;" />

<img src="./2d.png" style="zoom:50%;" />

> The functions in the figure are: $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$,$i_t=\sigma(W_i[h_{t-1,x_t}+b_i])$,$\tilde C_t=tanj(W_C[h_{t-1},x_t]+b_C)$, $C_t=f_t*C_{t-1}+i_t*\tilde C_t$, $o_t=\sigma(W_o[h_{t-1},x_t]+b_o),h_t=o_t*tanh(C_t)$

​	Use time series information as $x_t$, we can derive the fitted model.



#### 2.2 Seasonal Auto Regressive Integrated Moving Average (SARIMA)

We commonly see that financial data has evident cycle, especially those monthly or quarterly data. In a time series, if we find the data appears assimilation after $S$ time-unit, for example, they are both at the peak or trough, we say that the specific time series has periodic characteristics with $S$ as the cycle. And the model that depicts this kind of time series is called seasonal ARIMA model (SARIMA). 

Assume that a time series with seasonality $\{X_t\}$ becomes a stationary series after $D-order$ seasonal difference, and the new series is $W_t=(1-B^S)^DX_t$. Further more, we assume that $\{W_t\}$ follow $ARMA(P,Q)$. As a result, we get   
$$
U(B^S)(1-B^S)^DX_t=V(B^S)\epsilon_t\\
where\ U(B^S)=1-\Alpha_1B^S-\Alpha_2B^{2S}-\dots-\Alpha_PB^{PS}\\
			V(B^S)=1+H_1B^s+H_2B^{2s}+\dots+H_QB^{QS}
$$
In the above mentioned model, we assume that $u_t=\frac{U(B^S)(1-B^S)^D}{V(B^S)}X_t$ is a white-noise series, which is not necessarily the case. Since seasonal difference only deals with seasonality and the ARMA model based on $W_t$ only considers the correlation between the same period points in different periods, short-term influence may be ignored. To solve this problem, we instead assume that $u_t$ follows $ARIMA(p,d,q)$, and we will get
$$
\Phi(B)U(B^S)\nabla^d\nabla_S^DX_t=\Theta(B)V(B^S)\epsilon_t
$$
and we use $ARIMA(p,d,q)\times(P,D,Q)_S$ to denote this. For example, the $ARIMA(0,1,1)\times(0,1,1)_{12}$ represents the following model:
$$
(1-B)(1-B^{12})X_t=(1+\theta_1B)(1+\theta_{12}B^{12})\epsilon_t
$$
In a nutshell, the SARIMA model is constructed by the following three steps:

- Use ARMA(p,q) to extract short-term correlation
- Use ARMA(P,Q) with cycle $S$ to represent the seasonality
- Assume the multiplicative seasonal model with $\nabla^d\nabla_S^DX_t=\frac{\Theta(B)}{\Phi(B)}\frac{\Theta_s(B)}{\Phi_s(B)}\epsilon_t$

#### 2.3 Generalized Autoregressive Conditional Heteroscedastic Model (GARCH)

In the ARIMA model, we assume that the innovation sequence $\epsilon_t$ is independently and identically distributed with the same variance. However, it has been found that stock prices and other financial variables have the tendency to move between high volatility and low volatility. And volatility is an important measure of risk. Then the GARCH model was proposed and has become an essential tool of modern asset pricing theory and practice. 

Let $r_t$ denotes a log-return sequence, and $a_t=r_t-\mu_t=r_t-E(r_t|F_{t-1})$ represents the innovation sequence. We say that $\{a_t\}$ follows the $GARCH(m,s)$ model, if $\{a_t\}$ satisfies 
$$
a_t=\sigma_t\epsilon_t\\
\sigma_t^2=\alpha_0+\Sigma_{i=1}^m\alpha_ia_{t-i}^2+\Sigma_{j=1}^S\beta_j\sigma_{t-j}^2
$$
where $\{\epsilon_t \}$ is the independently and identically distributed white-noise sequence with zero-mean and unit variance, and $\alpha_0>0,\alpha_i\geq0,\beta_j\geq0,0<\Sigma_{i=1}^m\alpha_i+\Sigma_{j=1}^s\beta_j<1$

### 3. Application To Real-World Data

In this section, we apply the LSTM and traditional statistical methods like ARIMA to three datasets with different traits, trying to make comparisons between the two methods.

#### 3.1 Flight (Time Series With Seasonality and Rising Trend)

First, we employ LSTM and traditional methods to the *Flight* dataset, which is a commonly used benchmark datasets from *Python seaborn library*, containing monthly information about the number of passengers taking airplanes. The data consists 144 pieces of  data, starting from January 1949. To compare the predictiong results of the two methods, we chose the first 132 ones as training set, leaving the last 12 months as the test set, and we use the mean square error $mse=\sum (y_{predict}-y_{real})^2$ as the judging criterion.

To give an intuitive impression of the data, we first plot the time series. It's distinctive in Figure 3 that this series contains seasonality and a rising trend as the same time. Taking the rising trend into consideration, in the traditional methods, we first differential the sequence and draw the auto-correlation function plot and partial auto-correlation plot. And the two plots show that the series has evident seasonality, with distinct 12-order auto-correlation, which is consistent with our observation. After this, we find that the sequence after simple first-order difference and  seasonal difference successfully passes the ADF test, showing the newly-get sequence is stationary. Unfortunately, results from white-noise test indicate that we don't have enough evidence to reject the null hypothesis with the p-value around 0.05. After tentative modeling the sequence with ARIMA, we get a unsatisfing results. Reflecting on the whole process, we realize that the original sequence has a rapid rising trend with a high difference between the begin and end of the series. So we instead do the logarithmic and differential treatment to the traning series, and follows the above mentioned process. In the end, we find the residuals succeed to pass the white-noise test and the new model is much better with a much lower AIC and BIC. After getting a satisfing model, we predict the number of passengers 12 months ahead and gained the $mse$.

While using LSTM to predict the number of passengers of the following year, we first reconstruct the original dataset to make it suitable for our LSTM training. We build a new dataframe with 13 columns, with the first 12 elements representing the short-term memory for the 13th one in each row. Then we feed the dataframe into the LSTM model using $L_2$ loss as the minmization rule. After some parameter-tunning work, we gain the prediction of the last 12 months.

Figure 4(a) and Figure 4(b) show the predicting results from the two methods respectively. From simple observation, it seems that traditioanl methods outperforms the LSTM. To gain a more accurate conclusion, we compute the $mse$ as mentioned before to make further comparison and the results is shown in Table1. From Table 1, it's intuitively that the SARIMA model performs much better than LSTM with a much smaller $mse$.

One reason that might account for the results is that the Flight sequence enjoys some good traits that can be easily expressed in mathematical language and we have corresponding models to deal with this issue. For example, we find SARIMA a great model to depict the seasonality. On the other hand, though LSTM enjoys popularity for its flexibility to a variety of time series, this nonparametric approach may fail to give accurate results when there does in fact exists a math model.

<img src="./3.png" style="zoom:50%;" />

<img src="./4a.png" style="zoom: 50%;" /><img src="./4b.png" style="zoom:50%;" />

| SARIMA | 4.15e3 |
| ------ | ------ |
| LSTM   | 4.18e4 |

#### 3.2 Second Class Access Data of USTC (Time Series Without Evident Fluctuation)

The USTC Second Class program is a significant part of the students' campus life and every students get information about various activities through the Second Class platform. From another perspective, for event organizers, it's of great importance to show their programs to as many as possible students so that more university man can take part in the program. In a nutshell, it's worthwhile to analysis the daily visits of the platform and give predictions about future visits for fear that organizers can publish their projects on days with potentially high hits. Hopefully, in this way, students are more likely to notice those fascinating activities and planners can draw more attention.

To achieve this, a member of our group emailed the person in charge of the second class platform, aiming at getting daily visits to make further modeling and predictions. Many thanks for giving us such valuable data! After getting in contact with the director. At first, the whole data we got was too huge to make further analysis. Then after communicating many times with a student from the student union, two columns that are of our interest - date and vitits were extracted, making the modeling and prediction possible. Unfortunately, after simple processing, we found that the data itself has abnormal faults due to recording and handling errors by technicians from the platform. As a result, we only chose undergraduate students' visting data in October and November. The chosen data consists 61 pieces of information and we select the last 7 days as test data and the remainning days as training data. And we use the same criterion as that mentioned in Section 3.1 to do the comparison.

We first plot the data as shown in Figure 5. It seems that this series is relatively gentle without evident seasonality or trend. So while using traditional methods, we first do ADF test to check whether the sequence is truly a stationary sequence without unit-root. It turned out that the p-value is 0.7, implying that the original series is not stationary. So we differential the sequence, finding that the new series passes the stationary test but is not a white-noise sequence. Then we attempt to use ARIMA model to  fit this sequence and the residuals are truly white-noise sequence.

Then we use LSTM to fit the same sequence. This time we chose weekly information as the short-term memory, then follows the same steps as mentioned in detail before.

After the model-fitting process, we predict the last 7 days and compute corresponing $mse$, the predicting results and $mse$ are shown in Figure 6 and Table2. We find at when predicting the second-class data, LSTM has better behavior than ARIMA model according to the $mse$. And we notice that, since in the application process, we use MA model to deal with the sequence after first-order difference, and use the mean value of prediction model as our final result, the predictions keep time for the last 7 days, which may explain why the ARIMA model fail to give a satisfying result. 

In this study, the results show that there may be a lack of appropriate mathmatic model to fit the sequence, leading to the relative poor performance. On the contratry, since LSTM is a nonparametric method without any model assumptions, it may work better under circumstances when it's hard to find suitable model.



<img src="./5.png" style="zoom: 50%;" /> 

<img src="./6a.png" style="zoom:50%;" />

<img src="./6b.png" style="zoom:50%;" />

| ARIMA | 4.53e5 |
| ----- | ------ |
| LSTM  | 2.46e5 |

#### 3.3 Chinese GDP Data (Time Series With Distinct Rising Trend)

The previous two real-world applications show the comparison results of time series with seasonality and series that seems stationary respectively. In this subsection, we apply the two methods to time series with rising trend but no seasonality, which can be observed immediately from the plot in Figure 7. The dataset used is from *Maddison Project Database 2018*, and we only use the Chinese GDP data from 1950 to 2018. We totally have 69 pieces of data and we choose the last 5 years as the test data.

Considering the significant growth trend, while using traditional method, we differential the logarithmic sequence at the first step, checking the stationarity of the sequence after transformation. Then we use the ARIMA model to fit the sequence and find that the residuals come from white-noise series. Since GDP data is one type of finance data, where volatility analysis is of great significance, we also check whether the white-noise residuals has the *GARCH* effect and it turns out that the series does has *ARCH* effect. So we use *ARMA-GARCH(1,1)* model to refit the logarithmic difference sequence. In the end, we forecast the last 5 years' GDP based on the model we have built.

Since LSTM have fixed input format and operation framework, we just repeat our work again, choosing the past 5 years as short memory and doing parameter-tunning. 

The forecasting results from the two approaches are represented in Figure 8(a), Figure 8(b) and Table3. In this application, we find that LSTM holds up better than the ARIMA-GARCH model concluding from both the graphs and their $mse$. We think this might be a result of improper modeling using the traditional methods. On top of that it's surprising that LSTM does a quite good job in this case, showing the great power of machine learning.

<img src="./7.png" style="zoom:50%;" />

| ARIMA-GARCH | 7.83e8 |
| ----------- | ------ |
| LSTM        | 8.90e4 |

#### 3.4 Summary of Applications

In this subsection, we summarize our experimental results and analysis conclusions. To make it clear, a comparison of *mses* using the two methods on the three dataset are shown in Table 4.

| Dataset      | LSTM   | Traditional Methods |
| ------------ | ------ | ------------------- |
| Flight       | 4.18e4 | 4.15e3              |
| Second-Class | 2.46e5 | 4.53e5              |
| Chinese GDP  | 8.90e4 | 7.83e8              |

 On the whole, we conclude that when the time series satisfies the model assumptions of a given mathmatic model, the traditional methods perform much better than LSTM for the simple reason that LSTM only learn from data itself without considering the underlying model. Unfortunately, chances are that real-world data fail to meet the model assumptions. Or in another words, it's hard for us to find the mathematic model perfectly fits the series. Under such circumstances, LSTM may stand out, giving great outputs. Besides, the application process give us a deeper understanding of the modeling procedure and just like what we have learned in class, there's no omnipotent model or method and we still have a long way to go to get more accurate forecasting results.

### 4. Conclusion and Remarks

To conclude this article, we first discuss here several possible topics for future study. First, we have only applied the two methods to modeling and forecast time series. But it's noticeable that volatility modeling is a fascinating and meaningful topic considering the fact that finacial data like the daily-return of stocks has changing volatility. To the best of our knowledge, there is not any great method that utilize LSTM to do the volatility modeling task. Existing methods relies on estimation of volatility first, which is actually a harsh issue itself. Second, we only consider univariate time series. But there are many great machine-learning methods like LSTNet and MGTNN that considers temporal and spatial correlation among multivariate time series. We may compare those  methods with traditional methods like VAR in the future. Last but not least, concluding from our applications, we find that both traditioanl methods and LSTM are far from perfect. It is likely that we explore ways to combine the two methods take advantage of their merits to gain better results in the future.

At last, all our group members have contributed a lot to this project considering the harsh process attaining data and writing codes to realize those applications.