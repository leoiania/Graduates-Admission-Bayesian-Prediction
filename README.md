# Graduates Admission Prediction following a Bayesian approach

## Introduction

This report summarizes the statistical modeling and analysis results we obtained performing a Bayesian analysis on a [dataset](https://www.kaggle.com/datasets/mukeshmanral/graduates-admission-prediction) collecting the main parameters which are considered during application for Masters Programs.

The goal of our analysis has been to fit a model capable to predict the probability of admission for students following a Bayesian approach, and to understand what are the parameters that affect the most the probability of being accepted.

## About the Data

Our Dataset contains information about 500 indian students, for each of them we have seven important parameters usually considered during application for the Master Program and a response variable varying between {0,1}, that describes the chance of admission. Data are collected for 500 students.

To have a better idea, let's take a look at the first 7 row of the dataset:

![head_data](https://user-images.githubusercontent.com/78596444/192009988-a068574c-9bc7-4c65-bb79-1575405212be.png)

Let’s explore our data:

### GRE and TOEFL scores

GRE is a widely used test in U.S.A (but also in other countries) for admission in Master’s programs. It measure verbal reasoning, quantitative reasoning, analytical writing, and critical thinking, by testing algebra, geometry, arithmetic, and vocabulary knowledge.

In general, the minimum and the maximum points that a student can obtain are, respectively, 260 and 340.

The TOEFL, instead, measures the English language ability of non-native speakers. Its range goes from 0 to 120.

![tests](https://user-images.githubusercontent.com/78596444/192010254-02f10375-ac3c-41fa-bd6f-611a004ae385.png)

### University Rating

This is a categorical variable which indicates the rank of the university the student comes from. It will be split in dummy variables when implementing the model.

![uni_rank](https://user-images.githubusercontent.com/78596444/192010451-347ef690-eff1-4c5d-b1fe-987af30e679f.png)

### Statement of Purpose and Letter of Recommendation

To evaluate SOP and LOR of each student, they are given a rating from 1 to 5 in steps of 0.5, which indicates the “strength”.

![sop_lor](https://user-images.githubusercontent.com/78596444/192010568-78a20adf-3ce3-453e-a354-0136cdc0003d.png)

### CGPA

The CGPA indicates the Cumulative Grade Point Average of each student at university at the moment of the application for Master’s program.

![cgpa](https://user-images.githubusercontent.com/78596444/192010641-1611ca00-59cb-464c-9f47-a23c0aa43f1b.png)

### Research

A simple dummy variable which indicates whether the student already did research or not.


### Our Approach

We decided to implement 2 different bayesian models:

1. Logistic regression model
2. Beta regression model


## Logistic regression

We assumed our response variable Y as binary, $Y \in {0,1}$  by transforming the Chance of admission:

$$ y_i = \begin{cases} 1, & \mbox{if Chance of admit > 0.65} \\ 
0, & \mbox{if Chance of admit  }\leq{ 0.65} \end{cases}$$

So we can now assume:

$$ Y_i|\pi_i \stackrel{ind.}{\sim} Ber(\pi_i) \quad i=1,....,n $$

Now the aim is to model the effect of predictors on $\pi_i$, but we can't do it directly because then we would have a problem with negative and above 1 probabilities. So we will model $\eta = g(p)$, where $g(.)$ is a **link function**, such that $0 \leq g^{-1}(\eta) \le 1$.

$$\eta = g(p) = \beta_0 + \beta_1x_1+...+\beta_qx_q$$

In our case we decided to use a logit link function 
$$logit(\pi_i) = log(\frac{\pi_i}{1-\pi_i}) = \beta^Tx_i$$

and so

$$\pi_{i} = g^{-1}\left(\boldsymbol{\beta}^{\top} \boldsymbol{x}_{i}\right)=\frac{e^{\boldsymbol{\beta}^{\top} \boldsymbol{x}_{i}}}{1+e^{\boldsymbol{\beta}^{\top} \boldsymbol{x}_{i}}} $$

Under $Y_i|\pi_i \stackrel{ind.}{\sim} Ber(\pi_i) \quad i=1,....,n$, the **Likelihood** is

\begin{aligned}
p(\boldsymbol{y} \mid \boldsymbol{\beta}) &=\prod_{i=1}^{n} p\left(y_{i} \mid \pi_{i}\right) \\
&=\prod_{i=1}^{n} \pi_{i}^{y_{i}}\left(1-\pi_{i}\right)^{1-y_{i}} \\
&=\prod_{i=1}^{n} h\left(\boldsymbol{\beta}^{\top} \boldsymbol{x}_{i}\right)^{y_{i}}\left(1-h\left(\boldsymbol{\beta}^{\top} \boldsymbol{x}_{i}\right)\right)^{1-y_{i}}
\end{aligned}

And the **priors** on $\boldsymbol{\beta} = (\beta_1,.....,\beta_p)^T$ are:
$$\beta_j \stackrel{ind.}{\sim} N(\beta_{0j}, \sigma^2_{0j}) $$

Since the prior is not conjugate neither semiconjugate, to approximate the posterior 
$p(\boldsymbol{\beta} \mid \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \boldsymbol{\beta}) p(\boldsymbol{\beta})$ we need a Metropolis Hastings algorithm. 

To implement the Metropolis Hastings algorithm we use **JAGS**, a program for analysis of Bayesian  models using Markov Chain Monte Carlo (MCMC) simulation, that 
easily allows us to implement Gibbs sampler and Metropolis Hastings algorithms.

### MCMC diagnostic 

MCMC is a numerical technique and hence subject to *approximation* error. For this reason before using the output that we obtained for posterior inference on $\boldsymbol{\beta}$, we need to be "sure" that the resulting chain provide an appropriate approximation of the true posteriors distributions.

#### Trace plot 

The trace  plots provide a graphical representation of the Markov chain for $\theta_j$ for s = 1,....,S.

The chain should be concentrated within a region of high posterior probability centered around the mode of $p(\theta|\boldsymbol{y})$

![traceplot_logi_1](https://user-images.githubusercontent.com/78596444/192013373-0c576ab2-2da2-4b06-8af4-9a7bf898446c.png)

![traceplot_logi_2](https://user-images.githubusercontent.com/78596444/192013429-19d1f7fb-9ca5-4bc7-b2f3-6dbd4cfcc1b3.png)


#### Autocorrelation

![acf_logistic_1](https://user-images.githubusercontent.com/78596444/192013615-14c09c09-2a3a-4450-9240-eb429af4491b.png)

![acf_logistic_1](https://user-images.githubusercontent.com/78596444/192013642-212d722c-187c-49d2-8ab2-4d5cc531bfc9.png)

In trace plots and autocorrelation plot we see no problems. In the first case we want the chain to converge to the mode, while in the latter we would avoid to have high value of autocorrelation inside the chain and this is, clearly, the case.

#### ### Geweke test

The idea behind the Geweke test is: *if the chain has reached convergence then statistics computed from different portions of the chian should be similar*

Considering two portions of the chain:

* $\boldsymbol\theta_I$ : the initial 10% of the chain (with size $n_I$)
* $\boldsymbol\theta_L$ : the last 50% of the chain (with size $n_L$)

The Geweke statistics is 

<p align = "center">
$Z_{n}=\frac{\bar{\theta}_{I}-\bar{\theta}_{L}}{\sqrt{\hat{s}_{I}^{2}+\hat{s}_{L}^{2}}} \rightarrow \mathcal{N}(0,1) \quad \text { as } \quad n \rightarrow \infty$
</p>
Where *n* is the sum of the size of the 2 portion that we selected for the test and $\hat{s}_{I}^{2}$ and $\hat{s}_{I}^{2}$ are their sample variance.
 
Large absolute values of the Geweke Statistics lead us to reject the hypothesis of stationarity.

| $\beta_1$  | $\beta_2$  | $\beta_3$  | $\beta_4$  | $\beta_5$  | $\beta_6$  | $\beta_7$  | $\beta_8$  | $\beta_9$  | $\beta_{10}$  | $\beta_{11}$  |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|-0.23|	-0.32 |	0.72 |	1.7 |	1.4	| 1.07 |	1.26 |	-0.5 |	0.6 |	-0.37 |	-0.56 |


As we can see from the output, in our case the test does not suggest problems in our chain.

#### 
### Effective Sample Size(ESS)

Rather than a test, is a value that quantifies the reduction in the effective number of draws, due to the correlation in the chain.

$$ ESS = \frac{G}{1+2s\sum_{g=1} acf_g}$$

| $\beta_1$  | $\beta_2$  | $\beta_3$  | $\beta_4$  | $\beta_5$  | $\beta_6$  | $\beta_7$  | $\beta_8$  | $\beta_9$  | $\beta_{10}$  | $\beta_{11}$  |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|5000|	5000 |	50000 |	5000 |	5000	| 5000 |	5000 |	5310.3 |	5000 |	5000 |	5238.9 |

Both graphical and formal tests for MCMC convergence lead us to conclude that the Markov chain that we generated provide a good aproximation of the true posterior distribution of the beta regressors.

We can then state that each column of the output matrix will then approximate the posterior distribution of the relative $β_j$ and we can see how their values are distributed with a boxplot:

![image](https://user-images.githubusercontent.com/78596444/192014904-14e09fad-522d-49f4-8686-5eb452c5db85.png)

### Predictions

Now we want to use our model on "new" observations, and see if it is able to predict whether a student will be admitted or not.

As new observation we consider the test data frame, that we created at the beginning by splitting the original dataset. It contains 125 new observations: $\boldsymbol{x}^*_j =(x_{j,1}^*),....,(x_{j,11}^*)$ for $j = 1,...,125$


With the posterior distribution of $\boldsymbol{\beta} = (\boldsymbol\beta_1,.....,\boldsymbol\beta_p)^T$, that we approximated previously, what we are going to do is: 

For i = 1,...S:

*    For j = 1,....,J:
     1. Compute $\eta ^{(s)} = \boldsymbol\beta^{(s)T}\boldsymbol{x}^*$
     2. Compute $\pi ^{(s)} = g^{-1}(\boldsymbol\beta^{(s)T}\boldsymbol{x}^*)$


After obtaining 5000 values for $\pi_i$, we can sample from a bernoulli which has it as parameter and obtain the “fitted” values of $y_i$ as the **mode** of the distribution of each future observation.

We can compare what we’ve obtained with the observed value from test data:

![conf_matrix](https://user-images.githubusercontent.com/78596444/192017098-1b4ab31c-ac84-426a-99b0-d3051c7c9f27.png)

Watching at the confusion matrix we can see that we obtain:
 
* Overall accuracy $\approx81\%$
* Sensitivity: $\approx84\%$
* Specificity: $\approx76\%$


### Spike & Slab Variable Selection
To perform variable selection we introduce a (p,1) binary vector $\boldsymbol\gamma = (\delta_1,....,\delta_p)^T$ such that

$\delta_{j}=\left\{\begin{array}{lll}
1 & \text { if } & X_{j} \text { is included in the model } \\
0 & \text { if } & X_{j} \text { is not included in the model }
\end{array}\right.$


So we have that $\delta_j$ "controls" the inclusion of $X_j$ in the model, and we treat $\boldsymbol\delta$ as a *parameter*. We then write  

$$\boldsymbol\eta = g^{-1}(\delta_1\beta_1X_1+....+\delta_p\beta_pX_p)$$

and we assign a prior both on $\boldsymbol\delta$ and $\boldsymbol\beta$:
 $$\delta_j \stackrel{i.i.d.}{\sim} Bern(w) \quad\quad\quad \beta_j \stackrel{ind.}{\sim}N(\beta_{0j, \sigma^2_{0j}})$$

The resulting joint prior is $p(\beta_j,\delta_j)=(1-w)\lambda_0 + wdN(\beta_j|\beta_{0j},\sigma^2_{0j})$ and it's called **spike and slab prior**.

We also assigned a prior to $w \in (0,1)$ as $w \stackrel{}{\sim}Beta(a,b)$.
In our case we decide to set hypermarametrs a = b = 1, so that  $w \stackrel{}{\sim}Unif(0,1)$ and $E(w) = 0.5$, to avoid to favours simpler/complex models.

Also in this case JAGS will help use to find the posterior distribution for $\boldsymbol\beta \quad $and$ \quad \boldsymbol\delta$.

Notice that: 

* if $\delta_{j}^{(s)} = 0$ then implicitly $\beta_{j}^s = 0$ and so the predictor $X_j$ is not included in the model at time s
* if $\delta_{j}^{(s)} = 1$ then  the predictor $X_j$ is  included in the model at time s with coefficients $\beta_{j}^s$

Therefore, we can use the frequency of $\delta_j^{(s)} = 1 \quad for \quad s=1,...S$ to estimate $\hat{p}X_j$  *i.e* the **posterior probability of inclusion** of predictor $X_j$. Basically, for each predictor we count the number of times that $\delta_j$ is = 1 over the S draws.

Formally: $$\hat{p}(X_j) = \frac{1}{S}\sum_{s=1}^{S} \delta_{j}^{(s)} $$

We can see the posterior probability of inclusion for each predictor in a plot:

![probl_inc_logi](https://user-images.githubusercontent.com/78596444/192018701-15e46f9c-ef67-4a2a-8d0e-ea50fc07ef2b.png)

**How can we use this kind of results to do our predictions? BMA!**

#### Bayesian Model Averaging (BMA)

### Bayesian Model Averaging (BMA)

The idea: Instead of selecting one single model, we use all the models generated with the spike and slab, and then compute a prediction of y which is averaged with respect to all these models.

We can use the posterior model probabilities to *weight* prediction obtained under each of the $M_k$ model. 

First we compute the predictive distribution for all of our S model 
$$\begin{aligned}
p\left(y^{} \mid y_{1}, \ldots, y_{n}, \mathcal{M_s}\right) &=\int p\left(y^{} \mid \boldsymbol{\beta}, \mathcal{M_s}\right) \cdot p\left(\boldsymbol{\beta} \mid \mathcal{M_s}, y_{1}, \ldots, y_{n}\right) d \boldsymbol{\beta} \\
&=\int \text { Sampling distribution at } y^{*} \cdot \text { Posterior of } \boldsymbol{\beta} d \boldsymbol{\beta}
\end{aligned} $$
and now we use all these predictive distribution to obtain one last average predictive distribution for our individuals.

This approach is called **Bayesian Model averaging**


$$p\left(y^{*} \mid y_{1}, \ldots, y_{n}\right)=\sum_{k=1}^{K} p\left(y^{*} \mid y_{1}, \ldots, y_{n}, \mathcal{M}_{k}\right) p\left(\mathcal{M}_{k} \mid y_{1}, \ldots, y_{n}\right) \quad (1)$$

In general, this requires the computation of posterior model probabilities, but since we are dealing with GLMs, we are not able to compute the posterior model probability, so we need to find a way to approximate (1).

At each iteration we have:

* $\boldsymbol\beta^{(s)} = (\beta_1^{s},....,\beta_p^{(s)})$
* $\boldsymbol\delta^{(s)} = (\delta_1^{s},....,\delta_p^{(s)})$

and we can think to $\boldsymbol\delta^{(s)}$ as a draw from the posterior $p\left(\mathcal{M}_{k} \mid y_{1}, \ldots, y_{n}\right)$.
This means that we can approximate the *posterior predictive* distribution of Y*, for a new subject $\boldsymbol{x^*} = (x_1^*,....,x_p^*)^T$ in this way:

For s = 1,....,S 

1. Compute $\eta^{s} = \delta_1^{(s)}\beta_1^{(s)}x_1^* + ..... +\delta_p^{(s)}\beta_p^{(s)}x_p^*$ (basically we are computing a linear predictor under a different model)
2. Compute $\pi^{(s)} = g^{-1}(\eta^{(s)})$
3. Draw $y^{*(s)}$ from $p(y*|\pi^{(s)})$ 

And eventually the output ${y^{*(1)},.....,y^{*(S)}}$ is a *BMA sample from the predictive distribution of Y for subject i*.

We can now evaluate how good is our model in prediction, comparing our prediction with the true values from test data and creating a confusion matrix:

![conf_mat_bma](https://user-images.githubusercontent.com/78596444/192019650-fbca26ff-7cfd-43b4-9350-f2ab17a7da52.png)

We obtained:
 
* Overall accuracy: $\approx85.5%$
* Sensitivity: $\approx85.6\%$
* Specificity: $\approx86.6\%$

And comparing this results with the ones obtained performing prediction with the model with all variables included, we can state that thanks to spike and slab variable selection and BMA, we obtained an improvement in our prediction, especially in terms of specificity which raised to $\approx86.6\%$ 


## Beta regression

We also decided to model the original chance of being admitted, using a Beta regression.

The beta regression models are used to model variables that assume values in the interval (0, 1). They're based on the assumption that the response is beta-distributed and that its mean is related to a set of regressors through a linear predictor and a link function. The model also includes a precision parameter on which we to put a prior^[[Beta Regression in R - Francisco Cribari-Neto, Achim Zeileis](https://www.jstatsoft.org/article/view/v034i02)].


Since we assume that $Y_i \sim Beta(a,b)$ we need to re-parameterize $a$ and $b$ to be function of the mean $\mu$ and the precision $\phi$. We proceed by letting:

$$a = \mu \ \phi \\
b= (1-\mu)\phi$$
 
This parameterizazion holds and we can demonstrate it, since:
$$\phi=\frac{a}{\mu}\\
b=(1-\mu)\frac{a}{\mu}\\
b=\frac{a}{\mu}  -  \frac{a\mu}{\mu} \\
b=\frac{a}{\mu} - a \ \ \ ; \ \ \ 
a+b=\frac{a}{\mu} \\
\mu=\frac{a}{a+b}$$

Now by replacing $\mu$, we can solve for $\phi$, so we have that:
$$a=\bigg(\frac{a}{a+b} \bigg) \ \phi \\
a=\frac{a\phi}{a+b}   \\
(a+b)a=a\phi \ \ ; \ \ a^2+ba = a\phi \\
\phi = a+b$$


So we end with:

- Shape 1 = $a$ = $\mu\phi$
- Shape 2 = $b$ = $(1-\mu)\phi$


 We will create a linear model for $\mu$, while for $\phi$ we put a weakly informative Gamma prior, since we need the parameter to be >0.
 
To visualize how this new parameterization holds (i.e the distribution does not change) we can plot together some density beta distributions with the two different parameterization:

![proof_beta](https://user-images.githubusercontent.com/78596444/192021121-917c2131-059a-453e-89c6-4fe7007406e0.png)

The linear model for $g(\mu)$ will be:
$$g(\mu)=\eta=\beta^Tx_i $$
 and we decided to use, as in logistic, a logit link function:
 $$logit(\mu)=log\bigg(\frac{\mu}{1-\mu} \bigg)=\beta^Tx_i $$
$$\mu=h(\beta^Tx_i)=\frac{e^{\beta^T x_i}}{1+e^{\beta^T x_i}} $$
 
 To obtain the parameters of the Beta we need also $\phi$, on which we put a Gamma prior:
 $$\phi \sim Gamma(\delta,\tau)$$
 
 The likelihood of $Y_i|a_i,b_i \ \stackrel{ind.}{\sim}Beta(a_i,b_i)$ is:
 
 $$p (\boldsymbol{y}|\beta)=\prod^n_{i=1}p(y_i|a_i,b_i)=\\
 \prod^n_{i=1}p(y_i|\mu_i\phi,(1-\mu_i)\phi)=\\
 =\prod^n_{i=1}\frac{\Gamma(\phi)}{\Gamma(\mu_i\phi)\Gamma((1-\mu_i)\phi)}  \ y_i^{\mu_i\phi-1} + (1-y_i)^{(1-\mu_i)\phi-1}$$
 
 Which is a function of $\beta$ since $\mu_i$ comes from $\mu_i=h(\beta^Tx_i)$.
 
 As usual, we have a prior on $\beta$:
 $$\beta_j \stackrel{ind}{\sim} N(\beta_{0j}, \sigma^2_{0j})$$

With these element, we implemented a Metropolis-Hastings algorithm using JAGS.

JAGS algorithm has:
 * As likelihood: $Y_i \stackrel{ind.}{\sim} dBeta(\tilde{a},\tilde{b})$
 * $\tilde{a}=\mu_i \phi$
 * $\tilde{b}=(1-\mu_i) \phi$
 * a model which is: $log \frac{\mu_i}{1-\mu_i}=\beta^Tx_i$
 while the priors are:
 * for $\beta$ we put a $dN(0,0.001)$
 * for $\phi$ we put a $dGamma(0.01,0.01)$
 
 
We decided to be weakly informative and let the model "learn" from the data.


### MCMC Diagnostic

#### Traceplots
![traeplot_beta1](https://user-images.githubusercontent.com/78596444/192022000-c4b1204b-c38d-47f4-98dc-bad4509eb633.png)
![traeplot_beta2](https://user-images.githubusercontent.com/78596444/192022189-7deb582e-3f97-43bf-ae00-90fc2e14e6de.png)

We can observe that they're all centered around the mode, as we would like to have.

#### Autocorrelation

![acf_1](https://user-images.githubusercontent.com/78596444/192024597-060573ed-37c1-418f-b70d-233133002e3e.png)
![acf_2](https://user-images.githubusercontent.com/78596444/192024627-e3d40613-1387-492f-bf0a-ca8e5bf9721a.png)

In the graphical checks we didn’t find unusual pattern and the convergence near the mode is reached early, so we don’t need to make thinning or to set a burn-in period.

Now let’s see if the diagnostic tests show problems:

#### Geweke test

| $\beta_1$  | $\beta_2$  | $\beta_3$  | $\beta_4$  | $\beta_5$  | $\beta_6$  | $\beta_7$  | $\beta_8$  | $\beta_9$  | $\beta_{10}$  | $\beta_{11}$  |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|-0.89|	1.4|	0.96|	-0.47|	0.08|	0.16|	0.14|	-0.9|	-0.58|	-0.64|	0.19

#### Effective Sample Size

| $\beta_1$  | $\beta_2$  | $\beta_3$  | $\beta_4$  | $\beta_5$  | $\beta_6$  | $\beta_7$  | $\beta_8$  | $\beta_9$  | $\beta_{10}$  | $\beta_{11}$  |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|5000|	5000 |	50000 |	5000 |	5000	| 5000 |	5000 |	4555.9 |	5000 |	5000 |	5000 |

Then, to visualize how posterior values of the parameters are distributed we can use boxplots again:

![image](https://user-images.githubusercontent.com/78596444/192024966-02b0be83-eea4-440b-a528-fa7abefc128f.png)


### Predictions

After creating the model, we used the posterior distribution of Beta parameters to approximate the posterior predictive distribution of $y^*$ based on the test observations $x^*$ after the split we made before. 

Since in the test vector we have the original chances of admission, we can compare them with our prediction in order to evaluate the predictive performance of the model.

To approximate the posterior predictive distribution of $Y$:

For i = 1,...S:

*    For j = 1,....,125:
     1. Compute $\eta_j ^{(s)} = \boldsymbol\beta^{(s)T}\boldsymbol{x}^*_j$
     2. Compute $\mu_j ^{(s)} = h(\boldsymbol\beta^{(s)T}\boldsymbol{x}^*_j)$
     3. Compute the parameters $a_j^{(s)}=\mu_j^{(s)}\phi_j^{(s)}$
     3. Compute the parameters $b_j^{(s)}=(1-\mu_j^{(s)})\phi_j^{(s)}$
     4. Draw a $y^{*(s)}_j\sim Beta(a_j^{(s)},b_j^{(s)})$

Now we can compute the mode of the 125 predictive posterior distribution $(y^*_1,...y_{125}^*) \sim (a_j,b_j)$ and then compare it with the observed values:

![compare_test](https://user-images.githubusercontent.com/78596444/192025218-a97bd42d-c7b6-418a-84ee-afafb0e4a0d7.png)

We can also calculate the MSE, which is $\approx 0.004$

### Spike & Slab Variable Selection

To perform variable selection in our Beta regression we used a spike & slab prior as in the logistic case. Also here we assigned a prior to $\gamma$:
$$\gamma_j\stackrel{i.i.d.}{\sim}Ber(w)$$
and the usual prior to $\boldsymbol{\beta}$:
$$\beta_j\stackrel{ind}{\sim}N(\beta_0,\sigma^2_{0j})$$

such that their joint prior will be the "spike & slab" prior.

We assigned also a prior on the parameter $w$ of the Bernoulli distribution of $\gamma$.

To be non informative, we assigned $$w \sim Beta(1,1) $$

Barplot of the estimate of the posterior probability of inclusion for each predictor:

![barplot_prob_incl](https://user-images.githubusercontent.com/78596444/192025422-605e5899-909f-4db6-9fec-9ffda9eea648.png)

####  Bayesian Model Averaging (BMA)

As we did in the logistic case, we can implement a BMA strategy, starting from the result of the Spike & Slab variable selection, to predict new values $Y^*$.
We did it with an algorithm which is equal to the one used for predictions above, the only difference is that the linear predictor is the result of:
$$\eta=\boldsymbol{\gamma^T\beta^T} \  x^* $$

![compare_test_bma](https://user-images.githubusercontent.com/78596444/192025517-563c953e-8617-4669-af46-94a7a965ef08.png)

The MSE in this case is $\approx0.008$

The BMA strategy didn't improve the MSE, but this is a sort of "frequentist ground" for comparison. The strength of bayesian predictions is that we obtain a distribution for the future observation, instead of a single fitted value. A better idea would be to compare the posterior predictive distribution before and after the Spike and Slab variable selection with the true value from test data for four different students:

![compare_students_distrib](https://user-images.githubusercontent.com/78596444/192025665-e9263f15-90a5-422a-b99a-eb8632d32c5a.png)

## Conclusion

From the result of our analysis we can state that the logistic model we implemented performs very well, but it can tell a student only whether it has High or Low chance of being admitted. While whit the beta regression model, we are able to provide to each individuals a distribution of their probability of being admit. Comparing the mode of these distributions with the true value of the test dataset, the model doesn’t show high precision of the prediction, but this comparison is quite limiting for a bayesian approach in which we can get a distribution for the chance of each individual instead of a fitted value.
