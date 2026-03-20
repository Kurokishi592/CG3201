# Slide 1

© Copyright National University of Singapore. All Rights Reserved. 
© Copyright National University of Singapore. All Rights Reserved. 
CG3201 Machine Learning & 
Deep Learning
Lecture 3 Bayesian Learning and Modelling

# Slide 2

© Copyright National University of Singapore. All Rights Reserved. 
Recap Lecture 2
From linear classifiers to Support Vector Machines (SVM).
Hard-Margin SVM: classify with margins.
Primal form.
Dual form from KKT conditions.
Soft-Margin SVM: relaxing constraints for linearly inseparable data.
Primal form and dual form.
Converting constraints to losses via the hinge loss.
Kernel SVM: separating data in the feature space with the kernel trick.
The kernel trick and SVM formulation with the kernel trick.
Kernel functions and impact of hyperparameters.
The Mercer’s theorem and the Mercer’s condition.

# Slide 3

© Copyright National University of Singapore. All Rights Reserved. 
Comparing Models from Results
In our discussion of SVM, we compare the different linear classifiers via 
the margins available for the classification:
The larger the margin, the better the SVM (assuming hard-margin SVM).
In reality, we cannot really understand the exact features, not to say 
quantify the margins.
Instead, we care more about the accuracy of the task (and sth else?)

# Slide 4

© Copyright National University of Singapore. All Rights Reserved. 
Comparing Models from Results
Suppose we have a spam vs. ham email task.
Model A: of the 100 spams, it can correctly detect 70 of them.
Model B: of the 100 spams, it can correctly detect 80 of them.
Which one is better? What are we actually comparing?
Probability. We say Model B has a higher probability of detecting spams 
when received.
What is actually probability?

# Slide 5

© Copyright National University of Singapore. All Rights Reserved. 
Bayesian Learning Fundamentals 1:
✓ Introduction / Recall to Probability
✓ The Bayes’ Rule
✓ Univariate Probability Distribution Models

# Slide 6

© Copyright National University of Singapore. All Rights Reserved. 
Probability: Two Interpretations
Probability theory is nothing, but common sense reduced to calculation. 
            – Pierre Laplace, 1812
Frequentist interpretation: probabilities represent long run 
frequencies of events that can happen multiple times.
Flip the coin many times, we expect it to land heads about half 
the time.
Bayesian interpretation: quantify uncertainty or 
ignorance about something
The coin is equally likely to land heads/tails on next toss.

# Slide 7

© Copyright National University of Singapore. All Rights Reserved. 
Properties of Probability of event(s): I
Note: for this course, probability of an event denoted as: 𝑃(𝐴)
𝑃(𝐴) satisfy: 0 ≤ 𝑃 𝐴 ≤ 1.
Joint probability that events 𝐴 and 𝐵 both happening: 𝑃(𝐴, 𝐵);
If 𝐴 and 𝐵 are independent events, 𝑃 𝐴, 𝐵 = 𝑃 𝐴 𝑃(𝐵).
Probability of event 𝐴 or 𝐵 happening (probability of union of 𝐴 and 𝐵): 
𝑃 𝐴 ∨ 𝐵 = 𝑃 𝐴 + 𝑃 𝐵 − 𝑃(𝐴, 𝐵);
If 𝐴 and 𝐵 are mutually exclusive (cannot happen at the same time), then 
𝑃 𝐴 ∨ 𝐵 = 𝑃 𝐴 + 𝑃(𝐵);

# Slide 8

© Copyright National University of Singapore. All Rights Reserved. 
Properties of Probability of event(s): II
Note: for this course, probability of an event denoted as: 𝑃 𝐴  or 𝑝(𝐴).
Conditional probability of 𝐵 happening given 𝐴 has occurred:
𝑃(𝐵|𝐴) ≜ 𝑃(𝐴, 𝐵)
𝑃(𝐴)
Event 𝐴 is independent of 𝐵 if 𝑃 𝐴, 𝐵 = 𝑃 𝐴 𝑃(𝐵);
Events 𝐴 ad 𝐵 are conditionally independent given 𝐶 if:
𝑃 𝐴, 𝐵 𝐶 = 𝑃 𝐴 𝐶 𝑃(𝐵|𝐶)

# Slide 9

© Copyright National University of Singapore. All Rights Reserved. 
Discrete and Continuous Random Variables (RV)
A mathematical formalization of a quantity or object which depends 
on random events.
A mapping or a function from possible outcomes in a sample space to 
a measurable space, often the real numbers.
Discrete variable Continuous variable
Countable numbers of value (e.g., number of students) Infinitely many real values (e.g., time to complete a task)
Distribution defined by probability mass function (pmf) Distribution defined by cumulative distribution function 
(cdf) and probability density function (pdf)
𝑝 𝑥 ≜ 𝑃 𝑋 = 𝑥
∑𝑝 𝑥 = 1
𝑃 𝑥 ≜ 𝑃(𝑋 ≤ 𝑥)
𝑝 𝑥 ≜ 𝑑
𝑑𝑥𝑃(𝑥)
𝑃 𝑎 < 𝑋 ≤ 𝑏 = න
𝑎
𝑏
𝑝 𝑥 𝑑𝑥 = 𝑃 𝑏 − 𝑃(𝑎)

# Slide 10

© Copyright National University of Singapore. All Rights Reserved. 
Discrete and Continuous Random Variables (RV)
Important: Note whether the variable is discrete or continuous! Could be 
confusing sometimes. Example:
A large quantity of green tea (e.g., 1000kg) is to be sold. 𝑋 is the weight of green tea to be 
sold. Suppose any quantity can be sold, is 𝑋 discrete or continuous?
What if the green tea can only be sold in packets of 10kg, is 𝑋 now discrete or continuous?
Discrete variable Continuous variable
Countable numbers of value (e.g., number of students) Infinitely many real values (e.g., time to complete a task)
Distribution defined by probability mass function (pmf) Distribution defined by cumulative distribution function 
(cdf) and probability density function (pdf)
𝑝 𝑥 ≜ 𝑃 𝑋 = 𝑥
∑𝑝 𝑥 = 1
𝑃 𝑥 ≜ 𝑃(𝑋 ≤ 𝑥)
𝑝 𝑥 ≜ 𝑑
𝑑𝑥𝑃(𝑥)
𝑃 𝑎 < 𝑋 ≤ 𝑏 = න
𝑎
𝑏
𝑝 𝑥 𝑑𝑥 = 𝑃 𝑏 − 𝑃(𝑎)

# Slide 11

© Copyright National University of Singapore. All Rights Reserved. 
Bayes’ Rule
“Bayesian”: refer to inference methods that represent “degrees of 
certainty” using probability theory, and which leverage Bayes’ rule, to 
update the degree of certainty given data.
Layman term: Update our belief based on prior belief and data so that we infer 
something with a certain degree of uncertainty. 
Bayes’ rule: a formula for computing the probability distribution over 
possible values of an unknown quantity 𝐻 given observed data 𝑌 = 𝑦.
𝑝 𝐻 = ℎ|𝑌 = 𝑦 = 𝑝 𝐻 = ℎ 𝑝 𝑌 = 𝑦|𝐻 = ℎ
𝑝 𝑌 = 𝑦
Follows the product rule of probability:𝑝 ℎ 𝑦 𝑝 𝑦 = 𝑝 ℎ 𝑝 𝑦 ℎ = 𝑝(ℎ, 𝑦)
𝑝 𝐻 = ℎ 𝑌 = 𝑦  : Posterior distribution / conditional distribution.

# Slide 12

© Copyright National University of Singapore. All Rights Reserved. 
Bayes’ Rule: in Detail
𝑝 𝐻 = ℎ 𝑌 = 𝑦 = 𝑝 𝐻 = ℎ 𝑝(𝑌 = 𝑦|𝐻 = ℎ)
𝑝(𝑌 = 𝑦)
𝑝(𝐻): what we know about possible values of 𝐻 before we see any data – the prior 
distribution (layman terms: prior belief).
𝑝(𝑌|𝐻 = ℎ): the distribution over possible outcomes 𝑌 we expect to see if 𝐻 = ℎ – the 
likelihood (layman terms: the observed data).
Multiplying the prior distribution and the likelihood – unnormalized joint distribution 𝑝(𝐻 =
ℎ, 𝑌 = 𝑦). Convert into a normalized distribution by dividing by 𝑝(𝑌 = 𝑦), which is 
denoted as the marginal likelihood.
𝑝(𝐻 = ℎ|𝑌 = 𝑦): represents our new belief state (updated belief) about the possible 
values of 𝐻 – the posterior distribution.
posterior ∝ prior × likelihood

# Slide 13

© Copyright National University of Singapore. All Rights Reserved. 
Bayes’ Rule: Example 1 – Weather Forecast
Posterior 
Probability
Prior 
Probability
Likelihood
Marie is getting married tomorrow at 
an outdoor ceremony in the desert. 
In recent years, it has rained only 5 
days each year.
When it actually rains, the weatherman 
has forecast rain 90% of the time.
When it doesn't rain, he has forecast 
rain 10% of the time. 
Unfortunately, the weatherman is 
forecasting rain for tomorrow. 
What is the probability it will rain on the 
day of Marie's wedding?
𝑌 =  1 (𝑌): Weatherman forecast it rains
𝐻 =  1 (𝐻): It Actually Rains
𝑃 𝐻 = 5
365 = 0.0137
𝑃(𝐻|𝑌)  = ? ?
𝑃 𝑌 𝐻 =  0.9
𝑃 𝑌 ෩𝐻 = 0.1

# Slide 14

© Copyright National University of Singapore. All Rights Reserved. 
Bayes’ Rule: Example 1 – Weather Forecast
We know that: 𝑃 𝐻 =
5
365 = 0.0137 and 𝑃 𝑌 𝐻 =  0.9, 𝑃 𝑌 ෩𝐻 = 0.1. 
What we want: 𝑃(𝐻|𝑌).
𝑃 𝐻 𝑌 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃 𝑌 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃(𝑌|𝐻)𝑃 𝐻 + 𝑃(𝑌| ෩𝐻)𝑃 ෩𝐻
𝑃(𝐻|𝑌)  = 0.0137(0.9)
0.9(0.0137) + 0.1(1 − 0.0137) = 0.111
The probability it will rain on the day of Marie's wedding, given the 
weatherman is forecasting rain for tomorrow is 0.111

# Slide 15

© Copyright National University of Singapore. All Rights Reserved. 
Bayes Rule: Example 2 – Disease Testing
Suppose there is an outbreak of a disease 
(e.g., COVIDv2). Jack is taking a diagnostic 
test to determine if he is infected. 
Suppose the sensitivity (true positive rate) of 
the diagnostic is 87.5%, and the specificity 
(true negative rate) of the diagnostic is 97.5%.
Also suppose the prevalence of the disease in 
the area where Jack lives is 10%.
Unfortunately, Jack is tested positive. What is 
the probability of Jack being infected?
𝑌 = 1 (𝑌): Diagnostic test positive
𝐻 = 1 (𝐻): Jack is infected
Prior 
Probability 𝑃 𝐻 = 0.1
Likelihood
𝑃 𝑌 𝐻 =  0.875
𝑃 ෨𝑌 ෩𝐻 = 0.975
Posterior 
Probability 𝑃 𝐻|𝑌 = ? ?

# Slide 16

© Copyright National University of Singapore. All Rights Reserved. 
Bayes’ Rule: Example 2 – Disease Testing
We know that: 𝑃 𝐻 = 0.1 and 𝑃 𝑌 𝐻 = 0.875, 𝑃 ෨𝑌 ෩𝐻 = 0.975. 
What we want: 𝑃(𝐻|𝑌).
𝑃 𝐻 𝑌 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃 𝑌 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃(𝑌|𝐻)𝑃 𝐻 + 𝑃(𝑌| ෩𝐻)𝑃 ෩𝐻 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃 𝑌 𝐻 𝑃 𝐻 + 1 − 𝑃 ෨𝑌 ෩𝐻 𝑃 ෩𝐻
𝑃 𝐻 𝑌 = 𝑃 𝐻 𝑃 𝑌 𝐻
𝑃 𝑌 𝐻 𝑃 𝐻 + 1 − 𝑃 ෨𝑌 ෩𝐻 1 − 𝑃 𝐻
= 0.1 × 0.875
0.875 × 0.1 + 1 − 0.1 (1 − 0.975) = 0.795
The chance of Jack being infected is 79.5%.

# Slide 17

© Copyright National University of Singapore. All Rights Reserved. 
Bayes Rule: Example 2 – Disease Testing
Suppose there is an outbreak of a disease 
(e.g., COVIDv2). Jack is taking a diagnostic 
test to determine if he is infected. 
Suppose the sensitivity (true positive rate) of 
the diagnostic is 87.5%, and the specificity 
(true negative rate) of the diagnostic is 97.5%.
Now suppose the prevalence of the disease in 
the area where Jack lives is changed to 1%.
Jack is tested positive. What is the probability 
of Jack being infected now?
𝑌 = 1 (𝑌): Diagnostic test positive
𝐻 = 1 (𝐻): Jack is infected
Prior 
Probability 𝑃 𝐻 = 0.01
Likelihood
𝑃 𝑌 𝐻 =  0.875
𝑃 ෨𝑌 ෩𝐻 = 0.975
Posterior 
Probability 𝑃 𝐻|𝑌 = ? ?

# Slide 18

© Copyright National University of Singapore. All Rights Reserved. 
Bayes’ Rule: Example 2 – Disease Testing
We know that: 𝑃 𝐻 = 0.01 and 𝑃 𝑌 𝐻 = 0.875, 𝑃 ෨𝑌 ෩𝐻 = 0.975. 
What we want: 𝑃(𝐻|𝑌).
𝑃 𝐻 𝑌 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃 𝑌 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃(𝑌|𝐻)𝑃 𝐻 + 𝑃(𝑌| ෩𝐻)𝑃 ෩𝐻 = 𝑃 𝐻 𝑃(𝑌|𝐻)
𝑃 𝑌 𝐻 𝑃 𝐻 + 1 − 𝑃 ෨𝑌 ෩𝐻 𝑃 ෩𝐻
𝑃 𝐻 𝑌 = 𝑃 𝐻 𝑃 𝑌 𝐻
𝑃 𝑌 𝐻 𝑃 𝐻 + 1 − 𝑃 ෨𝑌 ෩𝐻 1 − 𝑃 𝐻
= 0.01 × 0.875
0.875 × 0.01 + 1 − 0.01 (1 − 0.975) = 0.261
The chance of Jack being infected is 26.1%, even if tested positive.

# Slide 19

© Copyright National University of Singapore. All Rights Reserved. 
Bernoulli Distribution
One of the simplest probability distribution – can be used to model binary 
events. (E.g., coin flipping)
Explaining the Bernoulli distribution with coin flipping. Denote 𝑌 = 1 as coin lands on 
heads, and 𝑌 = 0 as coin lands on tails.
Suppose the probability of the coin lands on heads is 0 ≤ 𝜃 ≤ 1. Thus, 𝑝 𝑌 = 1 = 𝜃
and 𝑝 𝑌 = 0 = 1 − 𝜃. Then random variable 𝑌 follow the Bernoulli distribution: 
𝑌~Ber(𝜃).
Ber 𝑦 𝜃 = ቊ1 − 𝜃 𝑖𝑓 𝑦 = 0
𝜃 𝑖𝑓 𝑦 = 1 ≜ 𝜃𝑦 1 − 𝜃 1−𝑦

# Slide 20

© Copyright National University of Singapore. All Rights Reserved. 
Binomial Distribution
Now, if we flip a coin multiple times. Then the sum of the Bernoulli random 
variables will follow a Binomial distribution.
The Bernoulli distribution is a special case of the binomial distribution.
Suppose we observe a set of 𝑁 Bernoulli trials (e.g., toss a coin 𝑁 times), 𝑦𝑛~Ber(⋅
|𝜃) for 𝑛 = 1: 𝑁. Define 𝑠 to be the total number of heads 𝑠 ≜ ∑𝑛=1
𝑁 𝕀(𝑦𝑛 = 1). 
The distribution of 𝑠 follows the binomial distribution:
Bin 𝑠 𝑁, 𝜃 ≜ 𝑁
𝑠 𝜃𝑠 1 − 𝜃 𝑁−𝑠 = 𝑪𝑠𝑁𝜃𝑠 1 − 𝜃 𝑁−𝑠
binomial coefficient: 𝑁
𝑠 ≜ 𝑁!
𝑁 − 𝑠 ! 𝑠!

# Slide 21

© Copyright National University of Singapore. All Rights Reserved. 
Binomial Distribution

# Slide 22

© Copyright National University of Singapore. All Rights Reserved. 
Binomial Distribution: Example

# Slide 23

© Copyright National University of Singapore. All Rights Reserved. 
Predicting a Binary Variable
When we want to predict a binary variable 𝑦 ∈ {0,1} given input 𝑥 ∈ 𝒳, we 
need to use a conditional probability distribution: 
𝑝 𝑦 𝒙, 𝜽 = Ber(𝑦|𝑓 𝒙; 𝜽 ). 
𝑓 𝒙; 𝜽  is a function that predicts the parameter of the distribution.
To avoid requiring 0 < 𝑓 𝒙; 𝜽 ≤ 1, let 𝑓 be an unconstrained function, we 
use the following model:
𝑝 𝑦 𝒙, 𝜽 = Ber(𝑦|𝜎(𝑓 𝒙; 𝜽 )). 
𝜎() is the sigmoid (logistic) function – “sigmoid” means S-shaped.
𝜎 𝑎 ≜ 1
1 + 𝑒−𝑎

# Slide 24

© Copyright National University of Singapore. All Rights Reserved. 
The Sigmoid Function Visualized
Comparing the sigmoid function 𝜎 𝑎 = 1 + 𝑒−𝑎 −1 and the Heaviside 
step function 𝕀(𝑎 > 0).
The sigmoid function can be thought of as a “soft” version of the heaviside 
step function.

# Slide 25

© Copyright National University of Singapore. All Rights Reserved. 
Categorical and Multinomial Distribution
Problems for Bernoulli and Binomial distributions?
Can only model the probability for an event with ONLY 2 outcomes.
Generalize the Bernoulli to 𝐶 > 2 values.
Categorical distribution: 
A discrete probability distribution with one parameter per class:
Cat 𝑦 𝜽 ≜ ς𝑐=1
𝐶 𝜃𝑐
𝕀 𝑦=𝑐 , thus 𝑝 𝑦 = 𝑐 𝜽 = 𝜃𝑐.
Alternatively, convert the discrete variable 𝑦 into a one-hot vector 𝒚 with 𝐶 elements.
E.g., if 𝐶 = 3, then the classes are encoded as (1,0,0), (0,1,0), and (0,0,1).
As such, the categorical distribution is also written for 𝒚:
Cat 𝒚 𝜽 ≜ ෑ
𝑐=1
𝐶
𝜃𝑐
𝑦𝑐

# Slide 26

© Copyright National University of Singapore. All Rights Reserved. 
Categorical and Multinomial Distribution
The categorical distribution is a special case of multinomial distribution. 
Repeat the categorical trials 𝑁 times (e.g., roll a 𝐶-sided dice 𝑁 times).
Define 𝑦𝑐 to be the total number of times face 𝑐 shows up: 𝑦𝑐 = 𝑁𝑐 ≜ ∑𝑛=1
𝑁 𝕀(𝑦𝑛 = 𝑐). 
Define 𝒚 as the vector with 𝐶-dim, each element being 𝑦𝑐.
The distribution of 𝒚 follows the multinominal distribution. Note that 𝒚 is no longer a 
one-hot vector.
ℳ 𝑦 𝑁, 𝜃 ≜ 𝑁
𝑦1 … 𝑦𝑐
ෑ
𝑐=1
𝐶
𝜃𝑐
𝑦𝑐 = 𝑁
𝑁1 … 𝑁𝑐
ෑ
𝑐=1
𝐶
𝜃𝑐
𝑁𝑐
𝜃𝑐 is the probability side 𝑐 shows up and 𝑁
𝑁1…𝑁𝑐
≜
𝑁!
𝑁1!𝑁2!…𝑁𝐶! is the multinominal 
coefficient, which is the number of ways to divide a set of size 𝑁 = ∑𝑐=1
𝐶 𝑁𝑐 into 
subsets with sizes 𝑁1 up to 𝑁𝐶. If 𝑁 = 1, the multinomial distribution becomes the 
categorical distribution, if 𝐶 = 2, the multinomial distribution is the binomial distribution.

# Slide 27

© Copyright National University of Singapore. All Rights Reserved. 
Predicting a Categorical Variable with Softmax
When we want to predict a categorical variable 𝑦 given input 𝑥 ∈ 𝒳, we 
need to use a conditional probability distribution: 
𝑝 𝑦 𝒙, 𝜽 = Cat(𝑦|𝑓 𝒙; 𝜽 ). 
To avoid requiring 0 < 𝑓𝑐 𝒙; 𝜽 ≤ 1 and ∑𝑐=1
𝐶 𝑓𝑐 𝒙; 𝜽 = 1, let 𝑓 be an 
unconstrained function, we use the following model:
𝑝 𝑦 𝒙, 𝜽 = Cat(𝑦|softmax(𝑓 𝒙; 𝜽 )). 
softmax(), also written as sfm() is the softmax (multinomial logistic) function
softmax 𝒂 𝑐 ≜ 𝑒𝑎1
∑𝑐′=1
𝐶 𝑒𝑎𝑐′ , … , 𝑒𝑎𝐶
∑𝑐′=1
𝐶 𝑒𝑎𝑐′

# Slide 28

© Copyright National University of Singapore. All Rights Reserved. 
Something More on Softmax
1. Properties of Softmax: 
a. softmax 𝒂 𝑐 ∈ 0,1 , and ∑𝑐=1
𝐶 softmax 𝒂 𝑐 = 1. The softmax function has been 
applied to almost ALL classification tasks in deep learning thanks to these properties.
The input to the softmax 𝒂 = 𝑓 𝒙; 𝜽 are called logits.
2. The softmax function, at its extremes, acts like the argmax function.
Divide each 𝑎𝑐 (the components for each class of 𝒂) by a constant 𝑇, denoted as the 
Temperature. Then as 𝑇 → 0, we find the following:
softmax 𝒂/𝑇 𝑐 = ൝ 1.0 if 𝑐 = argmax
𝑐′
𝑎𝑐′
0.0 otherwise
At low temperatures, the distribution puts most of its probability mass in the most 
probable state (denoted as winner takes all), whereas at high temperatures, it 
spreads the mass uniformly.

# Slide 29

© Copyright National University of Singapore. All Rights Reserved. 
Illustration of Softmax Properties

# Slide 30

© Copyright National University of Singapore. All Rights Reserved. 
Univariate Gaussian Distribution
The most widely used distribution of real-valued random variables 𝑦 ∈ ℝ 
is the Gaussian distribution (equivalently, normal distribution). 
Suppose a continuous variable 𝑋 follow the Gaussian distribution, then 
cdf is: 𝑃 𝑥 ≜ 𝑃(𝑋 ≤ 𝑥) formulated as:
Φ 𝑥; 𝜇, 𝜎2 ≜ න
−∞
𝑥
𝒩 𝑧 𝜇, 𝜎2 𝑑𝑧 = 1
2 1 + erf 𝑧
2
Here, 𝜇 encodes the mean of the distribution, 𝜎2 encodes the variance, 
when 𝜇 = 0 and 𝜎 = 1, it refers to the standard normal distribution.
𝑧 = (𝑥 − 𝜇)/𝜎 and erf(𝑢) is the error function defined as:
erf 𝑢 ≜ 2
𝜋 න
0
𝑢
𝑒−𝑡2
𝑑𝑡

# Slide 31

© Copyright National University of Singapore. All Rights Reserved. 
Univariate Gaussian Distribution
The pdf is the derivative of the cdf: 𝑝 𝑥 ≜
𝑑
𝑑𝑥 𝑃(𝑥) and the pdf of the 
Gaussian is given by:
𝒩 𝑥 𝜇, 𝜎2 ≜ 1
2𝜋𝜎2 𝑒− 1
2𝜎2 𝑥−𝜇 2
RV 𝑥 that follows such pdf can be denoted as: 𝑥~𝒩(𝜇, 𝜎2).
For Gaussian, the following results apply:
Mean (expected value): 𝔼 𝒩 ⋅ 𝜇, 𝜎2 = 𝜇.
Variance: 𝕍 𝑋 = 𝔼 𝑋2 − 𝜇2, therefore 𝔼 𝑋2 = 𝜇2 + 𝜎2.
Standard deviation: std 𝑋 = 𝕍[𝑋] = 𝜎.

# Slide 32

© Copyright National University of Singapore. All Rights Reserved. 
Univariate Gaussian Distribution
The pdf is the derivative of the cdf: 𝑝 𝑥 ≜
𝑑
𝑑𝑥 𝑃(𝑥) and the pdf of the 
Gaussian is given by:
𝒩 𝑥 𝜇, 𝜎2 ≜ 1
2𝜋𝜎2 𝑒− 1
2𝜎2 𝑥−𝜇 2

# Slide 33

© Copyright National University of Singapore. All Rights Reserved. 
Other Common Univariate Distributions
Dirac delta: when the 𝜎 of Gaussian → 0.
Student 𝑡-distribution: less sensitive to 
outliers than the Gaussian distribution.
Laplace distribution: double-sided 
exponential distribution.

# Slide 34

© Copyright National University of Singapore. All Rights Reserved. 
Other Common Univariate Distributions
Beta distribution: a suitable model for the random 
behavior of percentages and proportions
Gamma distribution: a flexible distribution for 
positive real valued rv’s.

# Slide 35

© Copyright National University of Singapore. All Rights Reserved. 
Bayesian Learning and Modelling 1:
✓ Maximum Likelihood Estimation (MLE)
✓ MLE for Common Probability Distributions
✓ Empirical Risk Minimization

# Slide 36

© Copyright National University of Singapore. All Rights Reserved. 
Estimating Probability Models: Model Fitting
Now suppose we know that the data would follow a certain probability model, 
controlled by parameters 𝜽, but we don’t know what the model exactly is.
Key problem: how to learn these parameters 𝜽 from data?
The process of estimation 𝜃 from data 𝒟 is called model fitting, or equivalently, 
“training” – the core of machine learning.
The training boils down to an optimization of the form:
෡𝜽 = argmin
𝜽
ℒ(𝜽)
෡𝜽: a point estimate; ℒ(𝜽): loss function / objective function.
Important: how to define the loss function / objective function?
Inference: quantify the uncertainty from data sample – compute 𝑝(𝑦|𝑥, ෡𝜽).

# Slide 37

© Copyright National University of Singapore. All Rights Reserved. 
Maximum Likelihood Estimation (MLE)
The intuitive objective: pick the parameters that assign the highest prob. 
to the training data (usually labelled) – maximum likelihood estimation:
෡𝜽𝑚𝑙𝑒 = argmax
𝜽
𝑝(𝒟|𝜽)
Assume: training samples are independently sampled from the same distribution – iid 
assumption (independent and identical distributed).
𝑝 𝒟 𝜽 = ς𝑛=1
𝑁 𝑝(𝑦𝑛|𝑥𝑛, 𝜽) – likelihood; log likelihood takes its log ℓ 𝜽 ≜ log 𝑝(𝒟|𝜽).
Most optimization algorithms are designed to min cost functions (argmin 
instead of argmax), redefine the objective function to be the (conditional) 
negative log likelihood (NLL): maximize likelihood = minimize NLL.
NLL 𝜽 ≜ − log 𝑝 𝒟 𝜽 = − ෍
𝑛=1
𝑁
log 𝑝(𝑦𝑛|𝑥𝑛, 𝜽)
෡𝜽 = argmin
𝜽
ℒ(𝜽) = argmin
𝜽
NLL 𝜽

# Slide 38

© Copyright National University of Singapore. All Rights Reserved. 
MLE for Bernoulli Distribution
Suppose 𝑌 is a random variable representing a coin toss, where the event 
𝑌 = 1 corresponds to heads and 𝑌 = 0 corresponds to tails. 
Let 𝜃 = 𝑝(𝑌 = 1) be the probability of heads. The probability distribution for 
this rv is the Bernoulli. The NLL for the Bernoulli distribution is given by:
Observe: this is just the empirical fraction of heads, an intuitive result!

# Slide 39

© Copyright National University of Singapore. All Rights Reserved. 
MLE for Univariate Gaussian Distribution
Suppose 𝑌~𝑁(𝜇, 𝜎2) and 𝒟 = 𝑦𝑛: 𝑛 = 1: 𝑁 is an iid sample of size 𝑁. 
The parameters are 𝜽 = (𝜇, 𝜎2). We estimate with MLE:
1. Derive NLL:
 2. Derive optimal parameter:

# Slide 40

© Copyright National University of Singapore. All Rights Reserved. 
Empirical Risk Minimization (ERM)
The MLE is based on: NLL 𝜽 = − ∑𝑛=1
𝑁 log 𝑝(𝑦𝑛|𝑥𝑛, 𝜽)
Can we further generalize the MLE?
Recall: get optimal 𝜽 by ෡𝜽 = argmin
𝜽
ℒ(𝜽) = argmin
𝜽
∑𝑛=1
𝑁 ℓ(𝜽).
Generalize MLE: replace log loss term with other loss function!
ℓ 𝜽 = − log 𝑝(𝑦𝑛|𝑥𝑛, 𝜽) → ℓ(𝑦𝑛, 𝜽; 𝑥𝑛).
With the more generalized form, we have:
ℒ 𝜽 = ∑𝑛=1
𝑁 ℓ(𝑦𝑛, 𝜽; 𝑥𝑛) – Empirical risk minimization (ERM).
The expected loss where the expectation is taken w.r.t. the “empirical” (sampled) 
distribution.

# Slide 41

© Copyright National University of Singapore. All Rights Reserved. 
ERM: Minimizing Misclassification Rate
If we are solving a classification problem, an intuitive loss selection would 
be the 0-1 loss: (𝑓 being some kind of predictor)
ℓ01 𝑦𝑛, 𝜽; 𝑥𝑛 = ቊ0 𝑖𝑓 𝑦𝑛 = 𝑓(𝑥𝑛; 𝜽)
1 𝑖𝑓𝑦𝑛 ≠ 𝑓(𝑥𝑛; 𝜽)
Then the empirical risk becomes:
ℒ 𝜽 = ෍
𝑛=1
𝑁
ℓ01(𝑦𝑛, 𝜽; 𝑥𝑛)
This is defined as the empirical misclassification rate on the training set.

# Slide 42

© Copyright National University of Singapore. All Rights Reserved. 
Bayesian Learning and Modelling 2:
✓ Bayesian Decision Theory and the Classification Problem
✓ Maximum A Posteriori (MAP).
✓ Evaluation of Classifiers

# Slide 43

© Copyright National University of Singapore. All Rights Reserved. 
Decision Theory: Making the Right Decision
More often, we don’t actually know the type of distribution for the data either – 
not to mention the parameter of the distribution.
How can we make decision then without such prior knowledge?
Decision Theory: a set of quantitative methods for reaching optimal decisions.
Concerned with the reasoning underlying an agent’s choices, whether this is a mundane 
choice between taking the bus or getting a taxi, or a more far-reaching choice about 
whether to pursue a demanding political career. 
Layman term: how we can decide what action should be taken that could end 
up with the best result?

# Slide 44

© Copyright National University of Singapore. All Rights Reserved. 
(Bayesian) Decision Theory: in Detail
In decision theory, we assume the decision maker, or agent, has a set of 
possible actions, 𝒜, to choose from.
Example: doctor treat someone who may have cancer, actions 𝑎 ∈ 𝒜:
Do nothing;
Give patient expensive drugs with bad side effect but can save their life;
We can compute (implicitly) the cost of treating a person (test = 0: cancer 
negative; age = 0: young age; action = 1: apply the drugs).
cancer

# Slide 45

© Copyright National University of Singapore. All Rights Reserved. 
(Bayesian) Decision Theory: in Detail
Every action 𝑎 ∈ 𝒜 has cost and benefits, depending on the underlying 
state of nature ℎ ∈ ℋ. How can we explicitly model such cost / benefit?
Define: loss function ℓ(ℎ, 𝑎) – loss incurred if action 𝑎 is taken at state ℎ. 
The next question: how to leverage this loss?
Equivalently: how can we actually know quantitatively what is the best action if we 
compute the loss function?
cancer

# Slide 46

© Copyright National University of Singapore. All Rights Reserved. 
(Bayesian) Decision Theory: in Detail
Define: loss function ℓ(ℎ, 𝑎) – action 𝑎 taken at state ℎ.
Compute a posterior expected loss (risk) for each possible 𝑎: (where the 
data is defined as 𝑥)
𝜌 𝑎 𝑥 ≜ 𝔼𝑝 ℎ 𝑥 ℓ ℎ, 𝑎 = ෍
ℎ∈ℋ
ℓ ℎ, 𝑎 𝑝(ℎ|𝑥)
We thus define the optimal policy 𝜋∗ 𝑥 , which is also called the Bayes 
estimator or Bayes decision rule 𝛿∗(𝑥):
𝜋∗ 𝑥 = argmin
𝑎∈𝒜
𝜌 𝑎 𝑥 = argmin
𝑎∈𝒜
𝔼𝑝 ℎ 𝑥 ℓ ℎ, 𝑎

# Slide 47

© Copyright National University of Singapore. All Rights Reserved. 
Decision Theory: in Detail
The optimal policy 𝜋∗ 𝑥 , or Bayes decision rule 𝛿∗(𝑥):
𝜋∗ 𝑥 = argmin
𝑎∈𝒜
𝜌 𝑎 𝑥 = argmin
𝑎∈𝒜
𝔼𝑝 ℎ 𝑥 ℓ ℎ, 𝑎
If we define a utility function 𝑈(ℎ, 𝑎) as the desirability of each action at 
each state, then: 𝑈 ℎ, 𝑎 = −ℓ ℎ, 𝑎
The optimal policy can be re-written as:
𝜋∗ 𝑥 = argmax
𝑎∈𝒜
𝔼ℎ[𝑈(ℎ, 𝑎)]
This is defined as the maximum expected utility principle.
Next, we use Bayesian decision theory to decide the optimal class label to 
predict given an observed input 𝑥 ∈ 𝒳.

# Slide 48

© Copyright National University of Singapore. All Rights Reserved. 
Classification with Zero-One Loss
Set the following assumptions:
The states of nature correspond to class labels: ℋ = 𝒴 = {1, … , 𝐶};
The actions correspond to class labels: 𝒜 = 𝒴.
Under the above assumptions, a common loss function (and a very simple 
one) is the zero-one loss ℓ ℎ, 𝑎 = ℓ01(𝑦∗, ො𝑦):
Or mathematically ℓ01 𝑦∗, ො𝑦 = 𝕀(𝑦∗ = ො𝑦)

# Slide 49

© Copyright National University of Singapore. All Rights Reserved. 
Classification with Zero-One Loss
The posterior expected loss 𝜌 𝑎 𝑥 ≜ 𝔼𝑝 ℎ 𝑥 ℓ ℎ, 𝑎 :
𝜌 𝑎 𝑥 = 𝜌 ො𝑦 𝑥 = 𝑝 ො𝑦 ≠ 𝑦∗ 𝑥 = 1 − 𝑝 ො𝑦 = 𝑦∗ 𝑥
The action that minimizes the expected loss is to choose the most 
probable action:
𝜋 𝑥 = argmax
ො𝑦∈𝒴
𝑝( ො𝑦|𝑥)
This corresponds to the mode of the posterior distribution, also known as 
the maximum a posteriori or MAP estimate.

# Slide 50

© Copyright National University of Singapore. All Rights Reserved. 
Cost-Sensitive Classification
Problem of the zero-one loss?
Cost of false positive (ℓ01) = cost of false negative (ℓ10)
For cost-sensitive classification, the loss function would be:
ℓ00
ℓ10
ℓ01
ℓ11
Set 𝑝0 = 𝑝(𝑦∗ = 0|𝑥) and 𝑝1 = 1 − 𝑝0
We would choose label ො𝑦 = 0 iff ℓ00𝑝0 + ℓ10𝑝1 < ℓ11𝑝1 + ℓ01𝑝0.
Usually, ℓ00 = ℓ11 = 0. Thus, the above simplifies to: 𝑝1 <
ℓ01
ℓ01+ℓ10
Example (Try out): if a false negative costs twice as much as false positive, then we use a 
decision threshold of 1/3 before declaring a positive.

# Slide 51

© Copyright National University of Singapore. All Rights Reserved. 
Classification with Reject Option
We may be able to say “I don’t know” instead of returning an answer that 
we don’t really trust – choosing the reject option. 
Important in domains such as medicine where we may be risk adverse.
https://www.youtube.com/watch?v=P18EdAKuC1U

# Slide 52

© Copyright National University of Singapore. All Rights Reserved. 
Classification with Reject Option
Set the following assumptions:
The states of nature correspond to class labels: ℋ = 𝒴 = {1, … , 𝐶};
The actions correspond to class labels: 𝒜 = 𝒴 ∪ {0}.
The loss function is defined as:
Here 𝜆𝑟: cost of the reject action, and 𝜆𝑒: cost of a classification error. Set መ𝜆 = 1 −
𝜆𝑟/𝜆𝑒, we thus compute the optimal policy as:
Here ො𝑦 = argmax
𝑦
𝑝 𝑦 𝑥 and 𝑝∗ = 𝑝 𝑦∗ 𝑥 = max
𝑦
𝑝 𝑦 𝑥 = 𝑝( ො𝑦|𝑥).
𝑙 𝑦∗, 𝑎 = ቐ
0,  if 𝑦∗ = 𝑎 and 𝑎 ∈ {1, … , 𝐶}
𝜆𝑟,  if 𝑎 = 0
𝜆𝑒,  otherwise
𝑎∗ = ൝ ො𝑦,  if 𝑝∗ > መ𝜆
reject,  otherwise

# Slide 53

© Copyright National University of Singapore. All Rights Reserved. 
Class Confusion Matrices
We can choose the optimal label in a binary classification problem by 
thresholding the probability – hard threshold!
Instead of picking a single threshold, consider using a set of different thresholds, 
and comparing the resulting performance.
For a threshold 𝜏, the decision rule: ො𝑦𝜏 𝑥 = 𝕀(𝑝 𝑦 = 1 𝑥 > 1 − 𝜏).
𝐹𝑃𝜏 = ∑𝑛=1
𝑁 𝕀( ො𝑦𝑛 𝑥𝑛 = 1, 𝑦𝑛 = 0), similar for 𝐹𝑁𝜏, 𝑇𝑃𝜏, and 𝑇𝑁𝜏.
The simple form of evaluation: accuracy
Accuracy = 𝑇𝑃 +  𝑇𝑁
𝑇𝑃 +  𝐹𝑁 +  𝐹𝑃 +  𝑇𝑁
Is this reliable?
If have imbalanced testing data (class 0 = 90, 
class 1 = 10), if the model predict everything 
as class 0, Accuracy = 90%

# Slide 54

© Copyright National University of Singapore. All Rights Reserved. 
Class Confusion Matrices
True positive rate (TPR), or 
sensitivity, recall, hit rate: 
𝑇𝑃𝑅𝜏 = 𝑝 ො𝑦 = 1|𝑦 = 1, 𝜏 = 𝑇𝑃𝜏
𝑇𝑃𝜏 + 𝐹𝑁𝜏
False positive rate (FPR), or false 
alarm rate, type I error rate:
𝐹𝑃𝑅𝜏 = 𝑝 ො𝑦 = 1|𝑦 = 1, 𝜏 = 𝐹𝑃𝜏
𝐹𝑃𝜏 + 𝑇𝑁𝜏
Plot the TPR vs FPR as an implicit 
function of 𝜏: Receiver Operating 
Characteristic (ROC) curve – Plot 
TPR vs FPR with varying 𝜏.

# Slide 55

© Copyright National University of Singapore. All Rights Reserved. 
Class Confusion Matrices
True positive rate (TPR), or 
sensitivity, recall, hit rate: 
𝑇𝑃𝑅𝜏 = 𝑝 ො𝑦 = 1|𝑦 = 1, 𝜏 = 𝑇𝑃𝜏
𝑇𝑃𝜏 + 𝐹𝑁𝜏
False positive rate (FPR), or false 
alarm rate, type I error rate:
𝐹𝑃𝑅𝜏 = 𝑝 ො𝑦 = 1|𝑦 = 1, 𝜏 = 𝐹𝑃𝜏
𝐹𝑃𝜏 + 𝑇𝑁𝜏
Plot the TPR vs FPR as an implicit 
function of 𝜏: Receiver Operating 
Characteristic (ROC) curve – Plot 
TPR vs FPR with varying 𝜏.

# Slide 56

© Copyright National University of Singapore. All Rights Reserved. 
Class Confusion Matrices
True positive rate (TPR), or 
sensitivity, recall, hit rate: 
𝑇𝑃𝑅𝜏 = 𝑝 ො𝑦 = 1|𝑦 = 1, 𝜏 = 𝑇𝑃𝜏
𝑇𝑃𝜏 + 𝐹𝑁𝜏
False positive rate (FPR), or false 
alarm rate, type I error rate:
𝐹𝑃𝑅𝜏 = 𝑝 ො𝑦 = 1|𝑦 = 1, 𝜏 = 𝐹𝑃𝜏
𝐹𝑃𝜏 + 𝑇𝑁𝜏
Plot the TPR vs FPR as an implicit 
function of 𝜏: Receiver Operating 
Characteristic (ROC) curve – Plot 
TPR vs FPR with varying 𝜏.

# Slide 57

© Copyright National University of Singapore. All Rights Reserved. 
Plotting the ROC Curve: Example
ROC curve – Plot the TPR vs FPR as an implicit function of 𝜏, i.e., plot 
TPR vs FPR with varying 𝜏.
How do we compare different ROC curves?
plot roc curve visualization api

# Slide 58

© Copyright National University of Singapore. All Rights Reserved. 
The ROC Curve
Quality of a ROC curve is often summarized using the Area Under the 
Curve (AUC). Higher AUC scores better; max is 1. 
Another statistic is the Equal Error Rate 
(EER), or the cross-over rate
Defined as the value which satisfies 𝐹𝑃𝑅 = 𝐹𝑁𝑅 =
1 − 𝑇𝑃𝑅. 
Compute the EER by drawing a line from the top 
left to the bottom right and seeing where it 
intersects the ROC curve. 
Lower EER scores are better; the minimum is 
obviously 0 (top left corner).

# Slide 59

© Copyright National University of Singapore. All Rights Reserved. 
Class Imbalance and the ROC Curve
In some problems, there is severe class imbalance. 
E.g., in information retrieval, the set of negatives (irrelevant items) is usually much 
larger than the set of positives (relevant items).
Discuss: would the ROC curve be affected by class imbalance?
Discuss: how would class imbalance affect the effectiveness of ROC?

# Slide 60

© Copyright National University of Singapore. All Rights Reserved. 
The Precision-Recall Curve
Problems of the ROC curve: 
Require well-defined “true positives” and “true negatives”;
However, in some problems, “negative” may not be well-defined. 
E.g., detecting objects in images: if the detector works by classifying patches, then 
the # of patches examined — the # of true negatives — is a param of the algorithm, 
instead of the problem definition.
Can we evaluate the classification computed just from positives?
Precision and the precision-recall curve:
Precision: replace the FPR with a quantity that is computed just from positives.

# Slide 61

© Copyright National University of Singapore. All Rights Reserved. 
The Precision-Recall (PR) Curve
Precision: fraction of our detections are actually positive:
𝒫 𝜏 ≜ 𝑝 𝑦 = 1 ො𝑦 = 1, 𝜏 = 𝑇𝑃𝜏/(𝑇𝑃𝜏 + 𝐹𝑃𝜏)
Recall (TPR): fraction of the positives we actually detected:
ℛ 𝜏 ≜ 𝑝 ො𝑦 = 1 𝑦 = 1, 𝜏 = 𝑇𝑃𝜏/(𝑇𝑃𝜏 + 𝐹𝑁𝜏)

# Slide 62

© Copyright National University of Singapore. All Rights Reserved. 
An Example of PR Curve in Object Detection
Run object detector on all test images (with NMS).
For each category, compute Average Precision 
(AP) = area under Precision vs Recall Curve.
For each detection (highest score to lowest score):
If it matches some GT box with loU > 0.5, mark it as 
positive and eliminate the GT.
Otherwise mark it as negative.
Plot a point on PR Curve.
Average Precision (AP) = area under PR curve
Mean Average Precision (mAP) = average of AP 
for each category.
For "COCO mAP": Compute mAP @ thresh for 
each loU threshold (0.5, 0.55, 0.6, ..., 0.95) and 
take average.

# Slide 63

© Copyright National University of Singapore. All Rights Reserved. 
An Example of PR Curve in Object Detection
Run object detector on all test images (with NMS).
For each category, compute Average Precision 
(AP) = area under Precision vs Recall Curve.
For each detection (highest score to lowest score):
If it matches some GT box with loU > 0.5, mark it as 
positive and eliminate the GT.
Otherwise mark it as negative.
Plot a point on PR Curve.
Average Precision (AP) = area under PR curve
Mean Average Precision (mAP) = average of AP 
for each category.
For "COCO mAP": Compute mAP @ thresh for 
each loU threshold (0.5, 0.55, 0.6, ..., 0.95) and 
take average.

# Slide 64

© Copyright National University of Singapore. All Rights Reserved. 
Plotting the PR Curve: Example
PR curve – Plot the precision vs recall as an implicit function of 𝜏, i.e., plot 
precision vs recall with varying 𝜏.
How do we compare different PR curves?

# Slide 65

© Copyright National University of Singapore. All Rights Reserved. 
The PR Curve
The statistics to summarize the PR curve:
Precision at 𝐾 score: the precision for a fixed recall level;
AUC: the area under the PR curve;
Interpolated precision:
Possible precision does not drop monotonically with 
recall. E.g., a classifier has 90% precision at 10% recall, 
and 96% precision at 20% recall. 
Measure the maximum precision we can achieve with at 
least a recall of 10% (which would be 96%). 
Average precision: avg of interpolated precisions
Equal to the area under the interpolated PR curve. 
The mean average precision (mAP): the mean of the AP 
over a set of different PR curves.

# Slide 66

© Copyright National University of Singapore. All Rights Reserved. 
The PR Curve and F-scores
For a given point on the PR curve, can combine 
the precision value 𝒫 and the recall value ℛ into a 
single statistics:
1
𝐹𝛽
= 1
1 + 𝛽2
1
𝒫 + 𝛽2
1 + 𝛽2
1
ℛ
If we set 𝛽 = 1, then we get the harmonic mean of 
the precision and recall, known as the 𝐹1 score.
1
𝐹1
= 1
2
1
𝒫 + 1
𝑅
Questions: 
Why do we use the “harmonic mean” instead of the 
“arithmetic mean” (𝒫 + ℛ)/2?
Would the PR curve be affected by class imbalance?

# Slide 67

© Copyright National University of Singapore. All Rights Reserved. 
Probabilistic Prediction Problem
In previous slides: assumed the set of possible actions was to pick a single class 
label (or possibly the “reject” action).
More often: the set of possible actions is to pick a probability 
distribution over some value of interest:
Perform probabilistic prediction, rather than predicting a specific value.
For Classification: states of nature → class labels: ℋ = 𝒴 = {1, … , 𝐶}; actions → class 
labels: 𝒜 = 𝒴.
For Probabilistic prediction: states of nature → ℎ = 𝑝(𝑌|𝑥); action → 𝑎 =  𝑞 𝑌 𝑥 .
What we want: to pick 𝑞 to minimize𝔼 ℓ ℎ, 𝑎 = 𝔼[ℓ(𝑝, 𝑞)] for a given 𝑥.
The key: how to define the loss function ℓ(𝑝, 𝑞)?

# Slide 68

© Copyright National University of Singapore. All Rights Reserved. 
Bayesian Learning Fundamentals 2:
✓ KL-Divergence
✓ Entropy and Mutual Information

# Slide 69

© Copyright National University of Singapore. All Rights Reserved. 
KL Divergence (KL-Div)
Loss function is measured between two “distributions”.
Kullback Leibler divergence (KL divergence) is a measure of how one 
probability distribution is different from a second:
𝔻𝐾𝐿 𝑝 ∥ 𝑞 ≜ ෍
𝑦∗∈𝒴
𝑝 𝑦∗ log 𝑝 𝑦∗
𝑞 ො𝑦
Here 𝑝 is the true distribution and 𝑞 is the predicted distribution.
KL divergence can be further expanded as:
𝔻𝐾𝐿 𝑝 ∥ 𝑞 = ∑ 𝑝 𝑦∗ log 𝑝(𝑦∗) − ∑ 𝑝 𝑦∗ log 𝑞 ො𝑦 = −ℍ 𝑝 + ℍ𝑐𝑒(𝑝, 𝑞). ℍ(𝑝): 
entropy of 𝑝; ℍ𝑐𝑒: cross entropy between 𝑝 and 𝑞.

# Slide 70

© Copyright National University of Singapore. All Rights Reserved. 
Entropy
The key component of KL Divergence: entropy.
The entropy of a probability distribution can be interpreted as a measure of 
uncertainty, or lack of predictability, associated with a random variable drawn from a 
given distribution.
Entropy: define the information content of a data source.
Suppose we observe a sequence of symbols 𝑋𝑛~ 𝑝 generated from distribution 𝑝. 
What does it mean by 𝑝 having high / low entropy?
If 𝑝 has high entropy, hard to predict the value of each observation 𝑋𝑛. Hence dataset 
𝒟 = 𝑋1, … , 𝑋𝑛  has high information content. 
If 𝑝 is a degenerate distribution with 0 entropy, then every 𝑋𝑛 will be the same, so 𝒟 
does not contain much information.

# Slide 71

© Copyright National University of Singapore. All Rights Reserved. 
Entropy for Discrete Random Variables
More specifically, formulate the entropy of a discrete RV 𝑋 with distribution 
𝑝 over 𝐾 states (classes) as:
ℍ 𝑋 ≜ − ෍
𝑘
𝐾
𝑝 𝑋 = 𝑘 log2 𝑝 𝑋 = 𝑘 = −𝔼𝑋 log 𝑝 𝑋
Note that ℍ 𝑋 is equivalent to the notation of ℍ 𝑝 .
Question: when will the distribution obtain the Max / Min Entropy?

# Slide 72

© Copyright National University of Singapore. All Rights Reserved. 
Entropy for Discrete Random Variables
ℍ 𝑋 ≜ − ෍ 𝑝 𝑋 = 𝑘 log2 𝑝 𝑋 = 𝑘 = −𝔼𝑋 log 𝑝 𝑋
Discrete distribution with Maximum entropy is uniform distribution.
For a 𝐾-ary RV, the entropy is max-ed if 𝑝(𝑥 = 𝑘) = 1/𝐾; 𝐻 𝑋 = log2𝐾.
Distribution with min entropy is any delta-function that puts all its mass on one state
Special case: binary RVs 𝑋 ∈ 0,1 , 𝑝(𝑋 = 1) = 𝜃 and 
𝑝(𝑋 = 0) = 1 − 𝜃:
ℍ 𝑋 = − 𝑝 𝑋 = 1 log2 𝑝 𝑋 = 1 + 𝑝 𝑋 = 0 log2 𝑝 𝑋 = 0
ℍ 𝑋 = − 𝜃 log2 𝜃 + 1 − 𝜃 log2 1 − 𝜃

# Slide 73

© Copyright National University of Singapore. All Rights Reserved. 
Mutual Information
KL divergence: measure how dissimilar two distributions were. 
Then can we measure how dependent two RVs are? 
Turn the question of measuring the dependence of two RVs into a question about the 
similarity of their distributions. 
The notion of Mutual Information (MI) btw two RVs, defined as:
𝕀 𝑋; 𝑌 ≜ 𝔻𝐾𝐿 𝑝 𝑥, 𝑦 ∥ 𝑝 𝑥 𝑝 𝑦 = ෍
𝑦
෍
𝑥
𝑝 𝑥, 𝑦 log 𝑝 𝑥, 𝑦
𝑝 𝑥 𝑝(𝑦)
Note: the notation is defined as 𝕀 𝑋; 𝑌 as 𝑋 and/or 𝑌 can represent a set of variable – 
𝕀(𝑋; 𝑌, 𝑍) is the MI between 𝑋 and 𝑌, 𝑍 .
MI is always non-negative: 𝕀 𝑋; 𝑌 ≥ 0. Reach the bound of 0 iff 𝑝 𝑥, 𝑦 = 𝑝 𝑥 𝑝 𝑦 .

# Slide 74

© Copyright National University of Singapore. All Rights Reserved. 
Mutual Information: Interpretation
𝕀 𝑋; 𝑌 ≜ 𝔻𝐾𝐿 𝑝 𝑥, 𝑦 ∥ 𝑝 𝑥 𝑝 𝑦 = ෍
𝑦
෍
𝑥
𝑝 𝑥, 𝑦 log 𝑝 𝑥, 𝑦
𝑝 𝑥 𝑝(𝑦)
MI is a KL-Div between the joint and factored marginal distributions:
MI measures the information gain if we update from a model that treats 
the two variables as independent 𝑝(𝑥)𝑝(𝑦) to one that models their true 
joint density 𝑝(𝑥, 𝑦).
To gain further, re-express MI in terms of joint and conditional entropies:
𝕀 𝑋; 𝑌 = ℍ 𝑋 − ℍ 𝑋|𝑌 = ℍ 𝑌 − ℍ 𝑌|𝑋 , ℍ 𝑌|𝑋 = ℍ 𝑋, 𝑌 − 𝐻(𝑋)
MI between 𝑋 and 𝑌 is thus expressed as the reduction in uncertainty 
about 𝑋 after observing 𝑌, or, by symmetry, the reduction in uncertainty 
about 𝑌 after observing 𝑋.

# Slide 75

© Copyright National University of Singapore. All Rights Reserved. 
Mutual Information: Interpretation
𝕀 𝑋; 𝑌 ≜ 𝔻𝐾𝐿 𝑝 𝑥, 𝑦 ∥ 𝑝 𝑥 𝑝 𝑦 = ෍
𝑦
෍
𝑥
𝑝 𝑥, 𝑦 log 𝑝 𝑥, 𝑦
𝑝 𝑥 𝑝(𝑦)
𝕀 𝑋; 𝑌 = ℍ 𝑋 − ℍ 𝑋|𝑌 = ℍ 𝑌 − ℍ 𝑌|𝑋 , ℍ 𝑌|𝑋 = ℍ 𝑋, 𝑌 − 𝐻(𝑋)
An alternative proof that conditioning, on average, reduces entropy!
𝕀 𝑋; 𝑌 = ℍ 𝑋 − ℍ 𝑋|𝑌 ≥ 0 → ℍ 𝑋|𝑌 ≤ ℍ 𝑋
One can also obtain the following equivalent results of MI:
𝕀 𝑋; 𝑌 = ℍ 𝑋, 𝑌 − ℍ 𝑋|𝑌 − ℍ 𝑌|𝑋
𝕀 𝑋; 𝑌 = ℍ 𝑋 + ℍ 𝑌 − ℍ 𝑋, 𝑌

# Slide 76

© Copyright National University of Singapore. All Rights Reserved. 
Mutual Information: An Example
Suppose two RVs 𝑋 and 𝑌 are related to the input integer 𝑛 ∈ 1, … , 8 . 
Define 𝑋 𝑛 = 1 if 𝑛 is even and 𝑌 𝑛 = 1 if 𝑛 is a prime number. Thus:
ℍ 𝑋, 𝑌 = −
1
8 log
1
8 +
3
8 log
3
8 +
3
8 log
3
8 +
1
8 log
1
8 = 1.81 𝑏𝑖𝑡𝑠
ℍ 𝑋 = ℍ 𝑌 = 1
ℍ 𝑌 𝑋 ≜ − ∑ 𝑝 𝑋, 𝑌 log 𝑝 𝑌 𝑋 = −
1
8 log
1
4 +
3
8 log
3
4 +
3
8 log
3
4 +
1
8 log
1
4 = 0.81
Verify that ℍ 𝑌|𝑋 = ℍ 𝑋, 𝑌 − 𝐻 𝑋 = 0.81
𝕀 𝑋; 𝑌 = ℍ 𝑋 − ℍ 𝑋|𝑌 = 1 − 0.81 = 0.19 𝑏𝑖𝑡𝑠

# Slide 77

© Copyright National University of Singapore. All Rights Reserved. 
Entropy and Mutual Information

# Slide 78

© Copyright National University of Singapore. All Rights Reserved. 
Bayesian Learning Fundamentals 3:
✓ The Frequentist Decision
✓ Application in Supervised Learning

# Slide 79

© Copyright National University of Singapore. All Rights Reserved. 
Recall: Frequentist vs. Bayesian 
Frequentist interpretation: probabilities represent long run frequencies of 
events that can happen multiple times.
E.g., Flipping the coin many times, we expect it to land heads about half 
the time.
Bayesian interpretation: quantify uncertainty or ignorance about 
something
E.g., The coin is equally likely to land heads/tails on next toss.

# Slide 80

© Copyright National University of Singapore. All Rights Reserved. 
Frequentist Decision Theory
In the Frequentist decision theory, treat the unknown state of nature 
(denoted by 𝜽 instead of ℎ) as a fixed yet unknown quantity, and treat 
the data 𝑥 as random.
Instead of conditioning over 𝑥, we average over it to compute the loss expect to incur 
if apply our estimator to different datasets.
Define the frequentist risk of an estimator 𝛿 given an unknown state of 
nature 𝜽 as the expected loss when applying 𝛿 to data 𝑥, where the 
expectation is over the data, sampled from 𝑝(𝑥|𝜽):
𝑅 𝜽, 𝛿 ≜ 𝔼𝑝 𝑥 𝜽 ℓ 𝜽, 𝛿 𝑥

# Slide 81

© Copyright National University of Singapore. All Rights Reserved. 
The Bayes Risk
𝑅 𝜽, 𝛿 ≜ 𝔼𝑝 𝑥 𝜽 ℓ 𝜃, 𝛿 𝑥
In general, the true state of nature 𝜽 that generates the data 𝑥 is unknown, 
so cannot compute the risk directly. Then how?
One solution: to assume a prior 𝜋0 for 𝜃 then average it out – the Bayes 
Risk (integrated risk):
𝑅𝜋0 𝛿 ≜ 𝔼𝜋0 𝜽 𝑅 𝜽, 𝛿 = න 𝑑𝜽𝑑𝑥𝜋0 𝜽 𝑝 𝑥 𝜽 ℓ 𝜽, 𝛿 𝑥
Decision rule that minimize the Bayes risk: Bayes estimator.

# Slide 82

© Copyright National University of Singapore. All Rights Reserved. 
The Bayes Estimator
Recall: in the Bayesian decision theory, we define the optimal policy, 
also the Bayes estimator or Bayes decision rule as:
𝜋∗ 𝑥 = argmin
𝑎∈𝒜
𝜌 𝑎 𝑥 = argmin
𝑎∈𝒜
𝔼𝑝 ℎ 𝑥 ℓ ℎ, 𝑎
In the Frequentist decision theory, the optimal estimator is:
𝛿 𝑥 = argmin
𝑎
න 𝑑𝜃𝜋0 𝜃 𝑝 𝑥 𝜃 ℓ(𝜃, 𝑎) ≡ argmin
𝑎
න 𝑑𝜃𝑝 𝜃 𝑥 ℓ(𝜃, 𝑎)
The two forms are actually equivalent! What does this mean?
Bayesian approach provides a good way of achieving the frequentist goal – picking 
the optimal action case-by-case is optimal on average!

# Slide 83

© Copyright National University of Singapore. All Rights Reserved. 
The Maximum Risk
Use of a prior might seem undesirable in certain cases. We can therefore 
define the maximum risk as:
𝑅max 𝛿 ≜ sup
𝜽
𝑅(𝜽, 𝛿)
Decision rule that min the max risk: minimax estimator 𝛿𝑀𝑀.
One example on the right:
𝛿1 has lower worst-case risk than 𝛿2.
𝛿1 is the minimax estimator (even if 𝛿2
has lower risk for most values of 𝜽)/
The minimax estimators are overly 
conservative!

# Slide 84

© Copyright National University of Singapore. All Rights Reserved. 
ERM and Frequentist Decision Theory
We consider how to apply the frequentist decision theory in the context of 
supervised learning.
The risk: 𝑅 𝜽, 𝛿 ≜ 𝔼𝑝 𝑥 𝜽 ℓ 𝜽, 𝛿 𝑥 . 
Unknown “state of nature” can correspond to the unknown parameters of some 
model 𝜽∗.
Meanwhile, the discussion w.r.t. data within a dataset 𝑥 ∈ 𝒟, the risk can be rewritten:
𝑅 𝛿, 𝜽∗ = 𝔼𝑝(𝒟|𝜃∗) ℓ 𝜽∗, 𝛿 𝒟

# Slide 85

© Copyright National University of Singapore. All Rights Reserved. 
ERM and Frequentist Decision Theory
In supervised learning, 
Different unknown state of nature (output 𝑦) for each different input 𝑥, 
Estimator 𝛿 is a prediction function ො𝑦 =  𝑓(𝑥), 
The state of nature (as a whole) is the true distribution 𝑝∗(𝑥, 𝑦∗),
Thus, the risk of an estimator is (in terms of 𝑓 and 𝑝∗) is:
𝑅 𝑓, 𝑝∗ = 𝑅 𝑓 ≜ 𝔼𝑝∗ 𝑥 𝑝∗ 𝑦 𝑥 [ℓ 𝑦∗, 𝑓 𝑥 )
Defined as the population risk, as the expectations are taken w.r.t. the true joint 
distribution 𝑝∗(𝑥, 𝑦∗).
Can we compute the population risk? If not how?

# Slide 86

© Copyright National University of Singapore. All Rights Reserved. 
ERM and Frequentist Decision Theory
In most cases, quite impossible to obtain the unknown true distribution 𝑝∗.
Instead, can approximate it with the empirical distribution of 𝑁 samples:
𝑝𝒟 𝑥, 𝑦 𝒟 ≜
1
𝒟 ∑ 𝑥𝑛,𝑦𝑛 ∈𝒟 𝛿 𝑥 − 𝑥𝑛 𝛿 𝑦 − 𝑦𝑛
Here the empirical distribution is exactly the training distribution 𝑝𝒟 𝑥, 𝑦 𝒟 = 𝑝𝑡𝑟(𝑥, 𝑦). We 
thus obtain the empirical risk (which is a random variable) as: 
𝑅 𝑓, 𝒟 ≜ 𝔼𝑝𝒟 𝑥, 𝑦 𝒟 [ℓ 𝑦, 𝑓 𝑥 = 1
𝑁 ෍
𝑛=1
𝑁
ℓ 𝑦𝑛, 𝑓 𝑥𝑛
A natural way to choose the predictor is to choose the predictor with minimum risk:
Ƹ𝑓𝐸𝑅𝑀 = argmin
𝑓
𝑅(𝑓, 𝒟) = argmin
𝑓
1
𝑁 ∑𝑛𝑁 ℓ(𝑦𝑛, 𝑓(𝑥𝑛)) = argmin
𝜃
ℒ(𝜃)
ℒ 𝜽 = ∑𝑛=1
𝑁 ℓ(𝑦𝑛, 𝜽; 𝑥𝑛) (as defined in Lecture 1)
The process of choosing predictor መ𝑓𝐸𝑅𝑀 is empirical risk minimization (ERM).

# Slide 87

© Copyright National University of Singapore. All Rights Reserved. 
Performance of Functions Fitted with ERM
What is the theoretical performance of functions that are fitted with the 
ERM principle?
Let 𝑓∗∗ = argmin
𝑓
𝑅 𝑓  be the function that achieves the min possible population risk, 
where we optimize over ALL possible functions.
!! ALL possible functions?
Define 𝑓∗ = argmin
𝑓∈ℋ
𝑅(𝑓) the best function in hypothesis space ℋ.
We still cannot compute what 𝑓∗ is – the population risk 𝑅 is unknown!
We can only compute by further approx. – minimizing the empirical risk:
𝑓𝑁
∗ = argmin
𝑓∈ℋ
𝑅 𝑓, 𝒟 = argmin
𝑓∈ℋ
𝔼𝑝𝑡𝑟 ℓ 𝑦, 𝑓 𝑥

# Slide 88

© Copyright National University of Singapore. All Rights Reserved. 
The risk of our chosen predictor 𝑓𝑁
∗ compared to the best possible 
predictor 𝑓∗∗ can be decomposed into two terms:
𝔼𝑝∗ 𝑅 𝑓𝑁
∗ − 𝑅 𝑓∗∗ = 𝑅 𝑓∗ − 𝑅 𝑓∗∗ + 𝔼𝑝 𝑅 𝑓𝑁
∗ − 𝑅 𝑓∗
𝜀𝑎𝑝𝑝(ℋ) is the approximate error: measures how closely ℋ can model the true 
optimal function 𝑓∗∗.
𝜀𝑒𝑠𝑡(ℋ, 𝑁) is the estimation error or generalization error: measures the difference in 
estimated risk due to having a finite training set.
Approx. 𝜀𝑒𝑠𝑡(ℋ, 𝑁) by the difference btw training set and test set error: 𝜀𝑒𝑠𝑡 ℋ, 𝑁 ≈
𝔼𝑝𝑡𝑟 ℓ 𝑦, 𝑓𝑁
∗ 𝑥 − 𝔼𝑝𝑡𝑒 ℓ 𝑦, 𝑓𝑁
∗ 𝑥  – Generalization Gap.
Performance of Functions Fitted with ERM
𝜀𝑎𝑝𝑝(ℋ) 𝜀𝑒𝑠𝑡(ℋ, 𝑁)

# Slide 89

© Copyright National University of Singapore. All Rights Reserved. 
Regularization and Regularized Risk
To avoid the chance of overfitting, it is common to add a complexity 
penalty to the objective function – regularization.
With regularization, we obtain the regularized empirical risk:
𝑅𝜆 𝑓, 𝒟 = 𝑅 𝑓, 𝐷 + 𝜆𝐶(𝑓)
𝐶(𝑓) measures the complexity of the prediction function 𝑓 and 𝜆 ≥ 0 is the 
hyperparameter that controls the strength of complexity penalty.
In practice, the regularization is often applied to the model parameters directly, thus 
the regularized ER is rewritten: 𝑅𝜆 𝜃, 𝒟 = 𝑅 𝜃, 𝐷 + 𝜆𝐶(𝜃).
If the loss function is a log loss term, while the regularizer is a negative log prior, 
𝑅𝜆 𝜃, 𝒟 = −
1
𝑁 ∑𝑛=1
𝑁 log 𝑝 𝑦𝑛 𝑥𝑛, 𝜃 − 𝜆 log 𝑝 𝜃 . Minimizing this is exactly equivalent to 
the MAP estimation.

# Slide 90

© Copyright National University of Singapore. All Rights Reserved. 
Regularization and Regularized Risk
To avoid the chance of overfitting, it is common to add a complexity 
penalty to the objective function – regularization.
Recall when we first introduce the MAP estimation:
The action that minimizes the expected loss is to choose the most 
probable action:
𝜋 𝑥 = argmax
ො𝑦∈𝒴
𝑝( ො𝑦|𝑥)
This corresponds to the mode of the posterior distribution, also known as 
the maximum a posteriori or MAP estimate.

# Slide 91

© Copyright National University of Singapore. All Rights Reserved. 
Occam’s Razor
Occam's razor is the problem-solving principle that recommends searching 
for explanations constructed with the smallest possible set of elements. 
Also known as the principle of parsimony or the law of parsimony.
Consider 2 models, a simple one, 𝑚1, and a more complex one, 𝑚2. Suppose both can 
explain the data by suitably optimizing their parameters, i.e., 𝑝(𝒟|෡𝜽1, 𝑚1) and 
𝑝(𝒟|෡𝜽2, 𝑚2) are both large. 
Intuitively should prefer 𝑚1, as it is simpler and as good as 𝑚2.
In regularization, it corresponds to the complexity penalty, where more 
complex models would result in a larger regularization term.

# Slide 92

© Copyright National University of Singapore. All Rights Reserved. 
Occam’s Razor

# Slide 93

© Copyright National University of Singapore. All Rights Reserved. 
Population Risk Estimation with Validation
A simple way to estimate the population risk for a supervised learning 
setup: cross-validation.
Partition the dataset into two, one used for training models (training set), another, the 
validation set or holdout set, used for assessing the risk.
Note that in supervised learning, the real “test data” are inaccessible.
Process: fit the model on the training set, use its performance on the 
validation set as an approx. to the population risk:
Make the dependence of the 𝑅𝜆(𝑓, 𝒟) on the dataset more explicit:
𝑅𝜆 𝜽, 𝒟 = 1
𝐷 ෍
𝑥,𝑦 ∈𝒟
ℓ 𝑦, 𝑓 𝑥; 𝜽 + 𝜆𝐶(𝜽)
Also define ෡𝜽𝜆 𝒟 = argmin
𝜃
𝑅𝜆 𝜃, 𝒟 , 𝒟𝑡𝑟 and 𝒟𝑣𝑎𝑙 are partitions.

# Slide 94

© Copyright National University of Singapore. All Rights Reserved. 
Cross-Validation
For each model, we fit it to the training set to obtain ෡𝜽𝜆 𝒟𝑡𝑟 . Then use the 
unregularized empirical risk on the validation set as an estimate of the 
population risk – validation risk: 𝑅𝜆
𝑣𝑎𝑙 ≜ 𝑅0 ෡𝜽𝜆 𝒟𝑡𝑟 , 𝒟𝑣𝑎𝑙 .
Any shortcomings of this method?
Consider the training data is very limited – not enough data to train on! 
Impossible to make reliable estimate of future performance.
Solution?

# Slide 95

© Copyright National University of Singapore. All Rights Reserved. 
Cross-Validation
Consider the training data is very limited – not enough data to train on! 
Impossible to make reliable estimate of future performance.
Solution?
Cross-validation!

# Slide 96

© Copyright National University of Singapore. All Rights Reserved. 
Cross-Validation
Split the training data into 𝐾 folds;
For each fold 𝑘 ∈ 1, … , 𝐾 , train on 
all but ONE fold, in a round-robin 
fashion (in turns, circled around all 
possible folds).
Formally, the cross-validation risk is formulated as:
𝑅𝜆
𝑐𝑣 ≜
1
𝐾 ∑𝑘=1
𝐾 𝑅0 ෡𝜽𝜆 𝒟−𝑘 , 𝒟𝑘
𝒟𝑘 is the data in the 𝑘-th fold, and 𝒟−𝑘 is all the other data. If set 𝐾 = 𝑁, 
the method is known as leave-one-out cross-validation.
Alternatively, we can estimate an optimal መ𝜆 with CV, then re-estimate the 
model parameters using all data: ෡𝜽 = argmin
𝜽
𝑅෡𝜆(𝜽, 𝒟).

# Slide 97

© Copyright National University of Singapore. All Rights Reserved. 
Cross-Validation: Example

# Slide 98

© Copyright National University of Singapore. All Rights Reserved. 
Summary
Fundamentals: Probability and the difference between distributions:
Probability and the Bayes’ Rule.
All sorts of univariate probability distributions.
KL-Divergence, Entropy, and Mutual Information.
Bayesian Learning and Modelling:
Modelling with known distribution type: MLE.
Empirical Risk Minimization (ERM).
Decision making for best prediction: classification problems and MAP.
The Bayesian and Frequentist Decision Theories.
Regularized risk with cross-validation.
