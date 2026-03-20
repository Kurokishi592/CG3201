# Page 1

Project 2: Bayesian Classification and Regression for Data Analysis. 
This continuous assessment is a project-style exercise on Bayesian Modeling, where 
you will experience how we leverage Bayesian modeling for both classification tasks 
and regression tasks. You will work on the cougar-face vs cougar-body classes for the 
classification task and the Computer Hardware (CPU Performance) dataset for the 
linear regression task.  
You are required to implement the learning algorithms from scratch (except where 
explicitly stated), and provide a comparative analysis supported by quantitative and 
qualitative evidence. 
Before you start the assignment, please read through the following guidelines so that 
you understand the requirements. Follow the guidelines strictly to avoid unnecessary 
mark deductions: 
1. Implement your codes with  Python (recommended) or MATLAB (or any other 
languages you deem comfortable with). 
2. For questions that require you to show your code, make sure your codes are clean 
and easily readable  (add meaningful comments, comments are markable). 
Embed your code into your answer with screenshots. 
3. Your submission should be one single PDF file with the necessary code and 
results collated in a single file . Incorrect submission format will result in a direct 
10 points (out of 100) deduction. 
4. Do submit your project on Canvas before the deadline: 23:59 (SGT), 20 Mar 2026. 
There will be 10 submissions attempts restricted for every student. 
5. Policy on late submission: the deadline is a strict one, so please prepare and plan 
early and carefully. Any late submission will be deducted 10 points (out of 100) for 
every 24 hours. 
6. This is an individual project, do NOT share your solutions with others, we have 
zero tolerance for plagiarism. 
7. If you made use of any AI tools for your project (which I strongly recommend), 
please include an AI Usage Statement that states the AI tool used and how you 
have leveraged the AI tool.


# Page 2

Part 1: Bayesian Modeling for Image Classification with MLE and MAP (50%). 
a. Data loading and feature construction (10%) 
You will be provided with the "caltech_cougar.zip" dataset. For this project, convert all 
images to grayscale and resize them to a fixed resolution of 64 × 64 pixels. Each 
image must be flattened into a 1D feature vector 𝐱 ∈ ℝ𝐷, where 𝐷 = 4096. In your 
report, provide basic dataset statistics, including sample counts per class for your 
chosen train/test split and a visualization of the "Mean Image" for each class. Your train 
versus test ratio should not exceed 75:25 (i.e., you cannot have more than 75% image 
for each class as your training image). 
 
b. Bayesian Modeling with Varying Distribution Assumptions (MLE) (20%) 
You will now implement two different models from scratch to compare how different 
distribution assumptions affect classification performance. First, implement a 
Gaussian Naive Bayes classifier by estimating the mean vector 𝝁𝑪 and the variance 
vector 𝝈𝑪
𝟐 for EACH class using the MLE formulas for a Gaussian distribution (show 
the derivation of the MLE formulas). 
Second, re -evaluate the task by assuming each pixel intensity follows a Laplace 
Distribution, which is often more robust to outliers than the Gaussian distribution. You 
must estimate the location parameter 𝝁 and the scale parameter 𝒃 for each class 
using their respective MLE formulas: 
𝝁̂𝑳𝒂𝒑𝒍𝒂𝒄𝒆 = median(𝒙𝒊) and 𝒃̂ =  1
𝑁 ∑|𝒙𝒊  −  𝝁̂𝑳𝒂𝒑𝒍𝒂𝒄𝒆|
𝑁
𝑖=1
 
Evaluate both models on your test set, report the accuracy and confusion matrices, 
and discuss which distribution assumption (Gaussian vs. Laplace) provides better 
generalization for this specific image dataset. 
 
c. Maximum A Posteriori (MAP) with Unique Prior Analysis (20%) 
You will now incorporate Prior Knowledge into your Gaussian model using a unique 
Prior Mean 𝝁𝟎 and Prior Variance 𝝈𝟎
𝟐 vector based on your Student ID. Let 𝑆 be 
the sequence of digits in your ID. Your prior mean 𝝁𝟎  for all pixels should be the 
average value of 𝑆 (scaled to the [0, 255] range), and your prior variance 𝝈𝟎
𝟐 should 
be a vector where each element 𝑖 is calculated as: 
𝝈𝟎,𝒊
𝟐 = 𝑉𝑎𝑟(𝑆)× (1 + 𝑖 (mod MaxDigit)
10 ) 
For example, if your ID digits average to 4.5, then 𝜇0,𝑖 =
4.5
9 × 255 ≈ 127.5 for all 𝑖. 
Implement the MAP estimation for the Gaussian:


# Page 3

𝝁𝑴𝑨𝑷 = 𝝈𝟐𝝁𝟎 + 𝑛𝝈𝟎
𝟐𝒙̅
𝝈𝟐 + 𝑛𝝈𝟎
𝟐  and 𝝈𝑴𝑨𝑷
𝟐 = 𝝈𝟐𝝈𝟎
𝟐
𝝈𝟐 + 𝑛𝝈𝟎
𝟐 
For simplicity, use the MLE variance obtained from Question b as the value for 𝝈𝟐. 
Compare the resulting test performance against your MLE-based Gaussian classifier. 
In your report, analyze the relationship between 𝑛 (sample size) and the posterior 
variance 𝝈𝑴𝑨𝑷
𝟐  . Explain why the model’s "certainty" increases as more data is 
observed and why MAP provides more robust parameter estimates than MLE when 
training samples are extremely limited.


# Page 4

Part 2: Bayesian Linear Regression & Uncertainty Estimation (50%). 
a. Implementation of MLE and MAP Regression (25%) 
You will use the "machine.data" (Computer Hardware) dataset  (provided in the 
“computer+hardware.zip” zipfile to perform a regression task predicting the "Estimated 
Relative Performance" of various hardware configurations. Firstly, you are to 
implement the MLE for linear regression  𝑌 = 𝜃𝑋 from scratch using the closed -form 
matrix solution: 
𝜃𝑀𝐿𝐸 =  (𝑋𝑇𝑋)−1𝑋𝑇𝑌. 
Try to include a formal derivation demonstrating how this MLE solution is equivalent to 
the Ordinary Least Squares (OLS) criterion, specifically showing how the optimization 
of the Gaussian log-likelihood leads to the minimization of squared errors. 
Secondly, you will develop a MAP estimation model by assuming a Gaussian prior 
on the weights 𝜃 ∼ 𝒩(0, 𝜆−1𝐼). The MAP solution is given by: 
𝜃𝑀𝐴𝑃 = (𝑋𝑇𝑋 + 𝜆𝐼)−1𝑋𝑇𝑌. 
To ensure unique results, your regularization parameter 𝜆  would be calculated by 
taking the sum of the digits in your Student ID and dividing it by 100 (for example, if 
your digits sum to 30, then 𝜆 = 0.30).  
For both the MLE and MAP models, you would compute and report the Mean Squared 
Error (MSE) on the test set. You should explain how the addition of the 𝜆𝐼 term affects 
the stability of the matrix inversion, particularly when dealing with potentially ill -
conditioned data. 
 
b. Scikit-Learn Benchmarking and Comparative Analysis (25%) 
Now, you would benchmark your "from-scratch" implementations against the standard 
Scikit-Learn library. You will train a “LinearRegression” model and a Ridge regression 
model (setting the 𝛼 parameter to your unique 𝜆 value) with “sklearn.linear_model”. 
You would validate your work by comparing the final weight vectors 𝜃  and the 
resulting test MSE between your implementations and the library’s output. Explain the 
discrepancies between your model and the Scikit-Learn results in detail. 
Further, you will perform a qualitative analysis of the model weights by plotting the 
magnitudes of the individual coefficients for both the MLE and MAP models. Explain 
why the MAP weights are generally smaller in magnitude than the MLE weights and 
how this "shrinkage" effect serves as a form of regularization. Further, analyze which 
model is more likely to achieve better generalization on unseen hardware data, 
grounding your conclusion in the Bias-Variance Tradeoff concepts and explain how 
the prior distribution helps mitigate the risk of overfitting.
