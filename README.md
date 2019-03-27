- Created By: **Prasannjeet Singh** (*ps222vt@student.lnu.se*)
- Originaly done as an assignment. The problem statements can be found [here](https://github.com/prasannjeet/image-recognition/blob/master/assignment3.pdf).
- Note that there are a total of 5 exercises solved in this project.

Exercise 5 - Hand-Written Digit Recognition
===========================================

Contents
--------

<div>

-   [Converting the imported variables into desirable format:](#1)
-   [Training a Linear SVM using a Subset](#2)
-   [Finding an optimal model](#6)
-   [The Confusion Matrix](#12)

</div>

Converting the imported variables into desirable format: {#1}
--------------------------------------------------------

As suggested in the assignment **cell2mat()** and **reshape()** is used
to rearrange the given data so that **fitcecoc()** can be applied. I
have also used **cellfun()** that allows us to apply a single function
to all the items in a cell, which makes it easier to transform the data
without using any for loops.

**Normalizing**

For normalizing the data between 0 and 1, the following formula was
used:

![\$\$z\_i = \\frac{x\_i - min(x)}{max(x) -
min(x)}\$\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq14804764601566062396.png)

Where ![\$x\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq12428413953531653171.png) is the vector, and
![\$z\_i\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq10694150929292851306.png) is the normalized
value for the element ![\$x\_i\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq08383031602117067423.png).
This can easily be obtained by using the function **mat2gray(x)**.

Also, if we treat this dataset like any other dataset, we can observe
that each row contains an individual set, and each column contains a
unique feature. Therefore, all these columns will be separately
considered and separately normalized.

``` {.codeinput}
% Loading and preprocessing the data
load mnist.mat;
img = cell2mat(cellfun(@(x) reshape(x,[28*28,1]),img,'UniformOutput',false))';
img_test = cell2mat(cellfun(@(x) reshape(x,[28*28,1]),img_test,'UniformOutput',false))';
% Normalizing each column:
img_test = cell2mat(cellfun(@(x) mat2gray(x), num2cell(img_test,1),'UniformOutput',false));
img = cell2mat(cellfun(@(x) mat2gray(x), num2cell(img,1),'UniformOutput',false));
```

Training a Linear SVM using a Subset {#2}
------------------------------------

For this section I\'ll use the first 1000 items for training and the
last 500 for validating the model. The method **templateLinear()**
creates a default linear-classification-model template, with Support
Vector Machines (SVM) as linear classification model type. This model
template is then passed as a parameter to the **fitcecoc()** function.

``` {.codeinput}
X = img(1:1000,:);
y = labels(1:1000,:);
Xt = img(end-499:end,:);
yt = labels(end-499:end,:);
t = templateLinear();
rng(1); % So that the answer remains same in every execution
mdl = fitcecoc(X,y,'Learners',t);
allSolution = predict(mdl,Xt);
idx = allSolution ~= yt;
totalError = sum(idx)
```

``` {.codeoutput}
totalError =

    35
```

As we can see, out of 500 test data, 35 were wrongly classified, which
is a 93% accuracy. We can look at first 9 of those 35 items:

``` {.codeinput}
X = Xt(idx,:);
y = yt(idx); allSolution = allSolution(idx);
luckyIdx = 1:9;
printX = X([luckyIdx],:);
correctAnswer = vec2mat(y([luckyIdx]),3)
calculatedAnswer = vec2mat(allSolution([luckyIdx]),3)
hFig = figure(2);
for i = 1:9
    subplot(3,3,i);
    imagesc(vec2mat(printX(i,:),28)');
end
snapnow;
close(hFig);
```

``` {.codeoutput}
correctAnswer =

     9     3    10
     4     2    10
     3     8     5


calculatedAnswer =

     4     8     5
     9     8     7
     8     1     9
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_01.png)

As we can observe from the observation above, although the accuracy of
the model appeared to be 93 percent, even simple handwritings were
miscalculated by the above model. This means we should improve the above
model.

We can also see the digits this method had most trouble predicting
correctly:

``` {.codeinput}
tabulate(y)
```

``` {.codeoutput}
  Value    Count   Percent
      1        3      8.57%
      2        4     11.43%
      3        5     14.29%
      4        5     14.29%
      5        5     14.29%
      6        1      2.86%
      7        0      0.00%
      8        2      5.71%
      9        5     14.29%
     10        5     14.29%
```

As per the above observation, digits 3,4,5,9 and 0 were the hardest to
calculate, with a total of 5 errors each.

Finding an optimal model {#6}
------------------------

As we know there are many Kernel Functions that can be used, but I have
used **rbf** for in this assignment, as suggested in Slide 26 of Lecture
7. After I chose the RBF Radial Basis Function for this model, I was
primarily left with to parameters to tune, which was **BoxConstraint**
and **KernelScale**. My interpretation of the parameters:

<div>

-   BoxConstraint (C): As we know that C controls the influence of each
    individual test item (penalty factor). A low value of C basically
    tries to ignore the \'outlier\', or the \'noise\' in the test items.
    Which is good for our model, because we would want to avoid all
    those numbers which were wrongly written by people. For example
    these two images:

</div>

``` {.codeinput}
[I,map] = imread('Data\badHandWriting.PNG',https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/'png');
imshow(I,map);
snapnow;
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_02.png)

Here, the first image is 2, and the second is 6. These are clearly some
outliers, and a lower value of C will give less weightage to these types
of data sets.

<div>

-   KernelScale (Gamma): We know that Gaussian Rdial Basis Function is:
    ![\$K\\left(x\_i,x\_j\\right) = exp\\left(- \\gamma\|\|x\_i -
    x\_j\|\|\^2\\right), \\gamma \>
    0\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq14973774818574534740.png). Since
    ![\$\\gamma\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq17096441642737911057.png) controls the
    similarity, and if
    ![\$\\gamma\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq17096441642737911057.png) is (let\'s say)
    0, all the points will be treated as same. Likewise, if
    ![\$\\gamma\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq17096441642737911057.png) is set to
    ![\$\\infty\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq16739024125614530807.png), K(x,y) will be
    1 if and only if x and y are identical, otherwise, it will be zero,
    i.e. the model will predict correctly only when the exact input (as
    in the training set) is provided. In other words, high
    ![\$\\gamma\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq17096441642737911057.png) may result in
    low variance and vice versa. Considering this, we can say that
    ![\$\\gamma\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq17096441642737911057.png) should be more,
    so that we can create a distinction between all the classes.
    Moreover, it should also be more because sometimes digits such as 9
    and 4 look familiar in natural handwritings of different people.
    Such confusions can be there for many other pairs as well,
    therefore, it is important to have a comparatively higher value of
    ![\$\\gamma\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq17096441642737911057.png) so that a
    distinction can be maintained between them. However, it shouldn\'t
    be too high so that even a slight change in handwriting results in
    misclassification.

</div>

Although I know that k value should be relatively higher, and C value
should be relatively low; I cannot predict any values without testing.
Therefore, in this assignment, I have tried to quantitatively calculate
the hyperparameters by testing many values in a subset.

Here, I will describe the steps taken by me to obtain the best possible
hyperparameters. However, the code to obtain the same is saved in a
separate m file **TrainFinal.m** which I have also included in the
submission. Moreover, the results that I obtain from the training are
also saved in Data/trainingResult.mat (if required).

<div>

1.  In the first step, first 100 items of training set were chosen for
    training, and last 100 items of the same training set were chosen
    for validation. No dataset for testing was chosen, as I did the
    testing after I got the best possible pair.
2.  Now all possible k and C combinations ranging from 1 to 100 were
    tested by using two nested for loops (10,000 iterations) in the
    subset chosen above, and total number of errors for each iteration
    was recorded in a **k** x **C** matrix (result). The lowest error
    that I found for 100 validation set was 20.
3.  Further, all the (k,C) pairs were extracted from the above matrix,
    which gave the least cost, and these (k,C) pairs were used to train
    a larger set of datasets (900) and a new validation set was also
    used.
4.  This time, only one pair of (k,C) gave the least error (52 errors in
    a validation set of 900), and the (k,C) pair was **(6,1)**. This
    value follows the initial hypothesis, where I mentioned that k
    should be relatively larger and C should be low.

</div>

After getting the optimal pair, I trained the model for the whole 60,000
datasets and tested it for the whole 10,000 dataset by the following
code (The lines are commented as they take long to execute. However, I
have already performed them and will load them below for observations):

``` {.codeinput}
% params = templateSVM('KernelFunction','rbf', 'KernelScale',6,'BoxConstraint',1);
% myModel = fitcecoc(img,labels,'Learners', params, 'Coding', 'onevsall');
% outputLabel = predict(myModel, img_test); %Predicted Class
```

**myModel** contains the model with above parameters and settings, where
as **outputLabel** contains the predicted vector based on our model.
Both these commands may take long to execute. These variables are
therefore loaded below:

``` {.codeinput}
load Data/9844mdl.mat;
idx = outputLabel ~= labels_test;
totalErrors = sum(idx)
```

``` {.codeoutput}
totalErrors =

   156
```

Therefore, it can be seen above that there are only 156 errors in the
above model. If we calculate the percentage, it comes out to be:

**98.44 Percent**

This value is better than my previous calculation in Assignment 2, where
I got accuracy of 96.9% using k-NN (k=25) and 91.98% using Logistic
Regression (![\$\\lambda\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_eq07657233533591063549.png)=1).
Moreover, it is very near to 98.84 percent, which is average human
accuracy. (Difference of only 0.4 percent or 40 observations). Note that
further tuning can also be done by running k from 5 to 7 (divided in 100
fractions) and C from 0.01 to 1 (divided in 100 fractions), as we
already know that (6,1) give us the best result at the moment.

We can also see 9 of the misclassified images to get some more insight:

``` {.codeinput}
allSolution = outputLabel;
X = img_test(idx,:);
y = labels_test(idx); allSolution = allSolution(idx);
luckyIdx = 1:9;
printX = X([luckyIdx],:);
correctAnswer = vec2mat(y([luckyIdx]),3)
calculatedAnswer = vec2mat(allSolution([luckyIdx]),3)
hFig = figure(3);
for i = 1:9
    subplot(3,3,i);
    imagesc(vec2mat(printX(i,:),28)');
end
snapnow;
close(hFig);
```

``` {.codeoutput}
correctAnswer =

     2     4     2
     5     3     6
     8     8     2


calculatedAnswer =

     9     2     7
     3     7    10
    10     2     8
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_03.png)

**About the above observation:** As we can see the image above almost
all of them (except 4 in the first row and 8 in ithe last row) are hard
to classify. Even we (humans) would need to put some effort if we want
to correctly identify these numbers. Hence, all these values are
essentially the **noise** which we were successfully able to ignore by
using appropriate C value. However, there is futher scope of improvement
because 4 and 8 were also misclassified.

The Confusion Matrix {#12}
--------------------

The confusion matrix was plotted using the method **plotconfusion()**
which is present in the neural network toolbox. And since it may not be
available in all the systems, I am commenting the method execution below
and directly loading the confusion matrix. However, the line can be
uncommented if the aforementioned toolbox is installed. The confusion
matrix image is saved in \'Data/confMat.jpg\' in this folder.

``` {.codeinput}
targetLabel = labels_test; %True Class
% Preprocessing the the data so that it can be usedin the plotconfusion()
% method.
outputMat = zeros(10,size(outputLabel,1));
targetMat = zeros(10,size(targetLabel,1));
for i = 1:size(outputLabel,1)
    targetMat(targetLabel(i),i) = 1;
    outputMat(outputLabel(i),i) = 1;
end
% plotconfusion(targetMat,outputMat);
[I,map] = imread('Data\confMat.jpg','jpg');
imshow(I,map);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise5_04.png)

In the confusion matrix above, the labes in the Y-Axis are our
**Calculated Values** from the model, where as the labels in the X-Axis
are the **Actual Values**.

Nomenclature: For example, a value of 9 in the location-(2,7) (MATLAB
Notation) means that there were a total of 9 instances where the number
**7** was misclassified as **2**. The red color background signifies
misclassifications and green signifies correct classification. Notable
points that can be taken from the confusion matrix:

<div>

1.  **1** and **0** are the two numbers which had the highest
    classification percentage (99.2%), with only 9 and 8 errors
    respectively. This is natural as 1 and 0 can be declared as two of
    the most simple shapes.
2.  **9** had the lowest classificaton percentage (97%), with a total of
    30 errors. As predicted in the hypothesis, 9 was wrongly classified
    mostly as 4 (total 8 times) which is the one of the highest
    misclassificaton pair, second only to 7 being misclassified as 2.
3.  Probability of our model classifying the number **1** correctly was
    99.4%, which is the highest probability amongst all other classes.
4.  Similarly, probability of our model classifying the number **7**
    correctly was the lowest with 97.7%.

</div>


Exercise 1 - Visualization and Cross-Validation
===============================================

Submitted by **Prasannjeet Singh**

Contents
--------

<div>

-   [Q1. Using fitctree()](#1)
-   [Q2. Decision Boundary](#2)
-   [Q3. k-Fold Cross Validation](#4)
-   [Q4. Zero training error decision tree.](#6)

</div>

Q1. Using fitctree() {#1}
--------------------

``` {.codeinput}
load Data/data1.mat;
mdl = fitctree(X,y);
view(mdl);
view(mdl, 'Mode','graph');
snapnow;
hTree=findall(0,'Tag','tree viewer');
close(hTree);
```

``` {.codeoutput}
Decision tree for classification
 1  if x1<13.5 then node 2 elseif x1>=13.5 then node 3 else 0
 2  if x2<10.5 then node 4 elseif x2>=10.5 then node 5 else 1
 3  class = 0
 4  if x2<-35 then node 6 elseif x2>=-35 then node 7 else 1
 5  class = 0
 6  class = 0
 7  if x2<-3.5 then node 8 elseif x2>=-3.5 then node 9 else 1
 8  class = 1
 9  if x2<-2 then node 10 elseif x2>=-2 then node 11 else 1
10  class = 0
11  class = 1
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise1_01.png)

Q2. Decision Boundary {#2}
---------------------

The decision boundary was calculated by using the inbuilt method
**predict()** which predicts the answer based on the decision tree
created earlier. Using predict, all the values were calculated for a
graph with both the axes (-50,50) and stored in a matrix. 50 was chosen
manually by observing the scatter of the input data X. Later
**contourf()** was used in the matrix to plot the decision boundary.

``` {.codeinput}
v = repmat(-50:50,[101,1]);
% Creating a two-columned matrix 'z', that contains all the integer points
% in a square graph of axes (-50,50). Next, using 'predict' to calculate
% predicted answers for all the points in 'z'.
z =  [v(:) repmat([-50:50]',[101,1])];
bMatrix = predict(mdl,z);
bMatrix = vec2mat(bMatrix,101);
hFig = figure(2);
contourf(-50:50,-50:50,bMatrix,1);
title('Decision Boundary for Decision Tree');
legend('Yellow - 1 | Blue - 0');
snapnow; close(hFig);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise1_02.png)

**What is the characteristic for the decision boundary of a tree?
Conceptually, can you obtain any kind of decision boundary using
decision trees?**

In case of a decision tree, the algorithm tries to split the entire
dataset in two parts based on any one feature, and the feature that will
be chosen depends upon the greedy algorithm. Therefore, in our example
which has a dataset with two features, the dataset with be split either
horizontally or verticaly.

It is evident that regardless of the chosen axis, the algorithm can only
split the dataset parallel to the x OR y axis at any iteration. This is
a unique characteristic that can be observed in a decision boundary for
a decision tree. Therefore, we can conclude that *any kind of* decision
boundary, like what we saw in the assignment 1 is not possible in this
case, and the boundaries will only consist of straight lines parallel to
the x and y axis, as clearly seen in the above figure. However, it
should be noted that **if the dataset is huge** and **if the minimum
parent size is very small**, the decision boundary might appear to be of
*any* shape, but a closer look (zoomed-in) will reveal that the
boundaries are actually parallel to either x or y axis.

Q3. k-Fold Cross Validation {#4}
---------------------------

For choosing a value of \'k\', following thoughts were taken into
consideration:

<div>

1.  *k* is preferably a divisor of number of training sets (n).
2.  *k* is considerably small such that there is enough training data.

</div>

Since *n* is 60 in this example, k was taken as *10*.

The function **kfoldLoss()** calculates the classification error,
specified by \'LossFun\',\'classiferror\'. It is the weighted fraction
of misclassified observations, with the equation:

![\$\$L = \\sum\_{j=1}\^n {w\_j I \\big\\{\\widehat{y\_j} \\neq y\_j
\\big\\} }\$\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise1_eq03196131173537897014.png)

Where ![\$\\widehat{y}\_j\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise1_eq02832347711439461536.png) is
the class label corresponding to the class with the maximal posterior
probability. I{x} is the indicator function. (Taken from [this
link](https://se.mathworks.com/help/stats/classificationpartitionedmodel.kfoldloss.html#bswic2v-2)
of the **Matlab Documentation**). Therefore, the final value can be
multiplied with *n* (60 in our case) to get the total number of
misclassifications.

``` {.codeinput}
k = 10;
cvmodel = crossval(mdl,'KFold',k);
WeightedLoss = kfoldLoss(cvmodel,'lossfun','classiferror','mode','average');
% round() is used below, as sometimes division by 6 can create fractional
% values, and multiplying it by 60 again does not completely remove the
% fractional part.
ClassificationError = round(WeightedLoss*60)
```

``` {.codeoutput}
ClassificationError =

    10
```

Above we can see the total classification error obtained with 10-Fold
cross validation. Note that the value is prone to change with every
execution of the function, as the datasets are permuted before we
perform k-Fold.

Q4. Zero training error decision tree. {#6}
--------------------------------------

``` {.codeinput}
mdl2 = fitctree(X,y,'MinParentSize',1);
bMatrix = predict(mdl2,z);
bMatrix = vec2mat(bMatrix,101);
```

Visualizing as a **Graph**:

``` {.codeinput}
view(mdl2, 'Mode','graph');
snapnow;
hTree=findall(0,'Tag','tree viewer');
close(hTree);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise1_03.png)

Visualizing by the **Decision Boundary**:

``` {.codeinput}
hFig = figure(2);
contourf(-50:50,-50:50,bMatrix,1);
title('Decision Boundary for MinParent Size = 1');
legend('Yellow - 1 | Blue - 0');
snapnow; close(hFig);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise1_04.png)

**k-Fold Cross Validation**

k Value will be kept the same.

``` {.codeinput}
cvmodel = crossval(mdl2,'KFold',k);
WeightedLoss = kfoldLoss(cvmodel,'lossfun','classiferror','mode','average');
% round() is used below, as sometimes division by 6 can create fractional
% values, and multiplying it by 60 again does not completely remove the
% fractional part.
ClassificationError = round(WeightedLoss*60)
```

``` {.codeoutput}
ClassificationError =

    12
```

In the test run, the total classification errors in case of minimum
parent size as the default value (10) was **9**, and in case of minimum
parent size as 1 was **13**. (Note that the values may slightly change
everytime this file is run).

The fact that total error in case of a decision tree with zero training
error is more than the former is very much expected, as this comes under
the category of **overfitting**, and as we have seen in various
questions in previous exercises, overfitting always leads to high
training errors. Although we do not have a training data in this case,
k-Fold converts parts of the test data into training data in each
iteration, therefore this can also, in a way, be considered as training
error, which is higher in this case of overfitting.

Exercise 2 - Depth control and pruning for Titanic Data
=======================================================

Submitted by **Prasannjeet Singh**

Contents
--------

<div>

-   [Q1. Preprocessing and Importing Titanic Data](#1)
-   [Q2.1. Depth Control](#2)
-   [Q2.2. Pruning](#7)
-   [Comparison](#10)

</div>

Q1. Preprocessing and Importing Titanic Data {#1}
--------------------------------------------

If we observe the CSV file, we find that it contains a total of 12
columns with different properties which may or may not be useful for the
training of decision trees. Moreover, we can also observe that many
cells are empty. It was observed that most (if not all) of the empty
sells belonged to the **age** category. While there was an option to
fill the cells with zeroes or something else using some metric, it would
still have affected the overall model, as those wouldn\'t have been the
real values. Therefore, it was decided to omit all the rows which had
empty cells for age.

Additionally, there were also some empty cells in the \'Embarked\'
category. Also in this case, the number of rows were only 2 (was
calculated by *isnan(table\_name)*), and therefore, it was also decided
to remove those rows.

Sibsp and Parch Sibsp and Parch respectively denote number of
siblings/spouse and number of parents/children. However, there is no
reason to believe that people are more likely to survive/not survive if
they have more siblings/spouse as compared to parents/children, or vice
versa. Therefore, all the relatives were merged to one by adding both
these columns into one.

Reasoning behind choosing variables for decision trees:

*Note: The property **Pclass** is assumed to be the ticket-class of the
passenger, where 1: 1st class, 2: 2nd Class and 3: 3rd Class.*

Properties like Passenger ID/Name/Ticket ID and Cabin number make no
difference in their chances of survival in the titanic, and therefore,
these properties were completely removed in the calculation of decision
trees. Additionally, it was observed that there are many conflicts in
the \'Fare\' property in the table. Minimum ticket fare for first class
was observed to be 5, however, minimum ticket fare for second class was
observed to be 10. Therefore, this property highly conflicts with the
\'PClass\' property, and I believe that including both \'PClass\' and
\'Fare\' may adversely affect the model, and therefore, I decided to opt
out of the \'Fare\' property for creating this model. Moreover, a person
with a First Class ticket was more likely to be given preference than
someone with a Second Class ticket, and the fares of the tickets
wouldn\'t have made any significant difference. The idea is inspired
from [this Independent.co.uk link.](https://ind.pn/2K7vH5S)

Therefore, finally we are left with the following properties, which will
be used in formulating the decision tree:

<div>

1.  PClass
2.  Age
3.  Sex
4.  Relatives (Sibsp + Parch)
5.  Embarked

</div>

There are strong reasons to believe that these five properties are
highly responsible for the survival or non-survival of a passenger. In
case of Age, it is more likely that children were preferred over others.
Likewise, women might have had a higher likelihood of being favored over
men. *Embarked* might also have had a slight affect on passenger\'s
survival, as people embarking later may not have gotten their preferred
seats/cabins. Properties PClass and Relatives were discussed earlier.
All the properties above, after performing the appropriate
preprocessing, were stored in a matrix *X* in the given sequence. The
**Survived** property is indubitably used, and is saved in the solution
matrix *y*.

All the pre-processing steps are accompanied with comments below:

``` {.codeinput}
% Extracting the Import Options object.
opts = detectImportOptions('Data/titanic.csv');
% In the Import Options, changing the rule to omit all the rows if a cell
% has missing value
opts.MissingRule = 'omitrow';
% Also specifying the number of columns that we need to be imported.
opts.SelectedVariableNames = [2 3 5 6 7 8 12];
% Reading the table according to the import options we created above
data = readtable('Data/titanic.csv',opts);
% Adding the 'Sibsp' and 'Parch' columns into one and renaming the column
% to 'Relatives'
data(:,'SibSp') = array2table(table2array(data(:,'SibSp')) + table2array...
    (data(:,'Parch')));
data(:,'Parch') = [];
data.Properties.VariableNames{'SibSp'} = 'Relatives';
% Changing the 'Pclass', 'Sex', 'Embarked' and 'Survived' columns into
% categorical values.
data.Pclass = categorical(data.Pclass);
data.Sex = categorical(data.Sex);
data.Embarked = categorical(data.Embarked);
data.Survived = categorical(data.Survived);
% Separating a part of data to use it as test
testData = data(1:100,:);
data(1:100,:) = [];
% Applying fitctree and viewing the tree with default configurations
tree = fitctree(data,'Survived'); % Passing tabular parameters in fitctree
view(tree,'Mode','graph');
hTree=findall(0,'Tag','tree viewer');
set(hTree, 'Position', [0 0 1000 500]);
snapnow;
close(hTree);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise2_01.png)

Q2.1. Depth Control {#2}
-------------------

Before we control the depth, first we will decide the opmimal number of
splits for the decision tree. As we know, the default value of
MaxNumSplits in case of **fitctree()** is **n-1**, where n is the size
of the sample data, therefore we will run a loop with MaxNumSplits from
1 to n-1 and apply 10-fold cross validation on each model, to find out
the MaxNumSplits that gives us the least cost. That will be the value we
will choose for our model.

Calculating the decision tree around 600 times takes a while, and
therefore, I have already performed it and saved the result in
**bestSplit.mat**, which I have loaded below:

``` {.codeinput}
% Commented code below calculates the bestDepth variable, which is already
% calculated and loaded to save time.
% k=10;
% maxPossibleSplits = size(data,1)-1;
% for i = 1:maxPossibleSplits
%     fprintf(strcat(num2str(i),'\n\r'));
%     mdl = fitctree(data,'Survived','MaxNumSplits',i);
%     cvmodel = crossval(mdl,'KFold',k);
%     WeightedLoss = kfoldLoss(cvmodel,'lossfun','classiferror','mode','average');
%     bestSplit(i,:) = [i WeightedLoss];
% end

% Loading the file and plotting MaxNumSplits vs the Cost
load 'Data/bestSplit.mat';
hFig = figure(2);
plot(bestSplit(:,1), bestSplit(:,2));
title('Maximum Number of Splits vs Cost');
xlabel('Number of Splits');
ylabel('Cost');
snapnow;
close(hFig);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise2_02.png)

As we can see above, we don\'t really find a pattern for the cost vs
maximum number of splits. Therefore, we will check the top ten split
values, which give us the minimum cost below:

``` {.codeinput}
bestSplit = sortrows(bestSplit,2);
bestSplit = bestSplit(1:10,:);
bestSplit = array2table(bestSplit);
bestSplit.Properties.VariableNames{'bestSplit1'} = 'MaxNumSplits';
bestSplit.Properties.VariableNames{'bestSplit2'} = 'Cost';
bestSplit
```

``` {.codeoutput}
bestSplit =

  10├ù2 table

    MaxNumSplits     Cost
    ____________    _______

         10         0.18791
          7         0.18954
        518         0.18954
         21         0.19118
        409         0.19118
          6         0.19118
        601         0.19281
        228         0.19281
          5         0.19444
          8         0.19444
```

Now if we take a closer look at the data above, we realize that split =
10 gives us lowest cost, however, split = 7 also gives us a considerable
amount, with less splits. In this case, had the split value that gives
us the least cost been very high, we could have gone for the next best,
i.e. 7. But since there is not much difference between number of splits,
we will stick to the least cost value, i.e. MaxNumSplits = 10.

``` {.codeinput}
cmdl = fitctree(data,'Survived','MaxNumSplits',10);
view(cmdl,'Mode','graph');
hTree=findall(0,'Tag','tree viewer');
snapnow;
close(hTree);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise2_03.png)

Now, if we observe the very first tree that we made (without any
splits), we can observe that the maximum number of depth was 15, and
generally, the first thing we would want to do is to reduce the depth by
half, which is around 7, to make the model simpler. However, after
choosing MaxNumSplits as 10, the resultant tree already has a maximum
depth of 4, which is much simpler. Therefore, we will consider this as
our final model, without choosing the depth again. However, if we were
to do it, we could have done it by converting the current dataset to
tall data and then applying **fitctree()** as follows:

``` {.language-matlab}
m=7
tallData = tall(data);
mdl = fitctree(data,'Survived','MaxNumSplits',10,'MaxDepth',m);
```

Since the maximum number of splits has already been finalized, the
depths would have been calculated by keeping the splits as 10.

Nevertheless, I have calculated the final model with maximum split as 10
and saved it as **splitModel.mat**. The model has already been
visualized above. To check the performance, we have chosen first 100 as
the test data and the rest as training to find out the total number of
erros:

``` {.codeinput}
clearvars cmdl;
load Data/splitModel.mat;
estimatedSurvival = predict(cmdl,testData);
actualSurvival = categorical(testData.Survived);
totalErrors = sum(estimatedSurvival ~= actualSurvival)
```

``` {.codeoutput}
totalErrors =

    26
```

Therefore, according to the current chosen model, total errors are 20.

Q2.2. Pruning {#7}
-------------

Pruning can directly be performed by comparing all the prune levels and
selecting the one which gives us the minimum cross-validated error. This
can be done like so:

*Note that in this case we will work on the fully grown tree.*

``` {.codeinput}
[~,~,~,bestlevel] = cvLoss(tree,'SubTrees','All','TreeSize','min')
```

``` {.codeoutput}
bestlevel =

     3
```

Therefore, according to above, the most optimized prune level is 4.

We can also find out the best pruning level by checking our test data on
each pruning level (0 to 8 in our case), and selecting the one that
gives us the least error. This can be done like so:

``` {.codeinput}
clearvars pruneError;
for i = 0:8
    prunedTree = prune(tree,'Level',i);
    estimatedSurvival = predict(prunedTree,testData);
    pruneError(i+1,:) = [i sum(estimatedSurvival ~= actualSurvival)];
end
pruneError = array2table(pruneError);
pruneError.Properties.VariableNames{'pruneError1'} = 'PruneLevel';
pruneError.Properties.VariableNames{'pruneError2'} = 'TotalErrors';
pruneError
```

``` {.codeoutput}
pruneError =

  9├ù2 table

    PruneLevel    TotalErrors
    __________    ___________

        0             23
        1             22
        2             20
        3             19
        4             18
        5             22
        6             23
        7             21
        8             34
```

Therefore, as seen above, even in this case, the best prune levels are 3
and 4, which is in concurrence to what we received above by comparing
crossvalidated results. Let us choose 4 as the final prune level for the
model, and view the pruned tree and error: (The pruned tree model was
already created and saved in the folder *Data*.

``` {.codeinput}
clearvars prunedTree hTree estimatedSurvival prunedError;
% prunedTree = prune(tree,'Level',4);
% Loading the already created pruned tree:
load Data/prunedTree.mat;
view(prunedTree,'Mode','Graph');
hTree=findall(0,'Tag','tree viewer');
snapnow;
close(hTree);
estimatedSurvival = predict(prunedTree,testData);
prunedError = sum(estimatedSurvival ~= actualSurvival)
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise2_04.png)

``` {.codeoutput}
prunedError =

    18
```

Comparison {#10}
----------

Total errors in case of MaxSplit = 10 was 23, where as total errors with
prune-level-4 was 18. Therefore, we can conclude that for our test data,
pruned tree (with prune level 4) performs better.

Exercise 3 - Support Vector Machines
====================================

Submitted by **Prasannjeet Singh**

Contents
--------

<div>

-   [Q1. Calculating the perpendicular distance:](#1)
-   [Q2. Appending \[-1 2\]](#2)

</div>

Q1. Calculating the perpendicular distance: {#1}
-------------------------------------------

If ![\$ax+by+c=0\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise3_eq00242236477706248943.png) is the
equation of the hyperplane, and one of the support vector point is
![\$(p,q)\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise3_eq11475799023219699482.png), then the distance
was calculated by:

![\$\$\\frac{\|ap+bq+c\|}{\\sqrt{a\^2+b\^2}}\$\$](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise3_eq16669572657172333325.png)

This has been implemented in the file **mmcPlot.m** at the designated
place.

Trying to run the function after the implementation of distance formula:

``` {.codeinput}
load mmcData.mat;
hFig = figure(2);
mmcPlot(X,y);
snapnow; close(hFig);
```

``` {.codeoutput}
Minimum found that satisfies the constraints.

Optimization completed because the objective function is non-decreasing in
feasible directions, to within the default value of the optimality tolerance,
and constraints are satisfied to within the default value of the constraint tolerance.


```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise3_01.png)

Q2. Appending \[-1 2\] {#2}
----------------------

``` {.codeinput}
X(end+1,:) = [-1 2];
y(end+1) = -1;
hFig = figure(2);
mmcPlot(X,y);
snapnow; close(hFig);
```

``` {.codeoutput}
Minimum found that satisfies the constraints.

Optimization completed because the objective function is non-decreasing in
feasible directions, to within the default value of the optimality tolerance,
and constraints are satisfied to within the default value of the constraint tolerance.


```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise3_02.png)

**Observation**

As we included the new point (-1,2) and classified it as -1 (red), the
window to create a hyperplane that separates the two sets became very
narrow. Moreover, this also changed the support vector for the negative
(red group). Additionally, it also shows us that Support Vector Machies
possess the property of **non robustness** in some cases, if the new
training points are placed near the separating hyperplane. However if
the new training points are placed farther away from the hyperplane, it
wouldn\'t have made any difference.

It is also worth noting that had the new point been classified as +1
(yellow), probably there wouldn\'t have been much difference in the
hyperplane.

Exercise 4 - Classifying the BM-Dataset
=======================================

Submitted by **Prasannjeet Singh**

Contents
--------

<div>

-   [1. Plotting the Dataset](#1)
-   [2. Running **fitcsvm()** with chosen hyperparameters](#2)
-   [3. Plotting Decision Boundary](#3)
-   [4. Observation with Complete Data:](#6)

</div>

1. Plotting the Dataset {#1}
-----------------------

Note that I have used only 10,000 datasets to reduce the time taken to
run the project. However, I have also performed the same with complete
dataset and saved the result. Those are also displayed at the bottom.

``` {.codeinput}
% Loading the dataset
load BM.mat
% Cropping the dataset to first ten thousand items
X = batmanX(1:10000,:);
Y = num2cell(num2str(batmany(1:10000,:)));
hFig = figure(1);
gscatter(X(:,1),X(:,2),Y);
title('Scatter Diagram of Batman (Partial Data)');
axis tight;
snapnow;
close(hFig);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise4_01.png)

2. Running **fitcsvm()** with chosen hyperparameters {#2}
----------------------------------------------------

Following hyperparameters were used to make the decision boundary as
precise without taking much time:

<div>

1.  KernelFunction: gaussian
2.  BoxConstant: 1
3.  KernelScale: 1

</div>

``` {.codeinput}
classes = unique(Y);
rng(1); % For reproducibility
SVMModels = fitcsvm(X,Y,'KernelFunction','gaussian','BoxConstraint',1,'KernelScale',1.5);
```

3. Plotting Decision Boundary {#3}
-----------------------------

Creating a Mesh-Grid (x-y axis) from min value of x to max value of x
with a separation of 0.02:

``` {.codeinput}
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
% Predicting for all the values in the mesh grid
maxScore = predict(SVMModels,xGrid);
% Plotting the Batman Decision boundary:
f1 = figure(3);
gscatter(xGrid(:,1),xGrid(:,2),maxScore,[0.1 0.5 0.5; 0.5 0.1 0.5]);
title('The Batman Decision Boundary');
axis tight;
snapnow;
close(f1);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise4_02.png)

**Training Error Rate**

``` {.codeinput}
trainingError = sum(str2num(cell2mat(predict(SVMModels,X))) ~= batmany(1:10000))
trainingErrorRate = trainingError/size(X,1)*100
```

``` {.codeoutput}
trainingError =

   174


trainingErrorRate =

    1.7400
```

Therefore, it can be observed that current model has only 174 errors out
of a total of 10000 datasets, with the training error rate of 1.74
percent.

4. Observation with Complete Data: {#6}
----------------------------------

Following hyperparameters were used which made nicer decision
boundaries, but took huge time. Note that to make a sharp decision
boundary we are basically trying to **overfit** our model. And in that
case, we have to impose huge penalty if any point tries to cross the
margins. And we can impose higher penalty by increasing the parameter
**BoxConstant**. However, an increase in BoxConstant results in
increased run time of the algorithm.

<div>

1.  KernelFunction: gaussian
2.  BoxConstant: 20
3.  KernelScale: 0.2

</div>

Loading the data to showcase the results:

The data contains both fitcsv model and the predicted solution to make
the decision boundary. However, the model will not be used as I have
already predicted the boundaries to save time.

``` {.codeinput}
load Data/batmodel.mat;
hFig = figure(4);
set(hFig, 'Position', [0 0 1500 500]);
subplot(1,2,1);
gscatter(batmanX(:,1), batmanX(:,2), batmany);
title('Scatter for complete Batman data');
axis tight;
subplot(1,2,2);
gscatter(xGrid(:,1),xGrid(:,2),batscore,[0.1 0.5 0.5; 0.5 0.1 0.5]);
title('Decision boundary with improved hyperparameters');
axis tight;
snapnow;
close(hFig);
```

![](https://github.com/prasannjeet/image-recognition/blob/master/publishedHTML-Files/Exercise4_03.png)

**Training Error Rate**

Since we have tried to overfit the data, we expect lower training error
rate than what we got below. Training error rate can be calculate via
the following code:

``` {.codeinput}
% trainingError = sum(str2num(cell2mat(predict(BatModel,batmanX))) ~= batmany)
% trainingErrorRate = trainingError/size(batmanX,1)*100
```

However, since it takes time, I have already calculated it and the
training error, on a total of 100,000 training set turned out to be only
107, with a training error rate of **0.107%**, which is noticeably lower
than our previous calculation.