# Shallow or Deep? An Empirical Study on Detecting Vulnerabilities using Deep Learning

<div style="text-align: justify">
Deep learning (DL) techniques are on the rise in the software engineering research community. More and more approaches have been developed on top of DL models, also due to the unprecedented amount of software-related data that can be used to train these models. One of the recent applications of DL in the software engineering domain concerns the automatic detection of software vulnerabilities. While several DL models have been developed to approach this problem, there is still limited empirical evidence concerning their actual effectiveness especially when compared with shallow machine learning techniques.
In this paper, we partially fill this gap by presenting a large-scale empirical study using three vulnerability datasets and five different source code representations (i.e., the format in which the code is provided to the classifiers to assess whether it is vulnerable or not) to compare the effectiveness of two widely used DL-based models and of one shallow machine learning model in (i) classifying code functions as vulnerable or non-vulnerable (i.e., binary classification), and (ii) classifying code functions based on the specific type of vulnerability they contain (or "clean", if no vulnerability is there).  As a baseline we include in our  study the AutoML utility provided by the Google Cloud Platform. Our results show that the experimented models are still far from ensuring reliable vulnerability detection, and that a shallow learning classifier represents a competitive baseline for the newest DL-based models.
</div>

## Empirical Study Desgin

<div style="text-align: justify">
I. The <strong><em>goal</em></strong> of this study is to empirically analyze the effectiveness of deep/shallow learning techniques for detecting software vulnerabilities in source code at function-level granularity when using different models and source code abstractions. We conduct experiments with three different models (two deep and one shallow). In particular, we experiment with: (i) Random Forest (RF), (ii) a Convolutional Neural Network (CNN), and (iii) a Recurrent Neural Network (RNN), with the first being representative of shallow classifiers and the last two of deep learning models. We chose RF due to its popularity in the software engineering domain. Concerning the two DL models, they have been used, with different variations, in previous studies on the automatic detection of software vulnerabilities.

<br/>

On top of the three experimented models, we also exploit as baseline for our experiments an automated machine learning (AutoML) approach. AutoML is a solution to build DL systems without human intervention and not relying on human expertise.  It has been widely used in Natural Language Processing (NLP) and it is provided by Google Cloud Platform (GCP). AutoML eases the hyper-parameter tuning and feature selection using Neural Architecture Search (NAS) and transfer learning. This solution was released by Google in 2019 for NLP and in 2018 for Computing Vision.

<br/>
II. The <strong><em>context</em></strong> of the study is represented by three datasets of C/C++ code reporting software vulnerabilities at the function granularity level, for a total of 1,841,323 functions, of which 390,558 are vulnerable ones. 
</div>

## Research Question and Main Context

Having that goal and context in consideration, our study addresses the following research question (RQ):

### What is the  effectiveness of different  combinations of classifiers and code representations to identify functions affected by software vulnerabilities?

<div style="text-align: justify">
We answer this RQ in two main steps. First, we create binary classifiers able to discriminate between vulnerable and non-vulnerable functions, without reporting the specific type of vulnerability affecting the code. This scenario is relevant for practitioners/researchers who are only interested in identifying potentially vulnerable code for inspection/investigation. Second, we experiment the same models in the more challenging scenario of classifying functions as clean (i.e., do not affected by any vulnerability) or as affected by specific types of vulnerabilities.
</div>

### Data Collection and Labeling

We relied on three datasets composed by C/C++ source code functions and information about the vulnerabilities affecting them.

1. [GitHub Archive Dataset (GH-DS)](https://tinyurl.com/yytt25yy)

2. [STATE IV Juliet Test Suite Dataset (J-DS)](https://tinyurl.com/yy4pj6ol)

3. [Russell et al. Dataset (R-DS)](https://tinyurl.com/yywc9nrs)

### Code Representation

<div style="text-align: justify">
From each dataset we extracted two sets of tuples. The first one, in the form (function_code, is_vulnerable), aims at experimenting the models in the scenario in which we want to identify vulnerable functions, but we are not interested to the specific type of vulnerability. In the second, the tuples are instead in the form (function_code, vulnerability_type), to experiment the models in the scenario in which we want to classify the vulnerability type exhibited by a given function. We use the non_vulnerable label to identify functions not affected by any vulnerability. 

Starting from these two datasets, we built five versions of each one by representing the code in different ways, to study how the code representation affects the performance of the experimented models.
</div>

The abstract representations can be found below:

1. [GitHub Archive Dataset (GH-DS) (Abstract representations)](https://tinyurl.com/y4p3q532)

2. [STATE IV Juliet Test Suite Dataset (J-DS) (Abstract representations)](https://tinyurl.com/y6lw34t5)

3. [Russell et al. Dataset (R-DS) (Abstract representations)](https://tinyurl.com/y6kb597a)


### Data Cleaning

<div style="text-align: justify">
Before using the three datasets to train and evaluate the experimented models, we performed a [transformation and cleaning process](https://github.com/reviewdoubleblind2/shallow_or_deep/tree/master/Scripts) on each of them to (i) make the data treatable with DL/shallow models, and (ii) avoid possible pitfalls that are common in studies of machine learning on code (e.g., duplicated functions due to forked projects). 
</div>

<div style="text-align: justify">
By starting with [five different datasets]() of function representations for each dataset. We addressed conflicting representation (i.e., two samples with same code representation and different labels) and duplicates. In case of conflicting representations, all instances were removed. As for the duplicates, we removed all duplicates having the same raw source code representation and the same label (i.e., type of vulnerability affecting them, if any), keeping only the first occurrence. This means that it is possible to have in our datasets two snippets having the same abstract representation, but not the same raw source code. Such a design choice is justified by the fact that the translation from raw source code to abstract representation is part of the classification pipelines used in ML implementations, and it is performed after the removal of duplicates.
</div>

### Classifiers

<div style="text-align: justify">
Given the variables involved in our study, namely four approaches (i.e., GCP-AutoML, RF, CNN & RNN), five representations, three datasets, and two types of classification---binary and multiclass), we built a total of 120 different models. We publicly release the source code of the [base models](https://github.com/reviewdoubleblind2/shallow_or_deep/tree/master/Scripts).
</div>

