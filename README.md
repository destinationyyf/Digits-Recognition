# Digits-Recognition

The techniques used in the project is EM algorithm under assumptions of mixture Bernoulli (since the picture can be vectorized as multi-dimensional boolean vectors). When two specific digits are chosen, basically we assume these two digits' pictures originate from `M` different Bernoulli distributions separately (`2M` Bernoullis in all), and use training set by means of EM to determine the parameter in every Bernoulli, along with the probabilities of allocating a new picture into these distributions. Then we classify pictures of both digits in test set based on MLE. 

This project contains four files.

* digits.RData is the data source, it has two datasets, 'training.data' and  'test.data'. `training.data` is a large array with dimension 10 (number of digits: 0 ~ 9) * 1000 (number of pictures of each digit) * 20 * 20 (dimensionality of each digit, boolean form). `test.data` is 10 * 500 * 20 * 20.

* Rcode.R includes all main functions that have been used in the project;

* Final Project.rmd includes the formula derivation of EM, description of how EM works and the final output using the functions created in Rcode.R

* Final_Project.pdf is the pdf file generated by Final Project.rmd using RMarkdown
