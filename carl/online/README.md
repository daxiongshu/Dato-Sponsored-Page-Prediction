The format of the input file for the ftrl is:

file sponsored fea1:value1 fea2:value2 

fea1 is the integer index of  the feature

value1 is the corresponding feature value

only non-zero value feature is included in the input svmlight file

For blending, first 'normalize' the value

the validation score is 0.969 on the full validation set (Xiaozhou's version)

running time is 10 mins

blend ftrl with xgb(0.9804) will get 0.98103 using my features only
