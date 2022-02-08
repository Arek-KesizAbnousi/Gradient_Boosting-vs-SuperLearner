 
# Gradient Boosting & SuperLearner
We implement Gradient Boosting & SuperLearner in R and compare the classification accuracy of the two methods.


  

     
 # Data  
     Sonar Dataset. To generate in R Code we use : library(mlbench) and data(Sonar)

     
 # Code implementation.
   
    R packages : gbm , xgboost,  SuperLearner , MASS, e1071 , mlbench.


   - For 100 independent replications, we will split the data into A training set of size 158 and a testing set of size 50.
     Then we will use the training data and fit A classifier based on Gradient boosting and SuperLearner combining prediction models.

 - The comparison will be possible by implementing   a simulated accuracy matrix which will be of a dimension 100 x 2 (100 rows and two columns)
This accuracy matrix in each of the two columns will represent the corresponding method used (column 1 = Gradient Boosting, column 2 =SuperLearner). 
By averaging through the 100 repetitions (rows) for each of the two approaches, 
we will obtain the average accuracy for each method and compare these values.


 # Conlusion
   SuperLeaner has higher accuracy; it performed better than the Gradient Boosting method.


