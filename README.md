# Bank Churning Task
## Python Files
There are 3 main files, initial_pipe.ipynb is my exploration of the data, pipeline.py is a class function to create a pipeline for future data to be scored, main.py is running on the csv with the expected output (although did not save this to csv) 


I tried 3 different ML models: Logistic Regression, Random Forest and XGBoost.
### Linear Regression 

Linear is a really simple model and I think is typically a good first pass for a classifier, it's relatively simple and easy to interpret which I think makes it a good first pass. One of the issues is that Regression assumes linearity. In this case, it didn't perform too bad but another model was more superior. Once I had properly sorted out some of the issues I was having, LR performed decently and had a recall of 0.83; misclassifying around 17% of customers who churn.

### Random Forest

Random Forest is an ensemble method that combines decision trees using a bagging method, it is again a pretty classic classification model. While i did end up scaling and transforming my data, one of the benefits with forests and trees is that you don't necessarily have to do this and that RF natively do feature selection. The draw backs would be the lack of control of the inner workings of the forrests, and that they're not super interpretable and if you have a large dataset it is computationally intensive. In this model, RF had a recall of 0.96.

## XGboost

XGBoost is another ensemble method that minimizes the loss function by adding weak learnings using gradients. Similar to the RF it doesnt need as much feature engineering, you can figure out the feature importance relatively easily. Unlike the RF, it handles large datasets well though. Again like RF, it is difficult to interpret due to the nature of the algorithm, it's also harder to tune than some simpler models. In this, the recall was also 0.96 and I ended up choosing this one as i felt it would be likely a bank might need to predict a large amount of customers.

## Memory issues

There's a couple options if a dataset can not fit in-device memory, probably the best in my opinion would be to use cloud services like AWS, Snowflake to at the very least transform the data there to see if it can be reduced in size but AWS, Azure etc have solutions for this. Alternatively using relational databases like oracle is a solution as well.

If those services are not available, pandas can load csvs in chunks, keras also has a solution using flow data.

Worst case scenario would be to take a sample chunk from the data available that your computer can load.

## Real Data
Some of the things you might have to consider is GDPR, you have to be careful about where the data is stored and how it's being sent across the team if it contains anything that could be under that umbrella so that's something that is always worth considering.

Depending on the business needs too, they might be looking for a more explainable model so they can understand the decisions, I can see that being less of a problem with Churn but maybe something like credit approval where they might have to explain to customers.

