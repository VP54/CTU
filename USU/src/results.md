## Results
1. Both models accomplished to reduce errors and increase R2Score significantly
2. I personally would use RobustRergession even though the scores are not the best for reasons I will mention in a bit

## Results
1. R2 scores very decent
2. MSE/MAE shrunk significantly

## Place for improvement
1. XGBoost/CatBoost instead of RandomForest
2. More variables such as lagging features
3. Combine more regressors
4. Delete features that do not contribute or maybe aggregate them (email/homepage) to one feature such as sentiment/loyalty

## Closing Thoughts
I could try out new methods and models and get better at forecasting. As far as my recommendation I would stick to robust regression for following reasons:
1. Good results
2. Fast to train (with more and more data as people buy groceries very often it will become lengthy and costly to train models and therefore I would prefer simple and fast, yet powerful method)
3. Scalability of RandomForest - In stores/meals etc. You will naturally want to evolve Your products/ try out new ones etc. Tree base models will not be able to predict for other than seen products and that will hinder progress/innovation of the store etc.. With a close to no performance loss You can have something that will still be very powerful to at least estimate some expectations about new/different/yet unseen product which will create space to for hypotheses and make smarter business decisions

I will try to implement app on top of this to improve my SWE skills. If anyone comes across this and would want to ask me anything or have any recommendations, critque I am more than happy to hear it since it will help me get better at what I do :) Thanks a lot and if You are reading this have a great day!