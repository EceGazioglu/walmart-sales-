Walmart Sales — Regression Analysis
Task

Supervised, regression. Weekly sales is the target — a model can be trained to predict weekly sales, and the analysis can also determine what feature affects the weekly sales most. Models used/compared: baseline OLS, then Lasso, Ridge, ElasticNet, SVR, RandomForestRegressor and XGBoost. Whether this model selection shift is meaningful is checked via a Friedman test.

Dataset Metadata
Store — 45 stores
Date — was in object type, every Monday is entered; type changed into datetime
Weekly_Sales — sales of each week; amount or revenue? According to the data owner it's revenue in k
Holiday_Flag — binary
Temperature — in Fahrenheit, a few outliers but possible, not treated as outlier
Fuel_Price — no outliers, weekly
CPI — no outliers, originally reported monthly but converted to weekly observations in this dataset
Unemployment — originally quarterly, not converted into weekly, possible edge values
EDA Findings
No nulls, no duplicates.
Since there's no metadata about this data, thorough examination was needed.
Holiday_Flag shows expected imbalance.
Temperature is in Fahrenheit, between -19°C and 38°C — reasonable.
Stores 30, 36, 37, 38, 44 show different behavior than others — their sales are higher in the non-holiday weeks. Further information would be needed to give insight about this, but maybe those stores are located in business areas that are crowded on business days.
Expected holidays to have the biggest impact on sales, but general economy affects sales more — from this, maybe the brand is a luxury goods brand.
Feature Engineering

Date → cyclical week signal: Since this data is not a time series and ML models cannot comprehend dates, before dropping the Date column a signal needed to be derived from it. Since the date is weekly and the range includes more than a year, the week number can be a signal for estimating sales. But there is a cycling issue — ML algorithms cannot know that week 1 comes after week 52, they can be interpreted as too far away from each other — so a sin/cos transformation was applied. As a result, at the end of the year week 52's sin value and the beginning of the year week 1's sin value came out perfectly close to each other.

Store → one-hot encoding: To eliminate the dummy trap, the first store was kept as the intercept (drop_first=True).

Model Results

Baseline (OLS / LinearRegression):

R² score: 0.9165
RMSE: 163,417.39

Model selection (GridSearchCV, 5-fold CV, scoring = RMSE): Grid chose XGBRegressor with learning_rate=0.2, max_depth=7, n_estimators=200 as the best estimator. But in order to retain more consistency, the second-best model with less std_score was chosen instead: XGBoost with learning_rate=0.2, max_depth=5, n_estimators=200.

Final model (XGBoost, max_depth=5, n_estimators=200, learning_rate=0.2):

R² score: 0.9667
RMSE: 103,117.57
Friedman Test

p-value came out really small: 0.0003. So, we can reject H0 — models' results are significantly different from each other. The trials of searching for the best model are not nonsense: model performance differences are not just noise, and the best model cannot be defended only by it having a lucky fold.

Feature Importance

This tells us Store 20 is important, but this is not a very actionable insight — what else is an important signal for sales other than stores?

Unemployment is the culprit — it has the most effect on sales, other than the store identity itself.

General relationship between weekly sales and unemployment: -0.1062 Store 20 relationship between weekly sales and unemployment: -0.0953

There is a trend in unemployment and we are in a rise point right now, so we can expect, unless a shock in economy doesn't occur, a little decrease in weekly sales — since weekly sales and unemployment have a little negative relationship.
