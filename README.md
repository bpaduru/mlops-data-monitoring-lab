# Data Monitoring Lab with Evidently AI (Bike Sharing Data)

This is my submission for the Data Monitoring lab from the course repo (Labs/Data_Labs/Data_Monitoring/Evidently_AI). The goal of the lab is to detect data drift, which means checking if new data coming into a system still looks like the data a model was trained on. If the data changes a lot, a model trained on the old data may stop working well, so monitoring for drift is an important part of MLOps.

## What the original lab does

The original notebook (Lab1.ipynb) uses the Adult census dataset from OpenML. It splits the data into two groups based on education level, treats one group as reference data and the other as production data, and then runs Evidently's DataDriftPreset to compare them. Since the split is based on education, drift is expected.

## What I changed

1. I used a different dataset. I used the Bike Sharing dataset from Kaggle (day.csv), which has daily bike rental counts in Washington D.C. for 2011 and 2012 along with weather and season information. It has 731 rows.
2. I split the data by year instead of by a feature. 2011 is the reference data and 2012 is the current data. I think this is closer to a real situation where a model is trained on last year's data and we want to check if this year's data is still similar.
3. I wrote my own schema for this dataset. Some columns like season, mnth, weekday and weathersit are stored as numbers but are actually categories, so I listed them as categorical columns. I dropped instant (row number), dteday (date) and yr (since I split on it).
4. I added DataSummaryPreset to the report. It was imported in the original notebook but never used.
5. I saved all reports as HTML files in the reports folder so they can be opened in a browser.
6. I added two more experiments (explained below): a random split and a simulated sensor problem.
7. I removed the CloudWorkspace cell from the original notebook because it had someone else's API token in it, and the report works locally without it.
8. I added a requirements.txt file and a .gitignore, which the original lab did not have.

## Experiments and results
## Dataset source

Bike Sharing Dataset from Kaggle: https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset/data

### How Evidently decides drift

For each column Evidently runs a statistical test and gives a p value as the drift score. For numerical columns it used the K-S test, for categorical columns with more than two values it used chi-square, and for the two value columns (holiday and workingday) it used a Z-test. If the p value is below 0.05 the column is marked as drifted.

### Experiment 1: 2011 vs 2012

4 out of 13 columns were flagged as drifted (share 0.308). These were cnt, registered and casual (all with p value 0.0) and hum (p value 0.009).

Average daily rentals in 2011 were 3405.76 and in 2012 were 5599.93, so rentals went up by around 64 percent.

The weather and calendar columns like temp, atemp, windspeed, season, mnth and weekday did not drift. weathersit was very close (p value 0.052) but was not flagged. So the conditions were mostly the same in both years, but the number of people renting bikes went up a lot. This shows that the drift is in demand and not in the weather. If a model was trained on 2011 data to predict rentals, it would probably predict too low for 2012. The humidity drift was a bit surprising to me, it looks like 2012 was slightly less humid than 2011.

### Experiment 2: random split

I split the whole dataset randomly into two halves using sample with random_state=42. Since both halves come from the same data, I expected little or no drift.

The result was not what I expected. 5 out of 13 columns were flagged (share 0.385): weekday, workingday, mnth, casual and hum. This is actually more columns than experiment 1.

When I looked at the distributions in the report, the main reason was that my random split did not divide the weekend days evenly. One half had 81 Saturdays and Sundays and the other half had 129. Because of this, weekday and workingday drifted, and casual also drifted because casual riders rent more on weekends. So one random imbalance caused drift in three related columns. hum (p value 0.039) and mnth (p value 0.006) were also flagged, while cnt, registered, temp and the others were not.

From this I understood that drift detection is sensitive when the dataset is small (around 365 rows in each half), and that with 13 columns tested at 0.05 some columns can be flagged just by chance. A random split is not always a clean "no drift" baseline, and the results should be checked by looking at the actual distributions and not only the drifted count.

### Experiment 3: simulated humidity drift

I copied the 2011 data and multiplied the hum column by 1.3 to act like a humidity sensor giving wrong readings. Then I compared it with the original 2011 data. If monitoring works, only hum should show drift.

Only 1 out of 13 columns was flagged (share 0.0769), and it was hum with a p value of 0.0. All the other columns had a p value of 1.0 because they were exactly the same as the reference data. So Evidently found exactly the column I changed and nothing else, which is what I expected.

## Problems I faced

My default Python version was 3.14. When I ran the notebook, importing pandas failed with an OverflowError from numpy (cannot convert longdouble infinity to integer). I tried installing numpy 1.26.4, but then scipy 1.18.1 complained because it needs numpy 2. Then I tried installing scipy 1.13.1 and it failed with a metadata generation error, because there was no ready package for Python 3.14 and pip tried to build it from source.

I fixed this by creating a new virtual environment with Python 3.11 (venv311) and installing the packages again there. I also installed pandas below version 3 because evidently 0.7.0 is older than pandas 3. After that everything ran without errors.

I also had to change the notebook kernel in VS Code to venv311, because it was still using the old environment.

## How to run

1. Clone this repo.
2. Create a virtual environment with Python 3.11 and activate it:

```
py -3.11 -m venv venv311
venv311\Scripts\activate
```

3. Install the packages:

```
pip install -r requirements.txt
```

4. Open bike_drift_monitoring.ipynb, select the venv311 kernel and run all cells. The reports will be saved in the reports folder.

## Files in this repo

Lab1.ipynb is the original lab notebook (with the cloud token cell removed).
bike_drift_monitoring.ipynb is my notebook with all my changes and experiments.
data/day.csv is the bike sharing dataset.
reports/ has the saved HTML drift reports.
requirements.txt has the package versions I used.

## What I learned

I learned how to compare a reference dataset with a current dataset using Evidently and how to read the drift report. The year comparison showed that the target (rentals) can change even when the input features stay mostly the same, which is something a model would not notice by itself. The fake humidity test showed that the report can point to the exact column that has a problem. The random split surprised me the most, because I expected zero drift but got 5 drifted columns, and it taught me to look at the distributions behind the numbers before trusting them. I also learned that setting up the environment matters a lot, since the Python version alone stopped the lab from running at first.