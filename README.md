# A Solution (10th, 0.27915 acc) for KKBOX Data Game - 17.06

## Method
This solution is an ensemble MLP which uses following features:

* last_movie_time
* prefers
* event_time_max
* event_time_hour
* event_time_weekday
* total_watch_time

To learn more about the feature definition, please reading `task/feature.py`.

## Reproduce
### Extract Files
Extract files from `download/assets.17.06.zip` to `data`.

### Install Dependencies
`pip install -r requirements.txt`

### Segment Data
`python -m task.segment`

### Extract Features
`python -m task.extract`

### Train and Predict
`python -m task.train` -- and find the generated submission file in `submission` dir.
