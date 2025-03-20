# fitness_tracker_v3

To run locally 

step 1. Create a virtual environment
$ python3 -m venv tracker 

step 2. Activate the environment 
$ source tracker/bin/activate

step 3. Install requirements
$ pip install -r requirements.txt

step 4. Run app.py file
$ python app.py


This machine learning project clssify the excercise using gyroscope and accelerometer.
The data for this is taken from 5 persons which perform 5 excercises using Metamotion sensor.
1. Squat
2. Rowing
3. Over head press
4. Bench press
5. Shoulder Press

Tools used to automate the process
1. Jenkins for CI workflow 
2. Mlflow for Experiment Tracking
3. Using google drive for data storage
4. when you provide a json file in which have info about accelerometer and gyrocsope can predict the excercise with 99.58% acccuracy