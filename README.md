## Titanic Dataset Analysis

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Structure of the project
- .gitignore --> git ignore file
- -gitattributes --> git attributes file
- train.csv --> training dataset
- test.csv --> testing dataset
- titanic.ipynb --> code
- visuals.py --> module containing helper visualization functions
- predictions_titanic.csv --> predictions of survival for observations in test.csv

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook titanic.ipynb
```  
or
```bash
jupyter notebook titanic.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The titanic  dataset consists of approximately 891 data points, with each datapoint having 11 features. 

**Features**
- `passenger_Id`: counter
- `name`: name of passenger
- `age`: Age
- `sex`: Sex (Female, Male)
- `Pclass`:	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
- `sibsp`:	# of siblings / spouses aboard the Titanic	
- `parch`:	# of parents / children aboard the Titanic	
- `ticket`:	Ticket number	
- `fare`:	Passenger fare	
- `cabin`:	Cabin number	
- `embarked`:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

**Target Variable**
- `Survived`: (1, 0)
