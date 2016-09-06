# MLforML
Trying to use different evolutionary algorithms to define the most optimized machine learning pipeline for a problem. It has been inspired from Randy Olson's work: http://www.randalolson.com/2015/11/15/introducing-tpot-the-data-science-assistant/

# Methodology
The basic idea that I've started with is to have an automated system that runs as many feature selection, regression and classification techniques on the data as possible. 
Step by step how I wish to proceed on this project are enumerated below:
 - Add as many techniques as possible into it.
 - Align options for user attributes as input, if the user wishes to run some specific tehcniques on the data.
 - Make the whole system smart by using PSO [can use any other, as of now it is my first priority]. So that the whole system can adopt and see which of the combination of different texhniques can be used to produce the most suitable model for the data.
 
# Basic flow
- Stratification
- Feature selection
- Model building
- - Change of cost function (for Regression and classification separately)
- - Cross-validation
- Model Selection
- All of these techniques have to be iterated for different combinations of techniques, brute in the smarting but eventually it should be smart selection of the combinations.

# Files
- MLforML.py is the main file. 
- MLforML.ipynb for playing around with data in-between while model building and for visualization. 

# Library Pre-requisites
- Scikit (scipy - sklearn)
- Pandas
- numpy
- seaborn
