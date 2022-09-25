#to reuse
import numpy as np
import pandas as pd
import json
import pickle

#load data encoded
with open('/content/one_hot_cols.json') as json_file: 
   encoded_data = json.load(json_file)


# load model
with open('water auto-classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)   

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def lg(x):
  return np.log2(x)

expected_columns = ['Classification as urban/semi-urban/rural_one-hot', 'log_Population',
       'water_source_category_one-hot', 'log_angle', '#status_id_one-hot',
       'log_days_passed', '#adm2_one-hot', 'log_dist', '#pay_one-hot',
       'water_tech_category_one-hot', 'Water demand (lpcd)_one-hot']
      
def feature_engineering():
  inp = []
  print( "Enter classification as of the area as \n 0. urban \n 1. semi-urban \n 2. rural: \n (Enter 0, 1, 2)")
  a1 = int(input())
  inp.append(a1)

  print( "Enter the population of the area: (Enter a positive integer)")
  a2 = int(input())
  inp.append(lg(a2)) #log population

  print( "Enter the water source category of the area as \n 0. well \n 1. spring \n (Enter 0, 1)")
  a3 = int(input())
  inp.append(a3)

  print( "Enter the latitude of the area:")
  a4 = int(input())
  print( "Enter the latitude of the area:")
  a5 = int(input())
  a4, a5 = cart2pol(a4, a5)

  inp.append(lg(a4 + 1000)) #log angle

  print( "Enter the status id of water source as \n 0. Yes (Functional) \n 1. No (non-functional): \n (Enter 0, 1)")
  a6 = int(input())
  inp.append(a6)

  print( "Enter the number of days passed since installation of water source: (Enter a postive integer)")
  a7 = int(input())
  inp.append(lg(a7))

  print( "Enter the local government name as")
  i = 0
  for adm in encoded_data['#adm2']:
    print( str(i) + ". " + str(adm) )
    i = i + 1
  print('(Enter a number from 0 to ' + str(i) + ')')
  print( str(i) + ". " + 'Unknown' )
  a8 = int(input())
  inp.append(a8)

  inp.append(lg(a5 +1000)) #log of distance

  print( "Enter the payment method as")
  i = 0
  for adm in encoded_data['#pay']:
    print( str(i) + ". " + str(adm) )
    i = i + 1
  print( str(i) + ". " + 'Unknown' )
  print('(Enter a number from 0 to ' + str(i) + ')')
  a9 = int(input())
  inp.append(a9)

  print( "Enter the water source category as")
  i = 0
  for adm in encoded_data['water_tech_category']:
    print( str(i) + ". " + str(adm) )
    i = i + 1
  print( str(i) + ". " + 'Unknown' )
  print('(Enter a number from 0 to ' + str(i) + ')')
  a10 = int(input())
  inp.append(a10)

  return inp



def make_prediction(input_data):

  column_names =['Classification as urban/semi-urban/rural_one-hot', 'log_Population',
        'water_source_category_one-hot', 'log_angle', '#status_id_one-hot',
        'log_days_passed', '#adm2_one-hot', 'log_dist', '#pay_one-hot',
        'water_tech_category_one-hot']

  data = input_data
  df = pd.DataFrame([data], columns = column_names)

  make_prediction = classifier.predict(df)
  print("The priority of target area is classified as: ")
  return  make_prediction  


inputed_data = feature_engineering()  
prediction = make_prediction(inputed_data)

print(prediction)