#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point


# In[8]:


data = pd.read_csv("22070243024 insurance.csv")
data.head()


# In[9]:


data.info()


# In[10]:


### Connecting to postgrsql using psycopg2


# In[ ]:


import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# connect to the PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="121314"
)

# create a cursor object
cur = conn.cursor()

# drop the medical_costs table if it exists
cur.execute("DROP TABLE IF EXISTS medical_costs")

# create the medical_costs table
cur.execute("""
    CREATE TABLE medical_costs (
        age INTEGER,
        sex TEXT,
        bmi FLOAT,
        children INTEGER,
        smoker TEXT,
        region TEXT,
        charges FLOAT
    )
""")

# read the data into a pandas DataFrame
df = pd.read_csv("22070243024 insurance.csv")

# insert the data into the medical_costs table using execute_values
execute_values(cur, """
    INSERT INTO medical_costs (age, sex, bmi, children, smoker, region, charges)
    VALUES %s
""", df.values)

# commit the changes and close the connection
conn.commit()
 


# In[191]:


data=pd.read_sql("select * from medical_costs",conn)
data


# In[192]:


data['region'].value_counts().sort_values()


# In[193]:


data['children'].value_counts().sort_values()


# In[194]:


clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
                 'smoker': {'no': 0 , 'yes' : 1},
                   'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}
               }
data_copy = data.copy()
data_copy.replace(clean_data, inplace=True)


# In[195]:


data_copy.describe()


# In[196]:


corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,cmap='BuPu',annot=True,fmt=".2f",ax=ax)
plt.title("Dependencies of Medical Charges")
plt.show()


# In[197]:


print(data['sex'].value_counts().sort_values()) 
print(data['smoker'].value_counts().sort_values())
print(data['region'].value_counts().sort_values())


# In[198]:


###visualization


# In[199]:


plt.figure(figsize=(12,9))
plt.title('Age vs Charge')
sns.barplot(x='age',y='charges',data=data_copy,palette='husl')


# In[164]:


region_counts = data['region'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%')
plt.title('Region vs Charge')
plt.show()


# In[165]:


plt.figure(figsize=(7,5))
sns.scatterplot(x='bmi',y='charges',hue='sex',data=data_copy,palette='Reds')
plt.title('BMI VS Charge')


# In[166]:


plt.figure(figsize=(10,7))
plt.title('Smoker vs Charge')
sns.barplot(x='smoker',y='charges',data=data,palette='Blues',hue='sex')


# In[167]:


plt.figure(figsize=(10,7))
plt.title('Sex vs Charges')
sns.barplot(x='sex',y='charges',data=data,palette='Set1')


# ### Plotting Skew and Kurtosis

# In[168]:


print('Printing Skewness and Kurtosis for all columns')
print()
for col in list(data_copy.columns):
    print('{0} : Skewness {1:.3f} and  Kurtosis {2:.3f}'.format(col,data_copy[col].skew(),data_copy[col].kurt()))


# In[169]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['age'])
plt.title('Plot for Age')
plt.xlabel('Age')
plt.ylabel('Count')


# In[170]:


plt.figure(figsize=(10,7))
sns.distplot(data['bmi'])
plt.title('Plot for BMI')
plt.xlabel('BMI')
plt.ylabel('Count')


# In[171]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['charges'])
plt.title('Plot for charges')
plt.xlabel('charges')
plt.ylabel('Count')


# ### scaling BMI and Charges 

# In[172]:


from sklearn.preprocessing import StandardScaler
data_pre = data_copy.copy()

tempBmi = data_pre.bmi
tempBmi = tempBmi.values.reshape(-1,1)
data_pre['bmi'] = StandardScaler().fit_transform(tempBmi)

tempAge = data_pre.age
tempAge = tempAge.values.reshape(-1,1)
data_pre['age'] = StandardScaler().fit_transform(tempAge)

tempCharges = data_pre.charges
tempCharges = tempCharges.values.reshape(-1,1)
data_pre['charges'] = StandardScaler().fit_transform(tempCharges)

data_pre.head()


# In[173]:


X = data_pre.drop('charges',axis=1).values
y = data_pre['charges'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print('Size of X_train : ', X_train.shape)
print('Size of y_train : ', y_train.shape)
print('Size of X_test : ', X_test.shape)
print('Size of Y_test : ', y_test.shape)


# In[174]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


# ## Linear Regression

# In[175]:


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)


# In[176]:


y_pred_linear_reg_train = linear_reg.predict(X_train)
r2_score_linear_reg_train = r2_score(y_train, y_pred_linear_reg_train)

y_pred_linear_reg_test = linear_reg.predict(X_test)
r2_score_linear_reg_test = r2_score(y_test, y_pred_linear_reg_test)

rmse_linear = (np.sqrt(mean_squared_error(y_test, y_pred_linear_reg_test)))

print('R2_score (train) : {0:.3f}'.format(r2_score_linear_reg_train))
print('R2_score (test) : {0:.3f}'.format(r2_score_linear_reg_test))
print('RMSE : {0:.3f}'.format(rmse_linear))


# In[177]:


y_pred_linear_reg = linear_reg.predict(X_test)

# calculate R2 score and RMSE
r2_score_linear_reg = r2_score(y_test, y_pred_linear_reg)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear_reg))
plt.scatter(y_test, y_pred_linear_reg)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Linear Regression: Actual vs Predicted Values')


# ## Support Vector Machine (Regression)

# In[178]:


svr = SVR(C=10, gamma=0.1, tol=0.0001)
svr.fit(X_train_scaled, y_train_scaled.ravel())

y_pred_svr_train = svr.predict(X_train_scaled)
r2_score_svr_train = r2_score(y_train_scaled, y_pred_svr_train)

y_pred_svr_test = svr.predict(X_test_scaled)
r2_score_svr_test = r2_score(y_test_scaled, y_pred_svr_test)

rmse_svr = (np.sqrt(mean_squared_error(y_test_scaled, y_pred_svr_test)))

print('R2_score (train) : {0:.3f}'.format(r2_score_svr_train))
print('R2 score (test) : {0:.3f}'.format(r2_score_svr_test))
print('RMSE : {0:.3f}'.format(rmse_svr))


# In[179]:


plt.scatter(y_test_scaled, y_pred_svr_test)
plt.plot([-2,2],[-2,2],ls='dashdot',c='r')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# In[180]:


### Random Forest Regressior


# In[181]:


# create a RandomForestRegressor object
rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7, n_estimators=1200)

rf_reg.fit(X_train_scaled, y_train_scaled.ravel())

y_pred_rf_train = rf_reg.predict(X_train_scaled)
r2_score_rf_train = r2_score(y_train, y_pred_rf_train)

y_pred_rf_test = rf_reg.predict(X_test_scaled)
r2_score_rf_test = r2_score(y_test_scaled, y_pred_rf_test)
rmse_rf = np.sqrt(mean_squared_error(y_test_scaled, y_pred_rf_test))

print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train))
print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test))
print('RMSE : {0:.3f}'.format(rmse_rf))


# In[182]:


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_rf_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual charges')
plt.ylabel('Predicted charges')
plt.title('Random Forest Regression')
plt.show()


# In[185]:


### Comparing the performence of different models


# In[183]:


r2_scores = [r2_score_linear_reg_test, r2_score_svr_test, r2_score_rf_test]
rmses = [rmse_linear, rmse_svr, rmse_rf]

plt.subplot(1, 2, 1)
plt.bar(['Linear Regression', 'SVR', 'Random Forest'], r2_scores)
plt.title('R2 Scores')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.bar(['Linear Regression', 'SVR', 'Random Forest'], rmses)
plt.title('RMSEs')

plt.show()


# In[184]:


conn.close()


# In[28]:


# read the CSV file into a pandas DataFrame
df = pd.read_csv('ran.csv')

# create a list of shapely Point objects from the lon and lat columns of the DataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

# create a geopandas GeoDataFrame from the DataFrame and the geometry
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# load the shapefile
shapefile = gpd.read_file("C:\\Users\\gamin\\Downloads\\Us shp\\cb_2014_us_nation_5m.shp")

# plot the shapefile and points
fig, ax = plt.subplots(figsize=(30, 9))
shapefile.plot(ax=ax, alpha=0.5, edgecolor='k')
gdf.plot(ax=ax, color='red', markersize=5)

# set the axis labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Random Coordinates within USA")

# show the plot
plt.show()


# In[20]:


# read the CSV file into a GeoDataFrame
df = gpd.read_file("ran.csv", GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")

# convert latitude and longitude to a Point geometry
df["geometry"] = gpd.points_from_xy(df["longitude"], df["latitude"])

# set the coordinate reference system (CRS) to WGS 84 (EPSG:4326)
df.crs = "EPSG:4326"

# plot the points
ax = df.plot(markersize=5)

# set the axis labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("")

# show the plot
plt.show()

