# interpreting results

import csv
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sb

rows = []
with open("main.csv","r") as f: 
  csvreader = csv.reader(f) 
  for row in csvreader:
    rows.append(row)

headers = row[0]
planet_data_rows = row[1:]

planet_masses = []
planet_radiuses = []
planet_names = []
for planet_data in planet_data_rows:
    planet_masses.append(planet_data[3]) 
    planet_radiuses.append(planet_data[7])
    planet_names.append(planet_data[11]) 

planet_gravity = []
for index, value in enumerate(planet_names):
    gravity = (float(planet_masses[index])*1.989e+30) / (float(planet_radiuses[index])*float(planet_radiuses[index])*.989e+30)
    planet_gravity.append(gravity)

fig = px.scatter(x=planet_radiuses, y=planet_masses, size=planet_gravity, hover_data=[planet_names])
fig.show()

X = []
for index, planet_mass in enumerate(planet_masses): 
  temp_list = [
                  planet_radiuses[index],
                  planet_mass
              ]
  X.append(temp_list)

wcss = []
for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42) 
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sb.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

gravity_planets_range = []
for index, gravity in enumerate(planet_gravity):
  if gravity<300 and gravity>150: 
    gravity_planets_range.append(planet_data_rows[index])

suitable_planets = []
for planet_data in gravity_planets_range: 
  if planet_data[2] <= 100:
    suitable_planets.append(planet_data)

print(len(suitable_planets))

planet_distance = []
for planet_data in planet_data_rows:
    planet_distance.append(planet_data[8]) 

graph1 = px.bar(x = planet_names, y = planet_mass)
graph2 = px.bar(x = planet_names, y = planet_radiuses)
graph3 = px.bar(x = planet_names, y = planet_gravity)
graph4 = px.bar(x = planet_names, y = planet_distance)

graph1.show()
graph2.show()
graph3.show()
graph4.show()



