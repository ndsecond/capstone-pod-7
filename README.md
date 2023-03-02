# Group One Final Notebook

## Import all necessary Libraries


```python
# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
%matplotlib inline
import scipy.stats as stats
from math import sqrt
```


```python
tmdb = pd.read_csv("data/zippedData/tmdb.movies.csv.gz")
tmdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>genre_ids</th>
      <th>id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>popularity</th>
      <th>release_date</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[12, 14, 10751]</td>
      <td>12444</td>
      <td>en</td>
      <td>Harry Potter and the Deathly Hallows: Part 1</td>
      <td>33.533</td>
      <td>2010-11-19</td>
      <td>Harry Potter and the Deathly Hallows: Part 1</td>
      <td>7.7</td>
      <td>10788</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[14, 12, 16, 10751]</td>
      <td>10191</td>
      <td>en</td>
      <td>How to Train Your Dragon</td>
      <td>28.734</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>7.7</td>
      <td>7610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[12, 28, 878]</td>
      <td>10138</td>
      <td>en</td>
      <td>Iron Man 2</td>
      <td>28.515</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>6.8</td>
      <td>12368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[16, 35, 10751]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28.005</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>7.9</td>
      <td>10174</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[28, 878, 12]</td>
      <td>27205</td>
      <td>en</td>
      <td>Inception</td>
      <td>27.920</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>8.3</td>
      <td>22186</td>
    </tr>
  </tbody>
</table>
</div>




```python
tn_budgets = pd.read_csv("data/zippedData/tn.movie_budgets.csv.gz")
tn_budgets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>release_date</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Dec 18, 2009</td>
      <td>Avatar</td>
      <td>$425,000,000</td>
      <td>$760,507,625</td>
      <td>$2,776,345,279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>May 20, 2011</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>$410,600,000</td>
      <td>$241,063,875</td>
      <td>$1,045,663,875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Jun 7, 2019</td>
      <td>Dark Phoenix</td>
      <td>$350,000,000</td>
      <td>$42,762,350</td>
      <td>$149,762,350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>May 1, 2015</td>
      <td>Avengers: Age of Ultron</td>
      <td>$330,600,000</td>
      <td>$459,005,868</td>
      <td>$1,403,013,963</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Dec 15, 2017</td>
      <td>Star Wars Ep. VIII: The Last Jedi</td>
      <td>$317,000,000</td>
      <td>$620,181,382</td>
      <td>$1,316,721,747</td>
    </tr>
  </tbody>
</table>
</div>



To clean this dataframe, the only things to do would be convert production_budget, domestic_gross, and worldwide_gross to integer data types and change release_date to datetime data type.

### Data Cleaning


```python
# change release_date column from str to datetime 
tn_budgets['release_date'] = pd.to_datetime(tn_budgets['release_date'])

# cleaning the production_budget column of dollar signs and commas and changing data type from string to int
tn_budgets['production_budget'] = tn_budgets['production_budget'].str.replace('$','')
tn_budgets['production_budget'] = tn_budgets['production_budget'].str.replace(',','')
tn_budgets = tn_budgets.astype({'production_budget': 'int64'})

# cleaning the domestic_gross column of dollar signs and commas and changing data type from string to int
tn_budgets['domestic_gross'] = tn_budgets['domestic_gross'].str.replace('$','')
tn_budgets['domestic_gross'] = tn_budgets['domestic_gross'].str.replace(',','')
tn_budgets = tn_budgets.astype({'domestic_gross': 'int64'})

# cleaning the worldwide_gross column of dollar signs and commas and changing data type from string to int
tn_budgets['worldwide_gross'] = tn_budgets['worldwide_gross'].str.replace('$','')
tn_budgets['worldwide_gross'] = tn_budgets['worldwide_gross'].str.replace(',','')
tn_budgets = tn_budgets.astype({'worldwide_gross': 'int64'})

# Find the net revenue and assigning the values to the new column named Net Revenue 
tn_budgets['Net Revenue'] = tn_budgets['worldwide_gross'] - tn_budgets['production_budget']
tn_budgets.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5782.000000</td>
      <td>5.782000e+03</td>
      <td>5.782000e+03</td>
      <td>5.782000e+03</td>
      <td>5.782000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>50.372363</td>
      <td>3.158776e+07</td>
      <td>4.187333e+07</td>
      <td>9.148746e+07</td>
      <td>5.989970e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>28.821076</td>
      <td>4.181208e+07</td>
      <td>6.824060e+07</td>
      <td>1.747200e+08</td>
      <td>1.460889e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.100000e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.002376e+08</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.000000</td>
      <td>5.000000e+06</td>
      <td>1.429534e+06</td>
      <td>4.125415e+06</td>
      <td>-2.189071e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>1.700000e+07</td>
      <td>1.722594e+07</td>
      <td>2.798445e+07</td>
      <td>8.550286e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.000000</td>
      <td>4.000000e+07</td>
      <td>5.234866e+07</td>
      <td>9.764584e+07</td>
      <td>6.096850e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>4.250000e+08</td>
      <td>9.366622e+08</td>
      <td>2.776345e+09</td>
      <td>2.351345e+09</td>
    </tr>
  </tbody>
</table>
</div>



___
# Genre Analysis
---


```python
genre_df = pd.merge(tmdb, tn_budgets, left_on=['title'], right_on=['movie'])
genre_df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2385 entries, 0 to 2384
    Data columns (total 17 columns):
     #   Column             Non-Null Count  Dtype         
    ---  ------             --------------  -----         
     0   Unnamed: 0         2385 non-null   int64         
     1   genre_ids          2385 non-null   object        
     2   id_x               2385 non-null   int64         
     3   original_language  2385 non-null   object        
     4   original_title     2385 non-null   object        
     5   popularity         2385 non-null   float64       
     6   release_date_x     2385 non-null   object        
     7   title              2385 non-null   object        
     8   vote_average       2385 non-null   float64       
     9   vote_count         2385 non-null   int64         
     10  id_y               2385 non-null   int64         
     11  release_date_y     2385 non-null   datetime64[ns]
     12  movie              2385 non-null   object        
     13  production_budget  2385 non-null   int64         
     14  domestic_gross     2385 non-null   int64         
     15  worldwide_gross    2385 non-null   int64         
     16  Net Revenue        2385 non-null   int64         
    dtypes: datetime64[ns](1), float64(2), int64(8), object(6)
    memory usage: 335.4+ KB


We see that we have shrunk the dataset to 2385 but believe that to be sufficient enough to conduct further analysis.




```python
genre_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>genre_ids</th>
      <th>id_x</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>popularity</th>
      <th>release_date_x</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>id_y</th>
      <th>release_date_y</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[14, 12, 16, 10751]</td>
      <td>10191</td>
      <td>en</td>
      <td>How to Train Your Dragon</td>
      <td>28.734</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>7.7</td>
      <td>7610</td>
      <td>30</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>165000000</td>
      <td>217581232</td>
      <td>494870992</td>
      <td>329870992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[12, 28, 878]</td>
      <td>10138</td>
      <td>en</td>
      <td>Iron Man 2</td>
      <td>28.515</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>6.8</td>
      <td>12368</td>
      <td>15</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>170000000</td>
      <td>312433331</td>
      <td>621156389</td>
      <td>451156389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[16, 35, 10751]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28.005</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>7.9</td>
      <td>10174</td>
      <td>37</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>30000000</td>
      <td>191796233</td>
      <td>364545516</td>
      <td>334545516</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2473</td>
      <td>[16, 35, 10751]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28.005</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>7.9</td>
      <td>10174</td>
      <td>37</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>30000000</td>
      <td>191796233</td>
      <td>364545516</td>
      <td>334545516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[28, 878, 12]</td>
      <td>27205</td>
      <td>en</td>
      <td>Inception</td>
      <td>27.920</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>8.3</td>
      <td>22186</td>
      <td>38</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>160000000</td>
      <td>292576195</td>
      <td>835524642</td>
      <td>675524642</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2380</th>
      <td>26323</td>
      <td>[]</td>
      <td>509316</td>
      <td>en</td>
      <td>The Box</td>
      <td>0.600</td>
      <td>2018-03-04</td>
      <td>The Box</td>
      <td>8.0</td>
      <td>1</td>
      <td>66</td>
      <td>2009-11-06</td>
      <td>The Box</td>
      <td>25000000</td>
      <td>15051977</td>
      <td>34356760</td>
      <td>9356760</td>
    </tr>
    <tr>
      <th>2381</th>
      <td>26425</td>
      <td>[10402]</td>
      <td>509306</td>
      <td>en</td>
      <td>The Box</td>
      <td>0.600</td>
      <td>2018-03-04</td>
      <td>The Box</td>
      <td>6.0</td>
      <td>1</td>
      <td>66</td>
      <td>2009-11-06</td>
      <td>The Box</td>
      <td>25000000</td>
      <td>15051977</td>
      <td>34356760</td>
      <td>9356760</td>
    </tr>
    <tr>
      <th>2382</th>
      <td>26092</td>
      <td>[35, 16]</td>
      <td>546674</td>
      <td>en</td>
      <td>Enough</td>
      <td>0.719</td>
      <td>2018-03-22</td>
      <td>Enough</td>
      <td>8.7</td>
      <td>3</td>
      <td>68</td>
      <td>2002-05-24</td>
      <td>Enough</td>
      <td>38000000</td>
      <td>39177215</td>
      <td>50970660</td>
      <td>12970660</td>
    </tr>
    <tr>
      <th>2383</th>
      <td>26322</td>
      <td>[]</td>
      <td>513161</td>
      <td>en</td>
      <td>Undiscovered</td>
      <td>0.600</td>
      <td>2018-04-07</td>
      <td>Undiscovered</td>
      <td>8.0</td>
      <td>1</td>
      <td>7</td>
      <td>2005-08-26</td>
      <td>Undiscovered</td>
      <td>9000000</td>
      <td>1069318</td>
      <td>1069318</td>
      <td>-7930682</td>
    </tr>
    <tr>
      <th>2384</th>
      <td>26508</td>
      <td>[16]</td>
      <td>514492</td>
      <td>en</td>
      <td>Jaws</td>
      <td>0.600</td>
      <td>2018-05-29</td>
      <td>Jaws</td>
      <td>0.0</td>
      <td>1</td>
      <td>41</td>
      <td>1975-06-20</td>
      <td>Jaws</td>
      <td>12000000</td>
      <td>260000000</td>
      <td>470700000</td>
      <td>458700000</td>
    </tr>
  </tbody>
</table>
<p>2385 rows × 17 columns</p>
</div>




```python
# We notice some duplicates and choose to remove those.
genre_df = genre_df[genre_df['title'] != 'Home']
genre_df = genre_df.drop_duplicates(subset='title')

```


```python
genre_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>genre_ids</th>
      <th>id_x</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>popularity</th>
      <th>release_date_x</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>id_y</th>
      <th>release_date_y</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[14, 12, 16, 10751]</td>
      <td>10191</td>
      <td>en</td>
      <td>How to Train Your Dragon</td>
      <td>28.734</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>7.7</td>
      <td>7610</td>
      <td>30</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>165000000</td>
      <td>217581232</td>
      <td>494870992</td>
      <td>329870992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[12, 28, 878]</td>
      <td>10138</td>
      <td>en</td>
      <td>Iron Man 2</td>
      <td>28.515</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>6.8</td>
      <td>12368</td>
      <td>15</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>170000000</td>
      <td>312433331</td>
      <td>621156389</td>
      <td>451156389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[16, 35, 10751]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28.005</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>7.9</td>
      <td>10174</td>
      <td>37</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>30000000</td>
      <td>191796233</td>
      <td>364545516</td>
      <td>334545516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[28, 878, 12]</td>
      <td>27205</td>
      <td>en</td>
      <td>Inception</td>
      <td>27.920</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>8.3</td>
      <td>22186</td>
      <td>38</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>160000000</td>
      <td>292576195</td>
      <td>835524642</td>
      <td>675524642</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>[12, 14, 10751]</td>
      <td>32657</td>
      <td>en</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>26.691</td>
      <td>2010-02-11</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>6.1</td>
      <td>4229</td>
      <td>17</td>
      <td>2010-02-12</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>95000000</td>
      <td>88768303</td>
      <td>223050874</td>
      <td>128050874</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>25825</td>
      <td>[28, 878]</td>
      <td>448764</td>
      <td>en</td>
      <td>Molly</td>
      <td>1.400</td>
      <td>2018-09-25</td>
      <td>Molly</td>
      <td>5.8</td>
      <td>5</td>
      <td>81</td>
      <td>1999-10-22</td>
      <td>Molly</td>
      <td>21000000</td>
      <td>17396</td>
      <td>17396</td>
      <td>-20982604</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>26040</td>
      <td>[]</td>
      <td>509314</td>
      <td>en</td>
      <td>The Box</td>
      <td>0.840</td>
      <td>2018-03-04</td>
      <td>The Box</td>
      <td>8.0</td>
      <td>1</td>
      <td>66</td>
      <td>2009-11-06</td>
      <td>The Box</td>
      <td>25000000</td>
      <td>15051977</td>
      <td>34356760</td>
      <td>9356760</td>
    </tr>
    <tr>
      <th>2382</th>
      <td>26092</td>
      <td>[35, 16]</td>
      <td>546674</td>
      <td>en</td>
      <td>Enough</td>
      <td>0.719</td>
      <td>2018-03-22</td>
      <td>Enough</td>
      <td>8.7</td>
      <td>3</td>
      <td>68</td>
      <td>2002-05-24</td>
      <td>Enough</td>
      <td>38000000</td>
      <td>39177215</td>
      <td>50970660</td>
      <td>12970660</td>
    </tr>
    <tr>
      <th>2383</th>
      <td>26322</td>
      <td>[]</td>
      <td>513161</td>
      <td>en</td>
      <td>Undiscovered</td>
      <td>0.600</td>
      <td>2018-04-07</td>
      <td>Undiscovered</td>
      <td>8.0</td>
      <td>1</td>
      <td>7</td>
      <td>2005-08-26</td>
      <td>Undiscovered</td>
      <td>9000000</td>
      <td>1069318</td>
      <td>1069318</td>
      <td>-7930682</td>
    </tr>
    <tr>
      <th>2384</th>
      <td>26508</td>
      <td>[16]</td>
      <td>514492</td>
      <td>en</td>
      <td>Jaws</td>
      <td>0.600</td>
      <td>2018-05-29</td>
      <td>Jaws</td>
      <td>0.0</td>
      <td>1</td>
      <td>41</td>
      <td>1975-06-20</td>
      <td>Jaws</td>
      <td>12000000</td>
      <td>260000000</td>
      <td>470700000</td>
      <td>458700000</td>
    </tr>
  </tbody>
</table>
<p>1923 rows × 17 columns</p>
</div>



We are ultimately left with 1923 rows in the dataset which we still believe to be ok.

We decide we want our target variable to be Net Revenue, so we subtract production budget from worldwide gross. We make the assumption that in order to produce the movie, all of the production budget was used and ONLY the production budget. In other words, no more and no less than the production budget was spent in the creation of a movie.


```python
genre_df['Net Revenue'] = genre_df['worldwide_gross'] - genre_df['production_budget']
genre_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>genre_ids</th>
      <th>id_x</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>popularity</th>
      <th>release_date_x</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>id_y</th>
      <th>release_date_y</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[14, 12, 16, 10751]</td>
      <td>10191</td>
      <td>en</td>
      <td>How to Train Your Dragon</td>
      <td>28.734</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>7.7</td>
      <td>7610</td>
      <td>30</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>165000000</td>
      <td>217581232</td>
      <td>494870992</td>
      <td>329870992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[12, 28, 878]</td>
      <td>10138</td>
      <td>en</td>
      <td>Iron Man 2</td>
      <td>28.515</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>6.8</td>
      <td>12368</td>
      <td>15</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>170000000</td>
      <td>312433331</td>
      <td>621156389</td>
      <td>451156389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[16, 35, 10751]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28.005</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>7.9</td>
      <td>10174</td>
      <td>37</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>30000000</td>
      <td>191796233</td>
      <td>364545516</td>
      <td>334545516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[28, 878, 12]</td>
      <td>27205</td>
      <td>en</td>
      <td>Inception</td>
      <td>27.920</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>8.3</td>
      <td>22186</td>
      <td>38</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>160000000</td>
      <td>292576195</td>
      <td>835524642</td>
      <td>675524642</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>[12, 14, 10751]</td>
      <td>32657</td>
      <td>en</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>26.691</td>
      <td>2010-02-11</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>6.1</td>
      <td>4229</td>
      <td>17</td>
      <td>2010-02-12</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>95000000</td>
      <td>88768303</td>
      <td>223050874</td>
      <td>128050874</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We realize the genre codes are in long strings. We remove the brackets and commas and split the codes by " " into a list.

genre_df['genre_ids'] = genre_df['genre_ids'].str.replace("[", "")
genre_df['genre_ids'] = genre_df['genre_ids'].str.replace("]", "")
genre_df['genre_ids'] = genre_df['genre_ids'].str.replace(",", "")
genre_df['genre_ids'] = genre_df['genre_ids'].apply(lambda x: x.split(" "))
genre_df['genre_ids'][0]
```




    ['14', '12', '16', '10751']



We found a key on the TMDB website that says what genre each number code relate to. Below we use a for loop to change them in the dataframe.


```python
for lst in genre_df['genre_ids']:
    for i in range(len(lst)):
            if lst[i] == '12':
                lst[i] = 'Adventure'
            elif lst[i] == '14':
                lst[i] = 'Fantasy'
            elif lst[i] == '28':
                lst[i] = 'Action'
            elif lst[i] == '16':
                lst[i] = 'Animation'
            elif lst[i] == '35':
                lst[i] = 'Comedy'
            elif lst[i] == '80':
                lst[i] = 'Crime'
            elif lst[i] == '99':
                lst[i] = 'Documentary'
            elif lst[i] == '18':
                lst[i] = 'Drama'
            elif lst[i] == '10751':
                lst[i] = 'Family'
            elif lst[i] == '36':
                lst[i] = 'History'
            elif lst[i] == '27':
                lst[i] = 'Horror'
            elif lst[i] == '10402':
                lst[i] = 'Music'
            elif lst[i] == '9648':
                lst[i] = 'Mystery'
            elif lst[i] == '10749':
                lst[i] = 'Romance'
            elif lst[i] == '878':
                lst[i] = 'SciFi'
            elif lst[i] == '10770':
                lst[i] = 'TV Movie'
            elif lst[i] == '53':
                lst[i] = 'Thriller'
            elif lst[i] == '10752':
                lst[i] = 'War'
            elif lst[i] == '37':
                lst[i] = 'Western'
```


```python
genre_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>genre_ids</th>
      <th>id_x</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>popularity</th>
      <th>release_date_x</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>id_y</th>
      <th>release_date_y</th>
      <th>movie</th>
      <th>production_budget</th>
      <th>domestic_gross</th>
      <th>worldwide_gross</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[Fantasy, Adventure, Animation, Family]</td>
      <td>10191</td>
      <td>en</td>
      <td>How to Train Your Dragon</td>
      <td>28.734</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>7.7</td>
      <td>7610</td>
      <td>30</td>
      <td>2010-03-26</td>
      <td>How to Train Your Dragon</td>
      <td>165000000</td>
      <td>217581232</td>
      <td>494870992</td>
      <td>329870992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>[Adventure, Action, SciFi]</td>
      <td>10138</td>
      <td>en</td>
      <td>Iron Man 2</td>
      <td>28.515</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>6.8</td>
      <td>12368</td>
      <td>15</td>
      <td>2010-05-07</td>
      <td>Iron Man 2</td>
      <td>170000000</td>
      <td>312433331</td>
      <td>621156389</td>
      <td>451156389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>[Animation, Comedy, Family]</td>
      <td>862</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>28.005</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>7.9</td>
      <td>10174</td>
      <td>37</td>
      <td>1995-11-22</td>
      <td>Toy Story</td>
      <td>30000000</td>
      <td>191796233</td>
      <td>364545516</td>
      <td>334545516</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[Action, SciFi, Adventure]</td>
      <td>27205</td>
      <td>en</td>
      <td>Inception</td>
      <td>27.920</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>8.3</td>
      <td>22186</td>
      <td>38</td>
      <td>2010-07-16</td>
      <td>Inception</td>
      <td>160000000</td>
      <td>292576195</td>
      <td>835524642</td>
      <td>675524642</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>32657</td>
      <td>en</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>26.691</td>
      <td>2010-02-11</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>6.1</td>
      <td>4229</td>
      <td>17</td>
      <td>2010-02-12</td>
      <td>Percy Jackson &amp; the Olympians: The Lightning T...</td>
      <td>95000000</td>
      <td>88768303</td>
      <td>223050874</td>
      <td>128050874</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2376</th>
      <td>25825</td>
      <td>[Action, SciFi]</td>
      <td>448764</td>
      <td>en</td>
      <td>Molly</td>
      <td>1.400</td>
      <td>2018-09-25</td>
      <td>Molly</td>
      <td>5.8</td>
      <td>5</td>
      <td>81</td>
      <td>1999-10-22</td>
      <td>Molly</td>
      <td>21000000</td>
      <td>17396</td>
      <td>17396</td>
      <td>-20982604</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>26040</td>
      <td>[]</td>
      <td>509314</td>
      <td>en</td>
      <td>The Box</td>
      <td>0.840</td>
      <td>2018-03-04</td>
      <td>The Box</td>
      <td>8.0</td>
      <td>1</td>
      <td>66</td>
      <td>2009-11-06</td>
      <td>The Box</td>
      <td>25000000</td>
      <td>15051977</td>
      <td>34356760</td>
      <td>9356760</td>
    </tr>
    <tr>
      <th>2382</th>
      <td>26092</td>
      <td>[Comedy, Animation]</td>
      <td>546674</td>
      <td>en</td>
      <td>Enough</td>
      <td>0.719</td>
      <td>2018-03-22</td>
      <td>Enough</td>
      <td>8.7</td>
      <td>3</td>
      <td>68</td>
      <td>2002-05-24</td>
      <td>Enough</td>
      <td>38000000</td>
      <td>39177215</td>
      <td>50970660</td>
      <td>12970660</td>
    </tr>
    <tr>
      <th>2383</th>
      <td>26322</td>
      <td>[]</td>
      <td>513161</td>
      <td>en</td>
      <td>Undiscovered</td>
      <td>0.600</td>
      <td>2018-04-07</td>
      <td>Undiscovered</td>
      <td>8.0</td>
      <td>1</td>
      <td>7</td>
      <td>2005-08-26</td>
      <td>Undiscovered</td>
      <td>9000000</td>
      <td>1069318</td>
      <td>1069318</td>
      <td>-7930682</td>
    </tr>
    <tr>
      <th>2384</th>
      <td>26508</td>
      <td>[Animation]</td>
      <td>514492</td>
      <td>en</td>
      <td>Jaws</td>
      <td>0.600</td>
      <td>2018-05-29</td>
      <td>Jaws</td>
      <td>0.0</td>
      <td>1</td>
      <td>41</td>
      <td>1975-06-20</td>
      <td>Jaws</td>
      <td>12000000</td>
      <td>260000000</td>
      <td>470700000</td>
      <td>458700000</td>
    </tr>
  </tbody>
</table>
<p>1923 rows × 17 columns</p>
</div>



To make the genre column easier to analyze, we use the .explode() function to create a unique row for each genre in the list. For instance, if a movie has 4 genres listed then it will now have 4 rows, each with a different one of the listed genres.


```python
genre_df_exploded = genre_df.explode('genre_ids')

#remove rows with empty genres 
genre_df_exploded = genre_df_exploded[genre_df_exploded['genre_ids'] != '']
```

We can now use groupby to find the average net revenue by genre. Note, if a movie has multiple rows, its revenue will be considered in multiple categories.




```python
average_group = genre_df_exploded.groupby(['genre_ids'])['Net Revenue'].mean().sort_values(ascending=False).reset_index()
average_group

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_ids</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Animation</td>
      <td>2.439302e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adventure</td>
      <td>2.421081e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fantasy</td>
      <td>2.058034e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Family</td>
      <td>1.920874e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SciFi</td>
      <td>1.782768e+08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Action</td>
      <td>1.589646e+08</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Comedy</td>
      <td>8.484031e+07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Thriller</td>
      <td>6.275487e+07</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Crime</td>
      <td>6.079877e+07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Music</td>
      <td>5.508409e+07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>War</td>
      <td>5.469261e+07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Mystery</td>
      <td>5.270099e+07</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Romance</td>
      <td>5.111473e+07</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Drama</td>
      <td>4.583196e+07</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Western</td>
      <td>4.506243e+07</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Horror</td>
      <td>4.048204e+07</td>
    </tr>
    <tr>
      <th>16</th>
      <td>History</td>
      <td>3.560990e+07</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Documentary</td>
      <td>3.013668e+07</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TV Movie</td>
      <td>2.918712e+07</td>
    </tr>
  </tbody>
</table>
</div>




```python
#groupby genre on count
count_group = genre_df_exploded.groupby('genre_ids')['Net Revenue'].count().sort_values(ascending=False).reset_index()
count_group
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_ids</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Drama</td>
      <td>872</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Comedy</td>
      <td>584</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Thriller</td>
      <td>518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Action</td>
      <td>472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adventure</td>
      <td>298</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Horror</td>
      <td>258</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Crime</td>
      <td>241</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Romance</td>
      <td>233</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SciFi</td>
      <td>217</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Family</td>
      <td>187</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Fantasy</td>
      <td>178</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Mystery</td>
      <td>139</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Animation</td>
      <td>123</td>
    </tr>
    <tr>
      <th>13</th>
      <td>History</td>
      <td>70</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Documentary</td>
      <td>69</td>
    </tr>
    <tr>
      <th>15</th>
      <td>War</td>
      <td>47</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Music</td>
      <td>47</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Western</td>
      <td>24</td>
    </tr>
    <tr>
      <th>18</th>
      <td>TV Movie</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



# Final Plot


```python

#average net revenue by movie genre
ax = sns.barplot(data=average_group, x='genre_ids', y='Net Revenue', color='#990000')
ax.set_xticklabels(ax.get_xticklabels(), rotation=75);
ax.set_title('Average Net Revenue by Movie Genre', fontsize = 14, weight = 'bold')
ax.set_ylabel('Average Net Revenue (hundred millions)', weight='bold')
ax.set_xlabel('Movie Genre', weight='bold');
sns.set_style('whitegrid')
```


    
![png](output_26_0.png)
    


# Studio Analysis


```python
bom = pd.read_csv('data/zippedData/bom.movie_gross.csv.gz')
bom
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>studio</th>
      <th>domestic_gross</th>
      <th>foreign_gross</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Toy Story 3</td>
      <td>BV</td>
      <td>415000000.0</td>
      <td>652000000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice in Wonderland (2010)</td>
      <td>BV</td>
      <td>334200000.0</td>
      <td>691300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Deathly Hallows Part 1</td>
      <td>WB</td>
      <td>296000000.0</td>
      <td>664300000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inception</td>
      <td>WB</td>
      <td>292600000.0</td>
      <td>535700000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shrek Forever After</td>
      <td>P/DW</td>
      <td>238700000.0</td>
      <td>513900000</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3382</th>
      <td>The Quake</td>
      <td>Magn.</td>
      <td>6200.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3383</th>
      <td>Edward II (2018 re-release)</td>
      <td>FM</td>
      <td>4800.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3384</th>
      <td>El Pacto</td>
      <td>Sony</td>
      <td>2500.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3385</th>
      <td>The Swan</td>
      <td>Synergetic</td>
      <td>2400.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3386</th>
      <td>An Actor Prepares</td>
      <td>Grav.</td>
      <td>1700.0</td>
      <td>NaN</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>3387 rows × 5 columns</p>
</div>



### Cleaning Studio columns to remove duplicates


```python
bom['studio'].unique()
bom['studio'] = bom['studio'].str.strip('()')
bom['studio'] = bom['studio'].str.strip('(NL')
bom['studio'] = bom['studio'].str.strip()
bom['studio'] = bom['studio'].str.strip()
bom['studio'] = bom['studio'].str.replace('BV','Walt Disney')
bom['studio'] = bom['studio'].str.replace('P/DW','Pixar')
bom['studio'] = bom['studio'].str.replace('Uni.','Universial')
bom['studio'] = bom['studio'].str.replace('Par.','Paramount')
bom['studio'] = bom['studio'].str.strip()
```

### Feature Creation create year column & Join Merge Bom & Tn_budgets


```python
## Adding a year column
tn_budgets['year'] =  pd.DatetimeIndex(tn_budgets['release_date']).year
### Merging bom dataframe and Tn budgets
bom_budgets = pd.merge(tn_budgets, bom[['studio','title', 'year']],left_on=['movie','year'], right_on = ['title','year'], how = 'inner')
### Creating a column called net that calculates the difference between worldwide_gross and production budget
bom_budgets['net'] = bom_budgets['worldwide_gross'] - bom_budgets['production_budget']
```

### Groups by studio and find the top ten Studio by net profit


```python
#Reduces the amount of digits displayed
pd.set_option('display.float_format', lambda x: '%.3f' % x)
net_studio = bom_budgets.groupby("studio")['net'].mean().to_frame(name = 'average_net').reset_index()
top10_studio = net_studio.sort_values(by=['average_net'], ascending= False, na_position='first').head(10)
```

### Bargraph of top ten studio by Average net revenue


```python
sns.set_style("whitegrid")
#setting figure size 
plt.figure(figsize =(9,7))
#creating bar plot
ax_studio = sns.barplot(data=top10_studio, x='studio', y='average_net',color='#990000' )
#setting title and axis names
ax_studio.set_title('Average Net Revenue by Studio', fontsize = 14, weight = 'bold')
ax_studio.set_ylabel('Average Net Revenue (hundred millions)', fontsize = 12, weight = 'bold' )
ax_studio.set_xlabel('Studio', fontsize = 12, weight = 'bold')
ax_studio.set_xticklabels(ax_studio.get_xticklabels(), rotation=75);
plt.show()
```


    
![png](output_36_0.png)
    


### Bar Graph of top ten studios whose production cost is in the 25th interquartile range


```python
#grouping by studios by production cost
studio_cost = bom_budgets.groupby("studio")['production_budget'].mean().to_frame(name = 'average_cost').reset_index()
#creating_quartiles to find what inter quartile ranges of cost
studio_cost_qt = studio_cost['average_cost'].quantile([0.25, 0.5, 0.75])
# Selecting only studios in the 25th quartile or lower 
low_budget_studios = studio_cost[studio_cost['average_cost'] <= 6407142.857]
list_low = list(low_budget_studios['studio'])
low_cost_studios = bom_budgets[bom_budgets['studio'] == 'Viv' ]
#looping through a the list of every studio that is in the 25th quartile and appending it to a newframe
for i in list_low:
    low_cost_studios = bom_budgets[bom_budgets['studio'] ==  i].append(low_cost_studios)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
low_net_studio = low_cost_studios.groupby("studio")['net'].mean().to_frame(name = 'average_net').reset_index()
top10_low = low_net_studio.sort_values(by=['average_net'], ascending= False, na_position='first').head(10)
```


```python
sns.set_style("whitegrid")
#creating bar plot
ax_studio_low = sns.barplot(data=top10_low, x='studio', y='average_net',color='#990000' )
#setting figure size 
plt.figure(figsize =(9,7))
#setting title and axis names 
ax_studio_low.set_title('Average Net Revenue by Studio', fontsize = 14, weight = 'bold')
ax_studio_low.set_ylabel('Average Net Revenue (hundred millions)', fontsize = 12, weight = 'bold' )
ax_studio_low.set_xlabel('Studio', fontsize = 12, weight = 'bold')
ax_studio_low.set_xticklabels(ax_studio_low.get_xticklabels(), rotation=75)
plt.show()
```


    
![png](output_39_0.png)
    



    <Figure size 648x504 with 0 Axes>


# Release Date Analysis

## Hypothesis

### Alternative hypothesis is that movies released in the summer season will generate a higher net revenue than the population average
#### $H_a$ = 𝜇<𝑀
_____________________________________________________________________________________________

### Null hypothesis is that movies released in the summer season will not generate a higher net revenue than population average
#### $H_0 $ = 𝜇≥𝑀


```python
# Taking month out of release date and creating new column with the values
tn_budgets["release_month"] = tn_budgets["release_date"].dt.month 

# Creating a variable that groups the release_month and Net Revenue columns and calculated the mean of each release month
by_month = tn_budgets.groupby("release_month")["Net Revenue"].mean()
```

## Map average gross revenue by month


```python
# Plotting the data from by_month into a bar graph
by_month.plot(kind='bar', title='AVG Net Per Month', ylabel='Mean Net',
         xlabel='Release Month', figsize=(6, 5))
```




    <AxesSubplot:title={'center':'AVG Net Per Month'}, xlabel='Release Month', ylabel='Mean Net'>




    
![png](output_44_1.png)
    


 From this chart we notice that movies released in summer generate a higher Net Revenue


```python
# Create dictionary to assign month to season to properly evaluate the data
season_month = {
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Fall', 10:'Fall', 11:'Fall'}

# map through data and create new column with movie release season    
tn_budgets['release_season'] = tn_budgets["release_month"].map(season_month)
tn_budgets.sort_values(by=["release_season"])
by_season = tn_budgets.groupby("release_season")["Net Revenue"].mean().reset_index()
```


```python
# Create new dataframe that groups the release_season and Net Revenue columns to see avg net revenue
season_mean = tn_budgets.groupby("release_season")["Net Revenue"].mean().reset_index()
season_mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_season</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fall</td>
      <td>47803627.875</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spring</td>
      <td>65128830.067</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Summer</td>
      <td>76676469.389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Winter</td>
      <td>51863328.816</td>
    </tr>
  </tbody>
</table>
</div>




```python
season_count = tn_budgets.groupby("release_season")["Net Revenue"].count().reset_index()
season_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>release_season</th>
      <th>Net Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fall</td>
      <td>1552</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spring</td>
      <td>1331</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Summer</td>
      <td>1415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Winter</td>
      <td>1484</td>
    </tr>
  </tbody>
</table>
</div>




```python
import scipy.stats as stats
from math import sqrt
x_bar = season_mean["Net Revenue"][2] # sample mean of summer 
n = season_count["Net Revenue"][2] # number of sample inputs
sigma = tn_budgets["Net Revenue"].std() # sd of all inputs
mu = tn_budgets["Net Revenue"].mean() # all inputs mean 

z_value = (x_bar - mu)/(sigma/sqrt(n))
z_value
```




    4.319856231441833




```python
p_value = stats.norm.sf(z_value)
print('p_value = 0.000007806531178731228')
p_value
```

    p_value = 0.000007806531178731228





    7.806544209520419e-06




```python
# Plot out the by_season variable that allows us to evaluate average net revenue by season
ax = sns.barplot(data=by_season, x='release_season', y='Net Revenue', color='#990000', order=['Winter', 'Spring', 'Summer', 'Fall'])
ax.set_xticklabels(ax.get_xticklabels());
ax.set_title('Average Net Revenue by Release Season', fontsize=12, weight='bold')
ax.set_ylabel('Average Net Revenue (ten millions)', weight='bold')
ax.set_xlabel('Release season', weight='bold')
```




    Text(0.5, 0, 'Release season')




    
![png](output_51_1.png)
    


# Conclusion
### alpha = .05
### z score = 4.319
### p score = 0.000007806531178731228
## After running the z score and p score, we can reject the null hypothesis with 99.9 percent confidence
