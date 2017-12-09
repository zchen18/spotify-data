---
title: Playlist Evaluation and Modelling
notebook: playlist.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}
### Building the Playlist Creation Model





    /Users/sarah/anaconda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    

First, we read in the data that was called from the Spotify API. This dataset includes 668 unique playlists from various Spotify-featured genres. For each playlist, there is playlist-specific data (followers, length, etc.) as well as aggregate data about its tracks (average danceability, tempo, etc.)



```python
playlists = pd.read_csv('playlists.csv')
playlists.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>index</th>
      <th>collaborative</th>
      <th>external_urls</th>
      <th>href</th>
      <th>id</th>
      <th>images</th>
      <th>name</th>
      <th>owner</th>
      <th>...</th>
      <th>explicit</th>
      <th>popularity</th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>energy</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>{'spotify': 'https://open.spotify.com/user/spo...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>37i9dQZF1DXcBWIGoYBM5M</td>
      <td>[{'height': 300, 'url': 'https://i.scdn.co/ima...</td>
      <td>Today's Top Hits</td>
      <td>spotify</td>
      <td>...</td>
      <td>0.368421</td>
      <td>83.631579</td>
      <td>0.195290</td>
      <td>0.670779</td>
      <td>0.672232</td>
      <td>0.170601</td>
      <td>-5.296589</td>
      <td>0.089700</td>
      <td>119.590074</td>
      <td>0.428257</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>{'spotify': 'https://open.spotify.com/user/spo...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>37i9dQZF1DX0XUsuxWHRQd</td>
      <td>[{'height': 300, 'url': 'https://i.scdn.co/ima...</td>
      <td>RapCaviar</td>
      <td>spotify</td>
      <td>...</td>
      <td>0.989130</td>
      <td>77.173913</td>
      <td>0.166482</td>
      <td>0.757576</td>
      <td>0.629641</td>
      <td>0.198623</td>
      <td>-6.440326</td>
      <td>0.221160</td>
      <td>130.062750</td>
      <td>0.430804</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
      <td>{'spotify': 'https://open.spotify.com/user/spo...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>37i9dQZF1DX4dyzvuaRJ0n</td>
      <td>[{'height': 300, 'url': 'https://i.scdn.co/ima...</td>
      <td>mint</td>
      <td>spotify</td>
      <td>...</td>
      <td>0.094737</td>
      <td>65.357895</td>
      <td>0.116051</td>
      <td>0.604547</td>
      <td>0.769189</td>
      <td>0.186685</td>
      <td>-5.135021</td>
      <td>0.067026</td>
      <td>125.485221</td>
      <td>0.390440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>{'spotify': 'https://open.spotify.com/user/spo...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>37i9dQZF1DXcF6B6QPhFDv</td>
      <td>[{'height': 300, 'url': 'https://i.scdn.co/ima...</td>
      <td>Rock This</td>
      <td>spotify</td>
      <td>...</td>
      <td>0.125000</td>
      <td>60.267857</td>
      <td>0.052656</td>
      <td>0.532304</td>
      <td>0.785554</td>
      <td>0.207500</td>
      <td>-5.404982</td>
      <td>0.062518</td>
      <td>124.610000</td>
      <td>0.519429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>4</td>
      <td>4</td>
      <td>False</td>
      <td>{'spotify': 'https://open.spotify.com/user/spo...</td>
      <td>https://api.spotify.com/v1/users/spotify/playl...</td>
      <td>37i9dQZF1DX4SBhb3fqCJd</td>
      <td>[{'height': 300, 'url': 'https://i.scdn.co/ima...</td>
      <td>Are &amp; Be</td>
      <td>spotify</td>
      <td>...</td>
      <td>0.500000</td>
      <td>64.558824</td>
      <td>0.224200</td>
      <td>0.629294</td>
      <td>0.549559</td>
      <td>0.160476</td>
      <td>-7.193765</td>
      <td>0.127618</td>
      <td>113.833868</td>
      <td>0.458021</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



### Feature Selection

Given that we have many potential predictors to work with (see the list below), we performed feature selection through exhaustive search selection. We perform feature selection within each playlist genre, given that the factors that will make a playlist successful intuitively differ between genres. We also attempted stepwise forward/backward selection, but the sample size of playlists per genre was too small to achieve satisfactory results. 



```python
#List of columns in the initial dataset 
list(playlists)
```





    ['Unnamed: 0',
     'Unnamed: 0.1',
     'index',
     'collaborative',
     'external_urls',
     'href',
     'id',
     'images',
     'name',
     'owner',
     'public',
     'snapshot_id',
     'tracks',
     'type',
     'uri',
     'Category',
     'followers',
     'length',
     'avgpopularity',
     'medianpopularity',
     'pid',
     'duration_ms',
     'explicit',
     'popularity',
     'acousticness',
     'danceability',
     'energy',
     'liveness',
     'loudness',
     'speechiness',
     'tempo',
     'valence']





```python
#Exhaustive search selection
import itertools

def exhaustive_search_selection(x, y):
    """Exhaustively search predictor combinations. .

    Parameters:
    -----------
    x : DataFrame of predictors/features
    y : response varible 
    
    
    Returns:
    -----------
    
    Dataframe of model comparisons and OLS Model with 
    lowest BIC for subset with highest R^2
    
    """
    
    # total no. of predictors
    d = x.shape[1]
    predictors = x.columns
    overall_min_bic = 10000 # A big number 
    output = dict()
    
    # Outer loop: iterate over sizes 1 .... d
    for k in range(1,d):
        
        max_r_squared = -10000 # A small number
        
        # Enumerate subsets of size ‘k’
        subsets_k = itertools.combinations(predictors, k)
        
        # Inner loop: iterate through subsets_k
        for subset in subsets_k:
            # Fit regression model using ‘subset’ and calculate R^2 
            # Keep track of subset with highest R^2
            
            features = list(subset)
            x_subset = x[features]
            
            model = OLS(y, x_subset)
            results = model.fit()
            r_squared = results.rsquared
            
            # Check if we get a higher R^2 value than than current max R^2, 
            # if so, update our best subset 
            if(r_squared > max_r_squared):
                max_r_squared = r_squared
                best_subset = features
                best_model = model
                best_formula = "y ~ {}".format(' + '.join(features))
        
        results = best_model.fit()
        bic = results.bic
        if bic < overall_min_bic:
            overall_min_bic = bic 
            best_overall_subset = best_subset
            best_overall_rsquared = results.rsquared
            best_overall_formula = best_formula
            best_overall_model = best_model
        
        #print("For k={0} the best model is {1} with bic={2:.2f} and R^2={3:.4f}".format(k,best_formula,bic,results.rsquared))
        output[k] = {'best_model':best_formula, 'bic':bic,'r_squared':results.rsquared}
        
    #print("The best overall model is {0} with bic={1:.2f} and R^2={2:.3f}".format(best_overall_formula,overall_min_bic, best_overall_rsquared))
    
    return pd.DataFrame(output).T,best_overall_model,best_overall_subset
```


Before proceeding with feature selection, we filter out the irrelevant variables. These include variables containing information like playlist ID, or variables that cannot be aggregated into a model.



```python
#Filtered columns of interest
cols = ['followers',
 'duration_ms',
 'popularity',
 'acousticness',
 'danceability',
 'energy',
 'liveness',
 'loudness',
 'speechiness',
 'tempo',
 'valence']
```




```python
#Preparing a DataFrame
dfcols = ('BIC','R2','Features')
catmodels = pd.DataFrame(index=playlists['Category'].unique(),columns=dfcols)
```


Now we conduct the feature selection, along with an initial OLS model based on the selected features. The selected features, BIC and R2 of each model is returned per genre, in the DataFrame below.



```python
#Performing Exhaustive Search Selection; Results are Below
for cat in playlists['Category'].unique():
    g1 = playlists.loc[playlists['Category'] == cat]
    g = g1[cols]
    x = g.drop(['followers'],axis=1)
    y = g['followers']
    stats,model,features = exhaustive_search_selection(x,y)
    bestx = g[features]
    model = OLS(y, bestx)
    results = model.fit()
    bic = results.bic
    r_squared = results.rsquared
    catmodels.at[cat,'BIC'] = bic
    catmodels.at[cat,'R2'] = r_squared
    catmodels.at[cat,'Features'] = features
catmodels
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BIC</th>
      <th>R2</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>toplists</th>
      <td>436.215</td>
      <td>0.700129</td>
      <td>[popularity, tempo]</td>
    </tr>
    <tr>
      <th>chill</th>
      <td>1166.46</td>
      <td>0.604576</td>
      <td>[popularity, loudness]</td>
    </tr>
    <tr>
      <th>mood</th>
      <td>891.997</td>
      <td>0.435306</td>
      <td>[popularity]</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>373.588</td>
      <td>0.924562</td>
      <td>[acousticness, danceability, energy, liveness,...</td>
    </tr>
    <tr>
      <th>edm_dance</th>
      <td>1154.91</td>
      <td>0.693318</td>
      <td>[popularity, danceability]</td>
    </tr>
    <tr>
      <th>hiphop</th>
      <td>570.006</td>
      <td>0.440007</td>
      <td>[popularity]</td>
    </tr>
    <tr>
      <th>party</th>
      <td>378.419</td>
      <td>0.913579</td>
      <td>[popularity, danceability, energy, loudness, t...</td>
    </tr>
    <tr>
      <th>rock</th>
      <td>729.646</td>
      <td>0.531455</td>
      <td>[popularity, liveness]</td>
    </tr>
    <tr>
      <th>workout</th>
      <td>746.553</td>
      <td>0.66118</td>
      <td>[duration_ms, popularity]</td>
    </tr>
    <tr>
      <th>focus</th>
      <td>419.172</td>
      <td>0.877931</td>
      <td>[duration_ms, popularity, energy, liveness, lo...</td>
    </tr>
    <tr>
      <th>decades</th>
      <td>432.39</td>
      <td>0.903993</td>
      <td>[popularity, energy, valence]</td>
    </tr>
    <tr>
      <th>dinner</th>
      <td>476.601</td>
      <td>0.536165</td>
      <td>[popularity, danceability]</td>
    </tr>
    <tr>
      <th>sleep</th>
      <td>713.205</td>
      <td>0.375338</td>
      <td>[popularity]</td>
    </tr>
    <tr>
      <th>indie_alt</th>
      <td>572.352</td>
      <td>0.619027</td>
      <td>[popularity, speechiness]</td>
    </tr>
    <tr>
      <th>rnb</th>
      <td>481.901</td>
      <td>0.815941</td>
      <td>[popularity, energy]</td>
    </tr>
    <tr>
      <th>popculture</th>
      <td>504.045</td>
      <td>0.188335</td>
      <td>[energy]</td>
    </tr>
    <tr>
      <th>metal</th>
      <td>990.883</td>
      <td>0.650117</td>
      <td>[popularity, liveness]</td>
    </tr>
    <tr>
      <th>soul</th>
      <td>-268.771</td>
      <td>1</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
    </tr>
    <tr>
      <th>romance</th>
      <td>572.351</td>
      <td>0.638632</td>
      <td>[popularity, speechiness]</td>
    </tr>
    <tr>
      <th>jazz</th>
      <td>545.473</td>
      <td>0.641763</td>
      <td>[popularity, energy]</td>
    </tr>
    <tr>
      <th>classical</th>
      <td>1233.69</td>
      <td>0.355472</td>
      <td>[popularity]</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>1186.1</td>
      <td>0.306946</td>
      <td>[popularity, tempo]</td>
    </tr>
    <tr>
      <th>country</th>
      <td>300.101</td>
      <td>0.962167</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
    </tr>
    <tr>
      <th>folk_americana</th>
      <td>709.793</td>
      <td>0.722024</td>
      <td>[duration_ms, popularity]</td>
    </tr>
    <tr>
      <th>blues</th>
      <td>317.262</td>
      <td>0.801783</td>
      <td>[popularity, liveness]</td>
    </tr>
    <tr>
      <th>travel</th>
      <td>453.239</td>
      <td>0.537812</td>
      <td>[popularity]</td>
    </tr>
    <tr>
      <th>kids</th>
      <td>-261.225</td>
      <td>1</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
    </tr>
    <tr>
      <th>reggae</th>
      <td>367.757</td>
      <td>0.671184</td>
      <td>[popularity, liveness]</td>
    </tr>
    <tr>
      <th>gaming</th>
      <td>855.338</td>
      <td>0.364424</td>
      <td>[popularity]</td>
    </tr>
    <tr>
      <th>punk</th>
      <td>440.189</td>
      <td>0.936401</td>
      <td>[popularity, danceability, energy, speechiness...</td>
    </tr>
    <tr>
      <th>funk</th>
      <td>383.881</td>
      <td>0.920881</td>
      <td>[acousticness, danceability, liveness, speechi...</td>
    </tr>
    <tr>
      <th>comedy</th>
      <td>-94.1374</td>
      <td>1</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
    </tr>
  </tbody>
</table>
</div>



## Model Selection

In addition to the OLS model, we explored other options for building models that predict success for a playlist. Below, we added polynomial values to each model to see whether a polynomial OLS model would perform better than a linear model. However, based on BIC, the linear model consistently performed better; we ultimately did choose the linear model.

We also experimented with other types of classification models, such as decision trees and random forests (sample code at the bottom). However, the sample size per genre was ultimately too small to generate robust results. More importantly, such models do not provide interpretable coefficients. We needed to have coefficients/a regression equation to be able to conduct the next step, generating a successful playlist based on the model. All of these factors strengthened the argument for selecting the linear model.



```python
for cat in playlists['Category'].unique():
    #Data Setup
    g1 = playlists.loc[playlists['Category'] == cat]
    g = g1[cols]
    feats = catmodels.loc[cat,'Features']
    train, test = train_test_split(g, test_size=0.5, random_state=1000)
    y_train = train['followers']
    x_train = train[feats]
    y_test = test['followers']
    x_test = test[feats]
    
    degree = 2
    poly = PolynomialFeatures(degree)

    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    model = OLS(y_train, x_train_poly)
    results = model.fit()
    bic = results.bic
    r_squared = results.rsquared
    catmodels.at[cat,'Poly BIC'] = bic
    catmodels.at[cat,'Poly R2'] = r_squared
catmodels
```


    /Users/sarah/anaconda/lib/python3.6/site-packages/statsmodels/regression/linear_model.py:1386: RuntimeWarning: divide by zero encountered in double_scalars
      return 1 - self.ssr/self.centered_tss
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BIC</th>
      <th>R2</th>
      <th>Features</th>
      <th>Poly BIC</th>
      <th>Poly R2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>toplists</th>
      <td>436.215</td>
      <td>0.700129</td>
      <td>[popularity, tempo]</td>
      <td>-88.199672</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>chill</th>
      <td>1166.46</td>
      <td>0.604576</td>
      <td>[popularity, loudness]</td>
      <td>586.239176</td>
      <td>0.723811</td>
    </tr>
    <tr>
      <th>mood</th>
      <td>891.997</td>
      <td>0.435306</td>
      <td>[popularity]</td>
      <td>431.798758</td>
      <td>0.747629</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>373.588</td>
      <td>0.924562</td>
      <td>[acousticness, danceability, energy, liveness,...</td>
      <td>-192.611815</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>edm_dance</th>
      <td>1154.91</td>
      <td>0.693318</td>
      <td>[popularity, danceability]</td>
      <td>575.743415</td>
      <td>0.724887</td>
    </tr>
    <tr>
      <th>hiphop</th>
      <td>570.006</td>
      <td>0.440007</td>
      <td>[popularity]</td>
      <td>264.889781</td>
      <td>0.342404</td>
    </tr>
    <tr>
      <th>party</th>
      <td>378.419</td>
      <td>0.913579</td>
      <td>[popularity, danceability, energy, loudness, t...</td>
      <td>-160.868050</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>rock</th>
      <td>729.646</td>
      <td>0.531455</td>
      <td>[popularity, liveness]</td>
      <td>355.173060</td>
      <td>0.771883</td>
    </tr>
    <tr>
      <th>workout</th>
      <td>746.553</td>
      <td>0.66118</td>
      <td>[duration_ms, popularity]</td>
      <td>366.290231</td>
      <td>0.608226</td>
    </tr>
    <tr>
      <th>focus</th>
      <td>419.172</td>
      <td>0.877931</td>
      <td>[duration_ms, popularity, energy, liveness, lo...</td>
      <td>-168.357843</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>decades</th>
      <td>432.39</td>
      <td>0.903993</td>
      <td>[popularity, energy, valence]</td>
      <td>-160.006263</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>dinner</th>
      <td>476.601</td>
      <td>0.536165</td>
      <td>[popularity, danceability]</td>
      <td>206.068112</td>
      <td>0.988035</td>
    </tr>
    <tr>
      <th>sleep</th>
      <td>713.205</td>
      <td>0.375338</td>
      <td>[popularity]</td>
      <td>350.721488</td>
      <td>0.369028</td>
    </tr>
    <tr>
      <th>indie_alt</th>
      <td>572.352</td>
      <td>0.619027</td>
      <td>[popularity, speechiness]</td>
      <td>287.245583</td>
      <td>0.834908</td>
    </tr>
    <tr>
      <th>rnb</th>
      <td>481.901</td>
      <td>0.815941</td>
      <td>[popularity, energy]</td>
      <td>218.329986</td>
      <td>0.939218</td>
    </tr>
    <tr>
      <th>popculture</th>
      <td>504.045</td>
      <td>0.188335</td>
      <td>[energy]</td>
      <td>244.058367</td>
      <td>0.204815</td>
    </tr>
    <tr>
      <th>metal</th>
      <td>990.883</td>
      <td>0.650117</td>
      <td>[popularity, liveness]</td>
      <td>514.176506</td>
      <td>0.578631</td>
    </tr>
    <tr>
      <th>soul</th>
      <td>-268.771</td>
      <td>1</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
      <td>-123.354232</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>romance</th>
      <td>572.351</td>
      <td>0.638632</td>
      <td>[popularity, speechiness]</td>
      <td>275.742352</td>
      <td>0.949320</td>
    </tr>
    <tr>
      <th>jazz</th>
      <td>545.473</td>
      <td>0.641763</td>
      <td>[popularity, energy]</td>
      <td>258.678694</td>
      <td>0.967081</td>
    </tr>
    <tr>
      <th>classical</th>
      <td>1233.69</td>
      <td>0.355472</td>
      <td>[popularity]</td>
      <td>613.199879</td>
      <td>0.397786</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>1186.1</td>
      <td>0.306946</td>
      <td>[popularity, tempo]</td>
      <td>590.579931</td>
      <td>0.589571</td>
    </tr>
    <tr>
      <th>country</th>
      <td>300.101</td>
      <td>0.962167</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
      <td>-154.487230</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>folk_americana</th>
      <td>709.793</td>
      <td>0.722024</td>
      <td>[duration_ms, popularity]</td>
      <td>337.803490</td>
      <td>0.958517</td>
    </tr>
    <tr>
      <th>blues</th>
      <td>317.262</td>
      <td>0.801783</td>
      <td>[popularity, liveness]</td>
      <td>-192.160925</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>travel</th>
      <td>453.239</td>
      <td>0.537812</td>
      <td>[popularity]</td>
      <td>213.933008</td>
      <td>0.637392</td>
    </tr>
    <tr>
      <th>kids</th>
      <td>-261.225</td>
      <td>1</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
      <td>-55.015161</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>reggae</th>
      <td>367.757</td>
      <td>0.671184</td>
      <td>[popularity, liveness]</td>
      <td>-133.403783</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>gaming</th>
      <td>855.338</td>
      <td>0.364424</td>
      <td>[popularity]</td>
      <td>421.560757</td>
      <td>0.077895</td>
    </tr>
    <tr>
      <th>punk</th>
      <td>440.189</td>
      <td>0.936401</td>
      <td>[popularity, danceability, energy, speechiness...</td>
      <td>-230.893263</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>funk</th>
      <td>383.881</td>
      <td>0.920881</td>
      <td>[acousticness, danceability, liveness, speechi...</td>
      <td>-211.828618</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>comedy</th>
      <td>-94.1374</td>
      <td>1</td>
      <td>[duration_ms, popularity, acousticness, dancea...</td>
      <td>-47.068720</td>
      <td>-inf</td>
    </tr>
  </tbody>
</table>
</div>





```python
#Data Manipulation: Combining Playlist & Track Data
fulltrackfeatures = pd.read_csv('allfeaturesfinal.csv')
playlistcats = playlists[['pid','Category']]
fulltrackfeatures = fulltrackfeatures.merge(playlistcats,on='pid')
```


## Generating the Playlists

After selecting the model and features for each genre, we created the below function that returns a playlist of 30 songs. The function takes in one argument: a requested playlist genre. The function first subsets our full list of tracks, only selecting tracks that appeared on at least one playlist of the requested genre. Then, it subsets the predictor variables based on the features selected in the previous step. Next, the predicted value of each track is calculated according to the linear model for the genre. The 30 top-performing tracks are selected and returned in a DataFrame, along with the predicted number of followers that the playlist will receive.

The function also compares the generated playlist to a baseline model, that selects tracks solely based on those that are the most popular. Intuitively, this is a strong baseline model because of the high correlation between track popularity and number of playlist followers discussed in the EDA section. The number of predicted followers for the baseline model is also returned in the function. 

Below, there is a sample output for one call of the function, as well as a comparison of the model's performance with the baseline.



```python
#List of potential playlist genres 

catlist = list(playlists['Category'].unique())
catlist
```





    ['toplists',
     'chill',
     'mood',
     'pop',
     'edm_dance',
     'hiphop',
     'party',
     'rock',
     'workout',
     'focus',
     'decades',
     'dinner',
     'sleep',
     'indie_alt',
     'rnb',
     'popculture',
     'metal',
     'soul',
     'romance',
     'jazz',
     'classical',
     'latin',
     'country',
     'folk_americana',
     'blues',
     'travel',
     'kids',
     'reggae',
     'gaming',
     'punk',
     'funk',
     'comedy']





```python
def create_playlists(genre):
    tracksubset = fulltrackfeatures.loc[fulltrackfeatures['Category'] == genre]
    tracksubset = tracksubset.drop_duplicates('id')
    feats = catmodels.loc[cat,'Features']
    x = tracksubset[feats]
    g1 = playlists.loc[playlists['Category'] == genre]
    g = g1[cols]
    y = g['followers']
    bestx = g[feats]
    bestx = sm.add_constant(bestx)
    model = OLS(y, bestx)
    results = model.fit()
    params = results.params
    for index,row in x.iterrows():
        coeffs = list()
        for f in range(len(feats)):
            pred = row[f]
            param = params[f+1]
            co = pred*param
            coeffs.append(co)
        tracksubset.at[index,'Followers'] = np.sum(coeffs) + params[0]
    playlist = tracksubset.nlargest(30, 'Followers', keep='first')
    baseline = tracksubset.nlargest(30, 'popularity', keep='first')
    means = playlist.mean(axis=0)
    basemeans = baseline.mean(axis=0)
    predict_x = means[feats]
    predict_x = predict_x.values
    predict_x = np.insert(predict_x,0,1.)    
    predict_baseline = basemeans[feats]
    predict_baseline = predict_baseline.values
    predict_baseline = np.insert(predict_baseline,0,1.)    
    playlistfollowers = results.predict(predict_x)
    baselinefollowers = results.predict(predict_baseline)
    return playlist, playlistfollowers, baselinefollowers
```




```python
#Comparing Results to Baseline
comparison = pd.DataFrame(index=catlist, columns = ['Playlist Followers', 'Baseline Followers'])
```




```python
for c in catlist:
    playlist, followers, baseline = create_playlists(c)
    comparison.at[c,'Playlist Followers'] = followers
    comparison.at[c, 'Baseline Followers'] = baseline

comparison['Difference'] = comparison['Playlist Followers'] - comparison['Baseline Followers']
comparison
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Playlist Followers</th>
      <th>Baseline Followers</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>toplists</th>
      <td>[17243764.7281]</td>
      <td>[9295405.23792]</td>
      <td>[7948359.49018]</td>
    </tr>
    <tr>
      <th>chill</th>
      <td>[1934683.16227]</td>
      <td>[1627697.47106]</td>
      <td>[306985.691209]</td>
    </tr>
    <tr>
      <th>mood</th>
      <td>[2388926.70725]</td>
      <td>[1926176.47865]</td>
      <td>[462750.228599]</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>[1623408.3223]</td>
      <td>[1438877.03593]</td>
      <td>[184531.286371]</td>
    </tr>
    <tr>
      <th>edm_dance</th>
      <td>[1588029.70216]</td>
      <td>[1568294.47887]</td>
      <td>[19735.2232887]</td>
    </tr>
    <tr>
      <th>hiphop</th>
      <td>[2139579.02761]</td>
      <td>[1945807.2438]</td>
      <td>[193771.783808]</td>
    </tr>
    <tr>
      <th>party</th>
      <td>[3497045.27714]</td>
      <td>[2302931.27116]</td>
      <td>[1194114.00598]</td>
    </tr>
    <tr>
      <th>rock</th>
      <td>[2502834.98699]</td>
      <td>[1111357.44394]</td>
      <td>[1391477.54305]</td>
    </tr>
    <tr>
      <th>workout</th>
      <td>[2130917.79577]</td>
      <td>[2047996.13502]</td>
      <td>[82921.6607459]</td>
    </tr>
    <tr>
      <th>focus</th>
      <td>[2963728.79173]</td>
      <td>[1673828.81446]</td>
      <td>[1289899.97727]</td>
    </tr>
    <tr>
      <th>decades</th>
      <td>[3068196.96542]</td>
      <td>[2395463.37155]</td>
      <td>[672733.593872]</td>
    </tr>
    <tr>
      <th>dinner</th>
      <td>[1064294.21432]</td>
      <td>[806050.408773]</td>
      <td>[258243.805552]</td>
    </tr>
    <tr>
      <th>sleep</th>
      <td>[866743.327066]</td>
      <td>[739709.859313]</td>
      <td>[127033.467753]</td>
    </tr>
    <tr>
      <th>indie_alt</th>
      <td>[1937622.98404]</td>
      <td>[955280.301923]</td>
      <td>[982342.682121]</td>
    </tr>
    <tr>
      <th>rnb</th>
      <td>[2123329.0103]</td>
      <td>[1862493.86574]</td>
      <td>[260835.14456]</td>
    </tr>
    <tr>
      <th>popculture</th>
      <td>[1537953.83677]</td>
      <td>[761987.701654]</td>
      <td>[775966.135112]</td>
    </tr>
    <tr>
      <th>metal</th>
      <td>[487884.670173]</td>
      <td>[475682.409863]</td>
      <td>[12202.2603095]</td>
    </tr>
    <tr>
      <th>soul</th>
      <td>[572809.423147]</td>
      <td>[242352.894815]</td>
      <td>[330456.528332]</td>
    </tr>
    <tr>
      <th>romance</th>
      <td>[1207254.3086]</td>
      <td>[1112025.76338]</td>
      <td>[95228.5452214]</td>
    </tr>
    <tr>
      <th>jazz</th>
      <td>[744496.594589]</td>
      <td>[575627.549776]</td>
      <td>[168869.044813]</td>
    </tr>
    <tr>
      <th>classical</th>
      <td>[416528.845484]</td>
      <td>[300269.134946]</td>
      <td>[116259.710537]</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>[2130152.73457]</td>
      <td>[1738493.36688]</td>
      <td>[391659.36769]</td>
    </tr>
    <tr>
      <th>country</th>
      <td>[922722.541159]</td>
      <td>[156777.398765]</td>
      <td>[765945.142393]</td>
    </tr>
    <tr>
      <th>folk_americana</th>
      <td>[1112466.58715]</td>
      <td>[1031114.30507]</td>
      <td>[81352.2820876]</td>
    </tr>
    <tr>
      <th>blues</th>
      <td>[700896.962786]</td>
      <td>[554045.64322]</td>
      <td>[146851.319566]</td>
    </tr>
    <tr>
      <th>travel</th>
      <td>[2126083.44172]</td>
      <td>[976903.762031]</td>
      <td>[1149179.67969]</td>
    </tr>
    <tr>
      <th>kids</th>
      <td>[1130889.09498]</td>
      <td>[-284282.514369]</td>
      <td>[1415171.60935]</td>
    </tr>
    <tr>
      <th>reggae</th>
      <td>[1350766.92597]</td>
      <td>[963562.159335]</td>
      <td>[387204.766633]</td>
    </tr>
    <tr>
      <th>gaming</th>
      <td>[297177.958631]</td>
      <td>[44291.4104256]</td>
      <td>[252886.548205]</td>
    </tr>
    <tr>
      <th>punk</th>
      <td>[1100380.39073]</td>
      <td>[956183.746332]</td>
      <td>[144196.644395]</td>
    </tr>
    <tr>
      <th>funk</th>
      <td>[707036.427849]</td>
      <td>[191685.861049]</td>
      <td>[515350.5668]</td>
    </tr>
    <tr>
      <th>comedy</th>
      <td>[623123.754346]</td>
      <td>[365204.02189]</td>
      <td>[257919.732456]</td>
    </tr>
  </tbody>
</table>
</div>



The highly positive values in the "comparison" column show that our model was successful in generating good playlists for each genre.



```python
#Decision tree models, using train/test split, that didn't work:

for cat in playlists['Category'].unique():
    #Data Setup
    g1 = playlists.loc[playlists['Category'] == cat]
    g = g1[cols]
    feats = catmodels.loc[cat,'Features']
    train, test = train_test_split(g, test_size=0.5, random_state=1000)
    y_train = train['followers']
    x_train = train[feats]
    y_test = test['followers']
    x_test = test[feats]
    
    #OLS
    model = OLS(y_train, x_train)
    results = model.fit()
    predictions = results.predict(x_test)
    score = np.mean(predictions==y_test)
    catmodels.at[cat,'OLS Score'] = score

    #Decision Tree
    depth = []
    tree_start = 3
    tree_end   = 20
    for i in range(tree_start,tree_end):
        dt = DecisionTreeClassifier(max_depth=i)
        scores = metrics.accuracy_score(y_test, dt.fit(x_train, y_train).predict(x_test))
        depth.append((i,scores.mean()))
    best_depth = np.argmax(np.array(depth)[:,1]) + tree_start
    dt = DecisionTreeClassifier(max_depth=best_depth)
    dt_fitted = dt.fit(x_train, y_train)
    dtscore = dt_fitted.score(x_test, y_test)
    catmodels.at[cat,'Tree Score'] = dtscore
    
    #Random Forest
    trees = [2**x for x in range(7)]
    ntrees = dict()
    for n_trees in trees:
        rf = RandomForestClassifier(n_estimators=n_trees, max_features='sqrt')
        rfscore = metrics.accuracy_score(y_test, rf.fit(x_train, y_train).predict(x_test))
        ntrees[n_trees] = rfscore
    best_n = max(ntrees, key=ntrees.get)
    catmodels.at[cat,'RF Score'] = ntrees[best_n]
    
```

