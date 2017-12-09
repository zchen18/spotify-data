---
title: Final Milestone
notebook: milestone-report.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}

# Final Milestone

By: Zhaodong Chen and Sarah Anderson

### 1. Project Statement

Music Streaming is one of the most active and fast-growing ways to enjoy music in our digital world today. Spotify, as one of the biggest player in the industry, was able to capture the demands of the industry through its services and playlist system. With more than 40 millions songs avaliable, Spotify is able have a lot of user-generated choices through curated playlists. Definitely, Spotify has a strong-curated algorithm designed to create a discovery playlist for users and the success of this playlist is still in development for Spotify. The goal of this project is to use the machine learning model we learned in class to gather data for Spotify API, build, and evaluate model on the effectiveness of playlist and the generation of a popular playlist according to a genre specified by an user.

### 2. Introduction and Data

With over 140 millions users and 40 millions songs, Spotify has a large share of the market. But in order to ward off competitors and keep its users active, Spotify needs to generate more reasons for users to stay engage on Spotify. One of the more important ways is allow the discovery of more tracks and music and thus the discovery-curated playlist is born. It is designed to suggest more music based on similarity between playlists of users of similar taste, the analysis of similar text associate with tracks, and raw audio data similarity. Spotify's Discovery Weekly is built on those three models of Collaborative Filtering, Natural Language Selection, and Audio models. Definitely, many people find it to extremely effective as the algorithm could generate interests tracks to the users week after week. We want to replicate its effectiveness through a simpler model as we first analyze the success of playlist(in terms of followers) by only the data local to the playlists themselves. Then, we want to create a playlist solely using the important predictors we discovered in the previous part and generate a "successful" playlist with genre as consideration.

Data were gathered from the Spotify API themselves. We did take a deep dive into the Million Song Dataset and found it extremely useful for songs up to the 2000s. With the latest song in the database being in 2010. Many tracks and remixes have been created since then and a lot of the more popular tracks we found were made after the latest date in the database. Many contemporary superstars did not release tracks that were recorded in the Million Song Dataset and we found a lot of blanks when cross-checking. In addition a lot of the features in the Million Song Dataset were in Spotify API and some of the features are rated differently and hard to consolidate together.  After our EDA with API data, we found that the audio features of individual tracks within a playlist to be extremely helpful when coming up with predictors for our model and we decide to solely focus on the API data gathered. 

In our EDA, first we need to find whether playlist followers and track popularities equal to the same thing. And in our first chart, we did see a strong positive correlation and thus was able to create the foundation for our case. Then we want to see if the length of the playlists matter and the data did not show a strong correlation and thus we could see that length is not a strong indicator of popularity with popular playlists.  But popular tracks are mostly contained within the 0-400 songs range. Then we were able to gather genres we want to tackle in the following EDA. Now through the EDA, we know to create models that will find the predictors that is right for each genres. We know that for certain genres like jazz where energy matters while some genres like metal where liveness matters.

### 3. Literature Review

We used this source to find some of the mddels behind how Spotify create their actual Discovery Weekly.
Ciocca, S. (2017, October 10). Spotify's Discover Weekly: How machine learning finds your new music. Retrieved from https://hackernoon.com/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe

This source helped us to understand how to use the attributes of tracks to generate a model
Pichl, M., Zangerle, E., & Specht, G. (2016). Understanding Playlist Creation on Music Streaming Platforms. 2016 IEEE International Symposium on Multimedia (ISM). doi:10.1109/ism.2016.0107

This source helped us to see a previous work done on the subject from the user side
Van den Hoven, (2015, August). Analyzing Spotify Data, Exploring the possibilities of user data from a scientific and a business perspective (Working paper). Retrieved http://www.math.vu.nl/~sbhulai/papers/paper-vandenhoven.pdf

We looked at this source for some inspiration regarding the subject at hand
Jannach, Dietmar & Kamehkhosh, Iman & Bonnin, Geoffray. (2014). Analyzing the characteristics of shared playlists for music recommendation. CEUR Workshop Proceedings. 1271. http://ceur-ws.org/Vol-1271/Paper1.pdf


### 4. Modelling Approach and Trajectory 

We were able to add more playlists to the amount of playlists from our preliminary EDA but were limited by the API itself. Since there is limit to amount of playlist we could grab from the API regarding the playlist. But we were able to get all track information within the playlists. Then we filtered the data to get rid of information like playlist ID, or variables that cannot be aggregated into a model. Our basic model for all genre is to just have one predictor, track popularity. Then we use OLS model to test for other significant predictors among the features we have and we can see that there are different features that are significant for the response variable which is the amount of followers for the playlist of that genre. 

Then we ran a polynomial model and found the results to be even less effective than the linear model. We did attempt other models such as decision trees and random forest because we are already splitting the playlist data into its individual generes and we could not have enough robust responses after train-test split. The code for our trials with those models are below. In addition, since we want to build a playlist from significant features, the other models will not give us good tools to work with as we build a model to generate the playlist. 

Then we generated a playlist based on the significant predictors from the previous models for each genres and compared it to the baseline for the genres which was generated by the mean of the playlists in the genre. We can see that our "generated" playlist would do better comparatively across the board when compared to the means and would be consider a "successful" playlist by the standard of above average.


### 5. Result, Conclusion, and Future Works

We found the OLS model to be the best for some of the top genres and gathered good predictors that account for a large share of the correlation in terms of followers. That being said, more work need to be done in terms of getting more information from API or perhaps contact Spotify themselves for the data as some of the literature review we took inspiration from. Definitely, future work should be done on the field in terms of more data gathered and perhaps if possible a better classfication of followers by country or culture could give us even better models.



```python

```

