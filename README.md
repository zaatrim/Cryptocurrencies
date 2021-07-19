# ** Cryptocurrencies

## *Project Overview*
In this Project I will create a report that includes what cryptocurrencies are on the trading market and how they could be grouped to create a classification system. Since the data I will be working with is not ideal, so it will need to be processed to fit the machine learning models. Since there is no known output for what I am looking for, I will use unsupervised learning. To group the cryptocurrencies, I decided on a clustering algorithm. I’ll use data visualizations to share The findings.
                  
## *Analysis & Results*
### Analysis

#### preprocess the dataset in order to perform PCA. 
1) Load the Data and Perform data clean up for PCA
            # Load the crypto_data.csv dataset.
            file_path = Path("Resources/crypto_data.csv")
            crypto_df = pd.read_csv(file_path, index_col=0)
            crypto_df.head(5)

            # Keep all the cryptocurrencies that are being traded.
            crypto_df = crypto_df[crypto_df["IsTrading"] == True]
            print(crypto_df.shape)
            crypto_df.head(10)

            # Keep all the cryptocurrencies that have a working algorithm.
            crypto_df = crypto_df[crypto_df["Algorithm"] != "N/A"]
            print(crypto_df.shape)
            crypto_df.head(10)

            # Remove the "IsTrading" column. 
            crypto_df = crypto_df[crypto_df["Algorithm"] != "N/A"]
            print(crypto_df.shape)
            crypto_df.head(10)

            # Remove rows that have at least 1 null value.
            crypto_df = crypto_df.dropna(axis=0, how="any")
            # crypto_df[~(crypto_df.isna() == True).any")
            print(crypto_df.shape)
            crypto_df.head(10)

            # Keep the rows where coins are mined.
            crypto_df = crypto_df[crypto_df["TotalCoinsMined"] > 0]
            print(crypto_df.shape)
            crypto_df.head(10)

            # Create a new DataFrame that holds only the cryptocurrencies names.
            coins_name = pd.DataFrame(crypto_df["CoinName"], index=crypto_df.index)
            print(coins_name.shape)
            coins_name.head()

            # Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
            crypto_df = crypto_df.drop("CoinName", axis=1)
            print(crypto_df.shape)
            crypto_df.head(10)

            # Use get_dummies() to create variables for text features.
            X = pd.get_dummies(data=crypto_df, columns=["Algorithm", "ProofType"])
            print(X.shape)
            X.head(10)

            # Standardize the data with StandardScaler().
            X = StandardScaler().fit_transform(X)
            X[:5]

#### Reducing Data Dimensions Using PCA
Next I will apply the Principal Component Analysis (PCA) algorithm to reduce the dimensions of the X DataFrame to three principal components and place these dimensions in a new DataFrame.

            # Using PCA to reduce dimension to three principal components.
            n_comp = 3
            pca = PCA(n_components=n_comp)
            principal_components = pca.fit_transform(X)
            principal_components

            # Create a DataFrame with the three principal components.
            col_names = [f"PC {i}" for i in range(1, n_comp + 1)]
            pcs_df = pd.DataFrame(principal_components, columns=col_names, index=crypto_df.index)
            print(pcs_df.shape)
            pcs_df.head(10)

#### Clustering Crytocurrencies Using K-Means

In this section, Using the K-means algorithm, I’ll create an elbow curve using hvPlot to find the best value for K from the pcs_df DataFrame created earlier in this project. Then, I’ll run the K-means algorithm to predict the K clusters for the cryptocurrencies’ data.

3.1) Finding the Best Value for k Using the Elbow Curve

            # Create an elbow curve to find the best value for K.
            inertia = []
            k = list(range(1, 11))
            # Calculate the inertia for the range ok k values
            for i in k:
                km = KMeans(n_clusters=i, random_state=0)
                km.fit(pcs_df)
                inertia.append(km.inertia_)
            # Create the Elbow Curve using hvPlot
            elbow_data = {"k": k, "inertia": inertia}
            df_elbow = pd.DataFrame(elbow_data)
            df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")

3.2) Make Prediction using K clusters = 4 of the cryptocurrencies’ data 
            # Initialize the K-Means model.
            model = KMeans(n_clusters=4, random_state=0)

            # Fit the model
            model.fit(pcs_df)

            # Predict clusters
            predictions = model.predict(pcs_df)
            predictions

3.3) Create a new DataFrame including predicted clusters and cryptocurrencies features.
            # Concatentate the crypto_df and pcs_df DataFrames on the same columns.
            clustered_df = pd.concat([crypto_df, pcs_df], axis=1, sort=False)

            #  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
            clustered_df["CoinName"] = coins_name["CoinName"]

            #  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
            clustered_df["Class"] = model.labels_

            # Print the shape of the clustered_df
            print(clustered_df.shape)
            clustered_df.head(10)

#### Visualizing Cryptocurrencies Results
Last step I will create scatter plots with Plotly Express and hvplot, I’ll visualize the distinct groups that correspond to the three principal components I created in 2nd step, then I’ll create a table with all the currently tradable cryptocurrencies using the hvplot.table() function.


### Results

#### in 1st step we did Data processing to get clean Dataset to use for the PCA modle. 
Data clean up, outcome 

![image1](https://user-images.githubusercontent.com/80013773/126110358-1d149ff8-cb82-4f0e-a9bc-566b6c02c1cb.PNG)

Standarize the Data :

![image2](https://user-images.githubusercontent.com/80013773/126110404-ef67d94e-119d-440e-aabd-2ef8b6470935.PNG)


#### Reducing Data Dimensions Using PCA

![image3](https://user-images.githubusercontent.com/80013773/126110439-258bbbd1-2473-4300-a562-29308cbb7e3c.PNG)

####  Clustering Crytocurrencies Using K-Means
3.1) Finding the Best Value for k Using the Elbow Curve

![image4](https://user-images.githubusercontent.com/80013773/126110486-7e7739c5-48d0-4945-aad6-1b6f3dacc470.PNG)

3.2) New DataFrame including predicted clusters and cryptocurrencies features outcome

![image5](https://user-images.githubusercontent.com/80013773/126110515-5870c7c4-e868-4427-9255-8faf2a25836b.PNG)


#### Visualizing Cryptocurrencies Results

3D-Scatter with Clusters

![image6](https://user-images.githubusercontent.com/80013773/126110563-97effca5-39db-432b-8ebe-882d717a6148.PNG)

Tradable cryptocurrencies table

![image7](https://user-images.githubusercontent.com/80013773/126110625-e3667502-5dc4-44a7-a815-8c2cce067b2a.PNG)

Visualizing total Coinsupply vs. Total Coin Mined. 

![image8](https://user-images.githubusercontent.com/80013773/126110662-cd170bd0-661c-4856-8cbd-64d104a2ebe2.PNG)
