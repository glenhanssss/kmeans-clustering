import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import numpy as np

# Set the page to wide mode
# st.set_page_config(layout="wide")

# test
st.write('hiiii')

# Upload file
file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
dataset = st.file_uploader(label = '')

use_defo = st.checkbox('Use example Dataset')
if use_defo:
    # dataset = r"C:\Users\075422749\Downloads\aichemist\dataset\obesity_used.csv"
    dataset = r'obesity_data.csv'
    # st.write("[Dataset Explanation Link](___________________________)")

if dataset:
    if file_format == 'csv' or use_defo:
        df = pd.read_csv(dataset)
    else:
        df = pd.read_excel(dataset)

    # Display the DataFrame
    st.write("Dataset:")
    st.dataframe(df)

    # Encode categorical features to number
    # Get list of object, string columns
    le = LabelEncoder()
    object_columns = df.columns[df.dtypes == 'object']
    string_columns = df.columns[df.dtypes == 'string']

    # Encode columns that are object type or string type
    df2=df.copy()
    for current_object_columns in object_columns:
        df2[current_object_columns] = le.fit_transform(df2[current_object_columns])

    for current_string_columns in string_columns:
        df2[current_string_columns] = le.fit_transform(df2[current_string_columns])
    
    # st.dataframe(df2)    

    # Select the target variable (y) and the independent variables (X)
    independent_variables = st.multiselect("Select Independent Variables (X)", df2.columns.tolist())

    if st.button("Perform K-Means Clustering"):
        st.markdown("<br>", unsafe_allow_html=True)

        # 
        # X = df2[independent_variables]


        # Find the optimal number of cluster

        # Select the features that the user wants
        selected_features = independent_variables
        df3 = df2[selected_features]
        st.write(selected_features)

        # Sihouette Method
        # Measure how similar a point is to its own cluster compared to other clusters
        sil = []
        kmax = 10
        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(df3)
            labels = kmeans.labels_
            sil.append(silhouette_score(df3, labels, metric = 'euclidean'))

        plt.figure(figsize=(10,5))
        plt.plot(range(2,kmax+1),sil,marker='+')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()
        st.pyplot(plt)

        # Get the biggest silhouette score 
        n_cluster = sil.index(max(sil)) + 2
        st.write("Number of clusters with highest distortion:", n_cluster)
        st.markdown("<br><br>", unsafe_allow_html=True)

################
        # Performs K-means clustering based on the total number of clusters obtained previously
        kmeans=KMeans(n_clusters=n_cluster,random_state=101)
        kmeans.fit(df3)
################
        # Add cluster number result to each of the data row to the dataframe
        labels=pd.Series(kmeans.labels_,name='cluster_number')
        df_with_cluster=pd.concat([df,labels],axis=1)

        # Age binning
        age_bins = [0,4,9,18,59,100]
        age_bin_labels = ['<5', '5-9', '10-18', '18-59', '60+']
        df_with_cluster['age_bins']=pd.cut(df_with_cluster['Age'],bins=age_bins, labels=age_bin_labels)

        # # Height binning
        height_bins = [0, 150, 160, 170, 180, 190, 300]  # Bins must cover the entire range
        height_bin_labels = ['<150', '150-160', '160-170', '170-180', '180-190', '190+']
        df_with_cluster['height_bins']=pd.cut(df_with_cluster['Height'],bins=height_bins, labels=height_bin_labels)

        # Weight binning
        weight_bins = [0, 10, 19, 29, 49, 69, 89, 109, 250]
        weight_bins_labels = ['<10', '10-19', '20-29', '30-49', '50-69', '70-89', '90-109', '110+']
        df_with_cluster['weight_bins']=pd.cut(df_with_cluster['Weight'],bins=weight_bins, labels=weight_bins_labels)

        # BMI binning
        bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, 100]
        bmi_bin_labels = ['<18.5', '18.5-24.9', '25-29.9', '30-34.9', '35-39.9', '40+']
        df_with_cluster['bmi_bins'] = pd.cut(df_with_cluster['BMI'], bins=bmi_bins, labels=bmi_bin_labels)
################
        # People count for every cluster
        # Plot with bar chart
        peopleCount = df_with_cluster['cluster_number'].value_counts().plot(kind='bar',cmap='Set2')

        # Add labels on top of each bar
        for bar in peopleCount.patches:  # Iterate over each bar in the plot
            peopleCount.text(
                bar.get_x() + bar.get_width() / 2, # X-coordinate (center of bar)
                bar.get_height(),                  # Y-coordinate (height of bar)
                int(bar.get_height()),             # Text (bar height as integer)
                ha='center',                       # Horizontal alignment
                va='bottom'                        # Vertical alignment
            )

        st.write('People count for every cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of People')
        plt.show()
        st.pyplot(plt)
        st.markdown("<br><br>", unsafe_allow_html=True)
################
################
        for current_selected_feature in selected_features:
            if current_selected_feature == "Age" or current_selected_feature == "Weight" or current_selected_feature == "Height" or current_selected_feature == "BMI":
                df_cluster_age=df_with_cluster.groupby(['cluster_number',current_selected_feature.lower()+'_bins']).size().unstack().fillna(0)
                plt.figure(figsize=(5, 4))
                sns.heatmap(df_cluster_age.apply(lambda x:x/x.sum(),axis=1),annot=True,cmap='BuPu')

                st.write(current_selected_feature + ' group distribution')
                plt.gcf().set_size_inches(13,4)
                plt.xlabel(current_selected_feature+ ' Group')
                plt.ylabel('Cluster')
                plt.show()
                st.pyplot(plt)
                st.markdown("<br><br>", unsafe_allow_html=True)
            else:
                df_cluster_gender=df_with_cluster.groupby(['cluster_number',current_selected_feature]).size().unstack().fillna(0)
                plt.figure(figsize=(5, 4))
                sns.heatmap(df_cluster_gender.apply(lambda x:x/x.sum(),axis=1),annot=True,cmap='BuPu')

                st.write(current_selected_feature + ' Distribution')
                plt.xlabel(current_selected_feature)
                plt.ylabel('Cluster')
                plt.show()
                st.pyplot(plt)
                st.markdown("<br><br>", unsafe_allow_html=True)
