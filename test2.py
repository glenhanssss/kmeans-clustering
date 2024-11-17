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

# Create a top container for the navbar
with st.container():
    # Split the space into columns
    col1, col2 = st.columns([1, 5])  # Adjust proportions as needed
    
    # Add the logo in the first column
    with col1:
        st.image(r"images/aichemy.png", width=100)  # Replace with the path to your logo

    # Add navigation links or a title in the second column
    with col2:
        st.markdown("""
        <style>
        .navbar-links {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 20px;
            font-size: 25px;
            font-weight: bold;
        }
        .navbar-links a {
            text-decoration: none;
            color: #004080;
        }
        .navbar-links a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="navbar-links">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#contact">Contact</a>
        </div>
        """, unsafe_allow_html=True)

# Main content
st.title("Obese K-Means Clustering Application")
st.markdown("<strong>AIchemist: Leveraging AI to Bridge Demographic and Characteristic Gaps in Health Education</strong>", unsafe_allow_html=True)

# Upload file
file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
dataset = st.file_uploader(label = '')

# Read the dataset
use_defo = st.checkbox('Use example Dataset')
if use_defo:
    dataset = r'obesity_data.csv'

if dataset:
    if file_format == 'csv' or use_defo:
        df = pd.read_csv(dataset)
    else:
        df = pd.read_excel(dataset)

    # Show the DataFrame
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
    
    # Select the target variable (y) and the independent variables (X)
    independent_variables = st.multiselect("Select Independent Variables (X)", df2.columns.tolist())
    if st.button("Perform K-Means Clustering"):
        st.markdown("<br>", unsafe_allow_html=True)
        ############# Find the optimal number of cluster #############
        # Select the features that the user wants
        selected_features = independent_variables
        df3 = df2[selected_features]

        # Sihouette Method
        # Measure how similar a point is to its own cluster compared to other clusters
        sil = []
        kmax = 10
        for k in range(2, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(df3)
            labels = kmeans.labels_
            sil.append(silhouette_score(df3, labels, metric = 'euclidean'))

        st.markdown("<h3>Optimal Number of Cluster</h3>", unsafe_allow_html=True)
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

        ############# K-MEANS CLUSTERING #############
        # Performs K-means clustering based on the total number of clusters obtained previously
        kmeans=KMeans(n_clusters=n_cluster,random_state=101)
        kmeans.fit(df3)

        # Add cluster number result to each of the data row to the dataframe
        labels=pd.Series(kmeans.labels_,name='cluster_number')
        df_with_cluster=pd.concat([df,labels],axis=1)

        ############# ADD BINNED FEATURES TO THE DATAFRAME #############
        # PREDEFINED BINNING VALUE  
        pre_defined_binning_features = ['Age', 'Height', 'Weight', 'BMI']
        # Age binning
        if 'Age' in df_with_cluster.columns:
            age_bins = [0,4,9,18,59,100]
            age_bin_labels = ['<5', '5-9', '10-18', '18-59', '60+']
            df_with_cluster['age_bins']=pd.cut(df_with_cluster['Age'],bins=age_bins, labels=age_bin_labels)

        # Height binning
        if 'Height' in df_with_cluster.columns:
            height_bins = [0, 150, 160, 170, 180, 190, 300]  # Bins must cover the entire range
            height_bin_labels = ['<150', '150-160', '160-170', '170-180', '180-190', '190+']
            df_with_cluster['height_bins']=pd.cut(df_with_cluster['Height'],bins=height_bins, labels=height_bin_labels)

        # Weight binning
        if 'Weight' in df_with_cluster.columns:
            weight_bins = [0, 10, 19, 29, 49, 69, 89, 109, 250]
            weight_bins_labels = ['<10', '10-19', '20-29', '30-49', '50-69', '70-89', '90-109', '110+']
            df_with_cluster['weight_bins']=pd.cut(df_with_cluster['Weight'],bins=weight_bins, labels=weight_bins_labels)

        # BMI binning
        if 'BMI' in df_with_cluster.columns:
            bmi_bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, 100]
            bmi_bin_labels = ['<18.5', '18.5-24.9', '25-29.9', '30-34.9', '35-39.9', '40+']
            df_with_cluster['bmi_bins'] = pd.cut(df_with_cluster['BMI'], bins=bmi_bins, labels=bmi_bin_labels)

        # Auto binning 
        # Automtically binn features (integer and float) that aren't on PREDEFINED BINNING FEATURES list

        # int_columns = df.columns[df.dtypes == 'int64'] 
        int_columns = df.columns[(df.dtypes == 'int64') & (df.nunique() > 10)]
        int_columns = [x for x in int_columns if x not in pre_defined_binning_features]

        # float_columns = df.columns[df.dtypes == 'float64']
        float_columns = df.columns[(df.dtypes == 'float64') & (df.nunique() > 10)]
        float_columns = [x for x in float_columns if x not in pre_defined_binning_features]

        # Combine the result for int and float features
        combined_auto_binned_columns = int_columns + float_columns
        
        # Version 1
        # final_combined_auto_binned_columns = combined_auto_binned_columns
        # for current_column in combined_auto_binned_columns:
        #     if df_with_cluster[current_column].nunique() > 10: # Check if the current column/feature has more than 10 Unique values/data. IF True THEN bin based on percentiles
        #         # Create bins based on specific percentiles
        #         # Defined percentiles (0%, 25%, 50%, 75%, 100%)
        #         percentiles = [0, 25, 50, 75, 100]  
        #         bin_edges = np.percentile(df_with_cluster[current_column], percentiles)

        #         # Create labels for bins
        #         labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

        #         # Bin the row averages
        #         df_with_cluster[current_column+'_bins'] = pd.cut(df_with_cluster[current_column], bins=bin_edges, labels=labels, include_lowest=True)
        #     else: # IF has <= 10 --> THEN no need to bin the feature/column
        #         final_combined_auto_binned_columns.remove(current_column)
        
        # Version 2
        for current_column in combined_auto_binned_columns:
            # Create bins based on specific percentiles
            # Defined percentiles (0%, 25%, 50%, 75%, 100%)
            percentiles = [0, 25, 50, 75, 100]  
            bin_edges = np.percentile(df_with_cluster[current_column], percentiles)

            # Create labels for bins
            labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

            # Bin the row averages
            df_with_cluster[current_column+'_bins'] = pd.cut(df_with_cluster[current_column], bins=bin_edges, labels=labels, include_lowest=True)


        ############# GENERATE THE RESULTS, CHARTS, ETC #############

        # People count for every cluster
        # Plot with bar chart
        peopleCount = df_with_cluster['cluster_number'].value_counts().plot(kind='bar',cmap='Set2')
        # Add labels on top of each bar
        for bar in peopleCount.patches:  # Iterate over each bar in the plot
            peopleCount.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height(),                  
                int(bar.get_height()),             
                ha='center',                       
                va='bottom'                       
            )
        st.markdown("<h3>Population Distribution Across Clusters</h3>", unsafe_allow_html=True)
        plt.xlabel('Cluster')
        plt.ylabel('Number of People')
        plt.show()
        st.pyplot(plt)
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Generate charts, results for SELECTED FEATURES ONLY
        for current_selected_feature in selected_features:
            # Check if the selected feature is in the list of binned features
            if (current_selected_feature in pre_defined_binning_features) or (current_selected_feature in combined_auto_binned_columns):
                df_cluster_age=df_with_cluster.groupby(['cluster_number',current_selected_feature.lower()+'_bins']).size().unstack().fillna(0)
                plt.figure(figsize=(5, 4))
                sns.heatmap(df_cluster_age.apply(lambda x:x/x.sum(),axis=1),annot=True,cmap='BuPu')

                st.markdown("<h3>" + current_selected_feature + ' group distribution' + "</h3>", unsafe_allow_html=True)
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

                st.markdown("<h3>" + current_selected_feature + ' Distribution' + "</h3>", unsafe_allow_html=True)
                plt.xlabel(current_selected_feature)
                plt.ylabel('Cluster')
                plt.show()
                st.pyplot(plt)
                st.markdown("<br><br>", unsafe_allow_html=True)

            # # Future Feature Text
            # st.markdown("""<hr style="height:4px;border:none;color:#333;background-color:#333;" />""", unsafe_allow_html=True)
            # st.title("Future Feature with Generative AI")
            # st.markdown("<strong>Form Input based on Clustering Result</strong>", unsafe_allow_html=True)
            # st. image(r"images/4. AIchemist generate.png")
            # st.markdown("<br>", unsafe_allow_html=True)
            # st.markdown("<strong>Recommendation and Material Result based on Form</strong>", unsafe_allow_html=True)
            # st.image(r"images/5. AIchemist generate -result.png")
