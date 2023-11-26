# Metrics for Text Clustering Validation

This repo involves a comprehensive metrics class for text clustering validation. The class incorporates a mix of existing solutions and custom-written functions to evaluate the quality of clustering results. In addition to leveraging existing solutions, the library includes self-written metrics tailored to specific nuances of clustering tasks.

The class encompasses a diverse set of clustering validation metrics to address different aspects of clustering quality.
The motivation behind creating this library is to provide a robust toolkit for data scientists and practitioners working with clustering algorithms. Clustering validation is a critical step in ensuring the reliability of results, and this library aims to simplify and enhance that process.

Metrics are derived from well-established solutions, including those discussed by Hui Xiong and Zhongmou Li in the ["Clustering Validation Measures"]([https://duckduckgo.com](https://www.amazon.com/Machine-Learning-Text-Charu-Aggarwal/dp/3319735306)https://www.amazon.com/Machine-Learning-Text-Charu-Aggarwal/dp/3319735306) section of Charu C. Aggarwal's book.

# List of the metrics
* Cohesion
* Error Sum of Squares (SSE)
* Separation
* SST
* Root mean square standard deviation (RMSSTD)
* R-squared index
* Calinski-Harabasz score
* The Dunn Index (DI)
* Silhouette index
* Davies-Bouldin score
* Xie-Beni index
* SD validity index
* S_Dbw
* Variance of the nearest neighbor distance (VNND)
* Clustering Validation index based on Nearest Neighbors (CVNN)

# The following third party libraries are currently used
* sklearn.metrics
  * Calinski-Harabasz score
* s_dbw.SD
  * S_Dbw
