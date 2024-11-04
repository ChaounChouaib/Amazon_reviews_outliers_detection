Amazon Reviews Outlier Detection and Distribution Shift Analysis

Overview

This project aims to detect outliers and measure distribution shifts in the Amazon Reviews Dataset. The approach integrates text and metadata features to identify unusual reviews and assess shifts in data distribution, potentially impacting model performance and reliability. The project leverages autoencoders, Population Stability Index (PSI), and Wasserstein Distance for a comprehensive analysis.

Methodology

	1.	Outlier Detection:
	•	Initial attempts used Isolation Forest on textual embeddings, but results were inconclusive.
	•	The final approach combined numerical features with text embeddings to train autoencoder models, allowing for a more refined outlier detection based on reconstruction error. Outliers were identified using an MSE threshold set at the 99.9th percentile.
	2.	Distribution Shift Scoring:
	•	Wasserstein Distance: Calculated between inliers and outliers in the test set, and between train and test sets, providing insight into drift magnitude.
	•	Population Stability Index (PSI): Calculated per feature, with a focus on identifying high-drift features. Analyzed review-specific features (“text_review” and “title_review”) to evaluate the feasibility of outlier detection based solely on review data.

Repository Structure

	•	app/: Contains data and model storage.
	•	data/: Organized data storage.
	•	embeddings/: Precomputed embeddings for reviews.
	•	models/: Saved models for serving.
	•	processed/: Processed datasets ready for analysis.
	•	raw/: Raw data files.
	•	src/: Source code for the project.
	•	data_preprocessing.py: Data cleaning and preprocessing routines.
	•	data_transformation.py: Data transformations, including feature engineering.
	•	drift_score.py: Implements drift scoring calculations (e.g., PSI, Wasserstein distance).
	•	features_engineering.py: Feature extraction and engineering methods.
	•	model.py: Defines and trains outlier detection models (e.g., autoencoder).
	•	outlier_detection.py: Main outlier detection module.
	•	app.py: API setup using FastAPI for model serving.
	•	back.py: Auxiliary backend functions for API support.
	•	notebooks/: Jupyter notebooks for data exploration and experimentation.
	•	Dockerfile: Docker configuration to containerize the application.
	•	README.md: Project documentation.
	•	requirements.txt: Python dependencies for setting up the environment.

Setup and Execution

Prerequisites

	•	Python 3.8 or higher
	•	Docker (optional for containerized setup)

Installation

	1.	Clone the Repository:

git clone https://github.com/chaoun.chouaib/amazon-reviews-outlier-detection.git
cd amazon-reviews-outlier-detection


	2.	Install Dependencies:

pip install -r requirements.txt


	3.	Run the Application:
To launch the API with FastAPI:

python src/app.py

This will start a local server where endpoints for outlier detection and drift scoring can be accessed.

	4.	Containerized Setup with Docker:

docker build -t amazon-reviews-outlier .
docker run -p 8000:8000 amazon-reviews-outlier



API Endpoints

	•	Outlier Detection: Endpoint to identify outliers based on review and metadata.
	•	Drift Scoring: Endpoint to calculate PSI and Wasserstein distance for drift analysis.

Assumptions and Limitations

	•	Data Limitations: The dataset selection is based on the Amazon Reviews categories, with a switch from “All_beauty” to multiple categories (Subscription Boxes, Software, Video Games) to capture more varied review patterns.
	•	Feature Selection: Outliers are primarily identified based on combined text and numerical features, with a focus on “text_review” and “title_review” for potential standalone detection.
	•	Threshold Selection: The 99.9th percentile of reconstruction error was chosen as a threshold for outliers. This may need adjustment for different datasets or use cases.
	•	PSI Binning: The PSI score calculation faced binning challenges, leading to potential variations in PSI thresholds.

Future Improvements

	•	Refine feature engineering to focus on review-specific data for more targeted outlier detection.
	•	Experiment with additional visualization techniques for better interpretability of detected outliers.

Let me know if you’d like any modifications!