import pandas as pd
import numpy as np
import logging
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Configure logging
logging.basicConfig(filename='anomaly_detection.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class AnomalyDetector:
    def __init__(self, input_file):
        self.input_file = input_file
        self.data = None
        self.anomalies = None

    def read_excel(self):
        """Read and load Excel data"""
        try:
            self.data = pd.read_excel(self.input_file, sheet_name=None)  # Read all sheets
            logging.info(f"Successfully read {self.input_file}")
        except Exception as e:
            logging.error(f"Error reading {self.input_file}: {e}")
            raise

    def preprocess_data(self, df):
        """Preprocess data: handle missing values, scale features"""
        df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
        df = df.dropna()  # Drop missing values
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    def detect_anomalies_isolation_forest(self, df):
        """Detect anomalies using Isolation Forest"""
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_if'] = model.fit_predict(df)
        df['anomaly_if'] = df['anomaly_if'].apply(lambda x: 1 if x == -1 else 0)
        return df

    def detect_anomalies_zscore(self, df):
        """Detect anomalies using Z-score method"""
        df['zscore'] = np.abs(zscore(df))
        df['anomaly_zscore'] = (df['zscore'] > 3).astype(int)
        df.drop(columns=['zscore'], inplace=True)
        return df

    def detect_anomalies_iqr(self, df):
        """Detect anomalies using IQR method"""
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df['anomaly_iqr'] = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1).astype(int)
        return df

    def detect_anomalies_autoencoder(self, df):
        """Detect anomalies using Autoencoder"""
        input_dim = df.shape[1]
        model = Sequential([
            Dense(input_dim // 2, activation='relu', input_shape=(input_dim,)),
            Dense(input_dim // 4, activation='relu'),
            Dense(input_dim // 2, activation='relu'),
            Dense(input_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        model.fit(df, df, epochs=50, batch_size=16, verbose=0)
        reconstructions = model.predict(df)
        mse = np.mean(np.abs(reconstructions - df), axis=1)
        df['anomaly_autoencoder'] = (mse > np.percentile(mse, 95)).astype(int)
        return df

    def process_sheets(self):
        """Process each sheet in the Excel file"""
        results = {}
        for sheet_name, df in self.data.items():
            logging.info(f"Processing sheet: {sheet_name}")
            df_processed = self.preprocess_data(df)
            df_processed = self.detect_anomalies_isolation_forest(df_processed)
            df_processed = self.detect_anomalies_zscore(df_processed)
            df_processed = self.detect_anomalies_iqr(df_processed)
            df_processed = self.detect_anomalies_autoencoder(df_processed)
            df['Anomaly'] = (df_processed[['anomaly_if', 'anomaly_zscore', 'anomaly_iqr', 'anomaly_autoencoder']].sum(axis=1) > 1).astype(int)
            results[sheet_name] = df
        return results

    def save_results(self, results):
        """Save the results to an Excel file"""
        output_file = "anomaly_results.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name, df in results.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.info(f"Results saved to {output_file}")

    def run(self):
        """Run the full anomaly detection pipeline"""
        self.read_excel()
        results = self.process_sheets()
        self.save_results(results)
        print("Anomaly detection completed. Results saved in 'anomaly_results.xlsx'.")

if __name__ == "__main__":
    input_file = "data.xlsx"  # Replace with actual file path
    detector = AnomalyDetector(input_file)
    detector.run()
