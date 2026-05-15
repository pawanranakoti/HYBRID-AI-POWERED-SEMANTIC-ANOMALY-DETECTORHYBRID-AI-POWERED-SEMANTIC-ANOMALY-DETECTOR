import json
import io
import warnings
import getpass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from google import genai
from google.genai import types

warnings.filterwarnings("ignore")

# MAIN CLASS

class ImprovedUniversalAnomalyAgent:

    def __init__(self, api_key=None):
        self.client = None

        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                print("✅ Ai Powered Semantic Anomaly Detector Enabled")
            except Exception as e:
                print(f"⚠ Gemini initialization failed: {e}")

        self.stat_flags = set()
        self.ai_flags = []
        self.row_anomalies = None

    # INPUT LOADER
   
    def get_dataframe(self, user_input):

        try:
            if user_input.endswith(".csv"):
                df = pd.read_csv(user_input)
            else:
                df = pd.read_csv(io.StringIO(user_input))

            print(f"✅ Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            return df

        except Exception as e:
            raise Exception(f"❌ Failed to load data: {e}")

# FEATURE TYPE INFERENCE
    
    def infer_feature_type(self, col_name, series):

        name = col_name.lower()

        if "date" in name or "time" in name:
            return "datetime"

        if "id" in name:
            return "identifier"

        if "email" in name:
            return "email"

        if "price" in name or "amount" in name or "salary" in name:
            return "financial"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        return "categorical"

   
    # DYNAMIC STATISTICAL ANALYSIS
    
    def generate_dynamic_dna(self, df):

        print("📈 Running Dynamic Statistical Engine...")

        dna = {
            "dataset_shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "features": {}
        }

        self.stat_flags.clear()

        missing_rates = df.isna().mean()

        dataset_missing_mean = missing_rates.mean()
        dataset_missing_std = missing_rates.std()

        dynamic_missing_threshold = (
            dataset_missing_mean + (2 * dataset_missing_std)
        )

        # Correlation Analysis
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        high_correlations = []

        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()

            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    corr_value = corr_matrix.iloc[i, j]

                    if corr_value > 0.95:
                        high_correlations.append({
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": round(corr_value, 3)
                        })

        dna["high_correlations"] = high_correlations

        # Feature Analysis
        for col in df.columns:

            feature_type = self.infer_feature_type(col, df[col])
            missing_rate = missing_rates[col]

            feature_info = {
                "type": feature_type,
                "missing_rate": round(float(missing_rate), 4)
            }

           
            # Dynamic Missing Data Detection
            
            if (
                missing_rate > dynamic_missing_threshold
                and missing_rate > 0.1
            ):
                self.stat_flags.add(col)

            
            # NUMERIC FEATURES
            
            if pd.api.types.is_numeric_dtype(df[col]):

                clean_series = df[col].dropna()

                if len(clean_series) > 0:

                    Q1 = clean_series.quantile(0.25)
                    Q3 = clean_series.quantile(0.75)
                    IQR = Q3 - Q1

                    lower = Q1 - (1.5 * IQR)
                    upper = Q3 + (1.5 * IQR)

                    outliers = (
                        (clean_series < lower)
                        |
                        (clean_series > upper)
                    ).sum()

                    outlier_rate = outliers / len(clean_series)

                    skewness = clean_series.skew()

                    feature_info.update({
                        "min": float(clean_series.min()),
                        "max": float(clean_series.max()),
                        "mean": float(clean_series.mean()),
                        "std": float(clean_series.std()),
                        "skewness": round(float(skewness), 3),
                        "outlier_rate": round(float(outlier_rate), 3)
                    })

                    # Dynamic statistical flagging
                    if outlier_rate > 0.12:
                        self.stat_flags.add(col)

                    if abs(skewness) > 5:
                        self.stat_flags.add(col)

            
            # CATEGORICAL FEATURES
            
            else:

                unique_ratio = df[col].nunique(dropna=True) / len(df)

                top_value = (
                    str(df[col].mode().iloc[0])
                    if not df[col].mode().empty
                    else "N/A"
                )

                feature_info.update({
                    "unique_ratio": round(float(unique_ratio), 3),
                    "top_value": top_value
                })

                # High cardinality anomaly
                if unique_ratio > 0.95 and len(df) > 100:
                    self.stat_flags.add(col)

            dna["features"][col] = feature_info

        return dna

    
    # PREPARE DATA FOR ML
    
    def prepare_ml_data(self, df):

        working_df = df.copy()

        # Encode categorical variables
        for col in working_df.columns:

            if not pd.api.types.is_numeric_dtype(working_df[col]):
                working_df[col] = working_df[col].astype(str)

                encoder = LabelEncoder()
                working_df[col] = encoder.fit_transform(working_df[col])

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(working_df)

        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X

    
    # ROW-LEVEL ANOMALY DETECTION
   
    def detect_row_anomalies(self, df):

        print("🤖 Running Isolation Forest Engine...")

        X = self.prepare_ml_data(df)

        model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )

        predictions = model.fit_predict(X)
        scores = model.decision_function(X)

        result_df = df.copy()

        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = predictions

        anomalies = result_df[result_df["is_anomaly"] == -1]
        anomalies = anomalies.sort_values("anomaly_score")

        self.row_anomalies = anomalies

        print(f"⚠ Row-Level Anomalies Found: {len(anomalies)}")

        return anomalies

    
    # SEMANTIC ANALYSIS
   
    def audit_semantics(self, dna):

        if self.client is None:
            print("⚠ Detector Disabled")
            return []

        # Smart Gemini Optimization
        if len(self.stat_flags) == 0:
            print("✅ No suspicious statistical patterns. Skipping .")
            return []

        print("🧠 Running Semantic Audit...")

        prompt = f"""
        You are a senior data quality auditor.

        Analyze the following dataset DNA.

        Detect ONLY semantic/logical anomalies.

        Examples:
        - impossible ages
        - negative salaries
        - contradictory ranges
        - invalid semantic relationships

        IMPORTANT:
        - Ignore statistical outliers.
        - Focus ONLY on logical impossibilities.
        - Be conservative.

        DATASET DNA:
        {json.dumps(dna, indent=2)}

        Return ONLY valid JSON:

        [
          {{
            "column": "column_name",
            "issue": "explanation",
            "severity": "HIGH/MEDIUM"
          }}
        ]

        Return [] if no semantic anomalies exist.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )

            return json.loads(response.text)

        except Exception as e:
            print(f"⚠ Gemini Error: {e}")
            return []

    
    # CONFIDENCE SCORING
    
    def calculate_confidence_score(self, column, ai_set):

        score = 0

        if column in self.stat_flags:
            score += 60

        if column in ai_set:
            score += 40

        return min(score, 100)

    
    # DASHBOARD
    
    def plot_dashboard(self, df, ai_set):

        print("📊 Generating Dashboard...")

        both = len(self.stat_flags & ai_set)
        ai_only = len(ai_set - self.stat_flags)
        stat_only = len(self.stat_flags - ai_set)

        clean = len(df.columns) - (
            both + ai_only + stat_only
        )

        fig = plt.figure(figsize=(18, 12))

        fig.suptitle(
            "🚀 Improved Universal Anomaly Dashboard",
            fontsize=20,
            fontweight="bold"
        )

        
        # 1. HEATMAP
        
        ax1 = fig.add_subplot(221)

        matrix = [
            [both, ai_only],
            [stat_only, clean]
        ]

        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='magma',
            cbar=False,
            xticklabels=['AI Anomaly', 'AI Normal'],
            yticklabels=['Math Anomaly', 'Math Normal'],
            ax=ax1
        )

        ax1.set_title("Convergence Matrix")

        
        # 2. PIE CHART
       
        ax2 = fig.add_subplot(222)

        labels = [
            'Clean',
            'Math Only',
            'Semantic Only',
            'Critical'
        ]

        values = [
            clean,
            stat_only,
            ai_only,
            both
        ]

        filtered = [
            (l, v)
            for l, v in zip(labels, values)
            if v > 0
        ]

        if filtered:
            flabels, fvalues = zip(*filtered)

            ax2.pie(
                fvalues,
                labels=flabels,
                autopct='%1.1f%%',
                startangle=140
            )

        ax2.set_title("Dataset Health")

       
        # 3. CORRELATION HEATMAP
        
        ax3 = fig.add_subplot(223)

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()

            sns.heatmap(
                corr,
                cmap='coolwarm',
                center=0,
                ax=ax3
            )

            ax3.set_title("Feature Correlation Map")

        
        # 4. ANOMALY SCORE DISTRIBUTION
        
        ax4 = fig.add_subplot(224)

        if self.row_anomalies is not None:

            scores = self.row_anomalies["anomaly_score"]

            ax4.hist(scores, bins=20)
            ax4.set_title("Row-Level Anomaly Score Distribution")
            ax4.set_xlabel("Anomaly Score")
            ax4.set_ylabel("Frequency")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    
    # MAIN ANALYSIS PIPELINE
    
    def analyze(self, user_input):

        df = self.get_dataframe(user_input)

        # Dynamic Statistical Analysis
        dna = self.generate_dynamic_dna(df)

        # Row-Level ML Detection
        anomalies = self.detect_row_anomalies(df)

        # Semantic AI Analysis
        self.ai_flags = self.audit_semantics(dna)

        ai_set = {
            item["column"]
            for item in self.ai_flags
        }

        # Dashboard
        self.plot_dashboard(df, ai_set)

       
        # FINAL REPORT
        
        print("\n" + "=" * 70)
        print("🛡 FINAL DATA INTEGRITY REPORT")
        print("=" * 70)

        print(f"\n📌 Statistical Flags : {len(self.stat_flags)}")
        print(f"📌 Semantic Flags    : {len(ai_set)}")
        print(f"📌 Row Anomalies     : {len(anomalies)}")

        print("\n🚨 COLUMN-LEVEL ISSUES")
        print("-" * 70)

        all_columns = sorted(
            list(self.stat_flags.union(ai_set))
        )

        if len(all_columns) == 0:
            print("✅ No major column-level anomalies found.")

        for col in all_columns:

            confidence = self.calculate_confidence_score(col, ai_set)

            status = "🔥 CRITICAL" if confidence >= 80 else "⚠ MODERATE"

            print(f"\n{status} | {col}")
            print(f"Confidence Score : {confidence}%")

            if col in self.stat_flags:
                print("• Statistical anomaly detected")

            for item in self.ai_flags:
                if item["column"] == col:
                    print(f"• Semantic issue: {item['issue']}")

        
        # TOP ROW ANOMALIES
        
        print("\n🚨 TOP ANOMALOUS ROWS")
        print("-" * 70)

        if len(anomalies) > 0:
            print(
                anomalies.head(10).to_string()
            )
        else:
            print("✅ No major row-level anomalies found.")



# RUNNER

if __name__ == "__main__":

    api_key = ""

    try:
        from kaggle_secrets import UserSecretsClient

        api_key = UserSecretsClient().get_secret(
            "GEMINI_API_KEY"
        )

    except:
        api_key = getpass.getpass(
            "🔑 Enter Gemini API Key (Optional): "
        )

    agent = ImprovedUniversalAnomalyAgent(api_key)

    print("\n--- UNIVERSAL INPUT ---")
    print("Paste OR Enter CSV file path")

    user_input = input(">> ").strip()

    if user_input:
        agent.analyze(user_input)
