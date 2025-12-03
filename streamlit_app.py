"""
Streamlit frontend for the UDPP (Universal Data Processing Pipeline) project.
"""
import streamlit as st
import pandas as pd
import os
import tempfile
from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils import read_data, ensure_dir
from src.preprocess import preprocess_dataframe, build_preprocessor, detect_column_types
from src.models import get_model_candidates
from src.trainer import Trainer
from src.evaluate import evaluate_model
from src.config import BASE_OUTPUT
import joblib

st.set_page_config(page_title="UDPP - Data Pipeline", layout="wide")

st.title("🚀 Universal Data Processing Pipeline (UDPP)")
st.markdown("Upload your dataset and configure the ML pipeline")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Step 1: Data Upload
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Save temporary file and load data
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        df = pd.read_csv(tmp_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(tmp_path)
        st.success(f"✅ Loaded: {uploaded_file.name} | Shape: {df.shape}")
        
        # Display data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Columns:** {list(df.columns)}")
            st.write(f"**Data types:** {df.dtypes.to_dict()}")
            st.write(f"**Missing values:** {df.isnull().sum().to_dict()}")

        # Step 2: Select Target Column
        st.header("Step 2: Select Target Column")
        target_column = st.selectbox(
            "Choose the target column for prediction",
            options=df.columns,
            index=len(df.columns) - 1
        )

        if target_column not in df.columns:
            st.error("❌ Target column not found in dataset")
        else:
            # Step 3: Task Type Detection
            st.header("Step 3: Task Type")
            target_ser = df[target_column]
            
            # Auto-detect task
            if target_ser.dtype == 'object' or target_ser.nunique() <= 20:
                suggested_task = "classification"
            else:
                suggested_task = "regression"
            
            task = st.radio(
                "Select task type:",
                options=["classification", "regression"],
                index=0 if suggested_task == "classification" else 1,
                help=f"Auto-suggested: {suggested_task}"
            )

            # Step 4: Advanced Options
            st.header("Step 4: Advanced Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                test_size = st.slider(
                    "Test Set Size (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Percentage of data to use for testing"
                )

            with col2:
                random_state = st.number_input(
                    "Random Seed",
                    value=42,
                    help="For reproducible results"
                )

            with col3:
                top_k_categories = st.number_input(
                    "Top Categories",
                    min_value=5,
                    max_value=100,
                    value=30,
                    help="Max number of categories to keep"
                )

            # Step 5: Run Pipeline
            st.header("Step 5: Train Model")
            
            if st.button("🎯 Train Model", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing..."):
                    try:
                        # Create output directory
                        dataset_name = Path(uploaded_file.name).stem
                        dataset_output = os.path.join(BASE_OUTPUT, dataset_name)
                        ensure_dir(dataset_output)

                        st.info(f"📁 Output directory: {dataset_output}")

                        # Step 1: Detect types
                        st.write("🔍 Detecting column types...")
                        types = detect_column_types(df.drop(columns=[target_column]))
                        datetime_cols = types.get("datetime", None)

                        # Step 2: Preprocess
                        st.write("🔧 Preprocessing data...")
                        processed_df, numeric, categorical = preprocess_dataframe(
                            df, datetime_cols=datetime_cols
                        )
                        features = numeric + categorical
                        st.write(f"✅ Features: {len(features)} (numeric: {len(numeric)}, categorical: {len(categorical)})")

                        # Step 3: Build preprocessor
                        st.write("⚙️ Building preprocessor...")
                        preprocessor = build_preprocessor(processed_df, numeric, categorical)

                        # Step 4: Get models
                        st.write("📦 Loading model candidates...")
                        models = get_model_candidates(task=task)
                        st.write(f"✅ Models available: {list(models.keys())}")

                        # Step 5: Train
                        st.write("🚂 Training models...")
                        trainer = Trainer(
                            processed_df,
                            features,
                            target_column,
                            task,
                            preprocessor,
                            output_dir=dataset_output
                        )
                        result = trainer.train_and_select(models)
                        st.success(f"🏆 Best model: {result['best_model_name']}")

                        # Step 6: Evaluate
                        st.write("📈 Evaluating model...")
                        test_split = pd.read_csv(result['test_split_path'])
                        X_test = test_split[features]
                        y_test = test_split[target_column]
                        pipeline = joblib.load(result['model_path'])
                        metrics = evaluate_model(pipeline, X_test, y_test, task, dataset_output)

                        # Display Results
                        st.success("✅ Pipeline completed successfully!")
                        
                        st.header("📊 Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Model Information")
                            st.write(f"**Best Model:** {result['best_model_name']}")
                            st.write(f"**Task:** {task}")
                            st.write(f"**Features Used:** {len(features)}")
                            st.write(f"**Training Samples:** {len(processed_df) - len(test_split)}")
                            st.write(f"**Test Samples:** {len(test_split)}")

                        with col2:
                            st.subheader("Performance Metrics")
                            for metric_name, metric_value in metrics.items():
                                if isinstance(metric_value, float):
                                    st.metric(metric_name, f"{metric_value:.4f}")
                                else:
                                    st.write(f"**{metric_name}:** {metric_value}")

                        # Show saved files
                        st.subheader("📁 Generated Files")
                        output_files = []
                        if os.path.exists(dataset_output):
                            for file in os.listdir(dataset_output):
                                file_path = os.path.join(dataset_output, file)
                                if os.path.isfile(file_path):
                                    output_files.append(file)
                        
                        if output_files:
                            st.write("Files saved in output directory:")
                            for file in output_files:
                                st.write(f"• {file}")

                        st.balloons()

                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
                        st.exception(e)

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        os.unlink(tmp_path)

else:
    st.info("👆 Please upload a CSV or Excel file to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    **UDPP** - Universal Data Processing Pipeline | 
    [GitHub](https://github.com) | 
    [Documentation](https://github.com)
    """
)
