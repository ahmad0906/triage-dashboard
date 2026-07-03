import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the exact model file you are using
model = joblib.load('02_optimized_lightgbm_pipeline.pkl')

print("\n--- The 23 Columns Expected by the Model ---")
try:
    # For standard scikit-learn pipelines
    print(list(model.feature_names_in_))
except AttributeError:
    # For direct LightGBM models
    print(list(model.feature_name_))
print("--------------------------------------------\n")