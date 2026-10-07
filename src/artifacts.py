import joblib

def load_models(fraud_model_path, fs_model_path, iso_for_model_path):
    xgb_model = joblib.load(fraud_model_path)
    xgb_fs_model = joblib.load(fs_model_path)
    iso_forest = joblib.load(iso_for_model_path)

    return xgb_model, xgb_fs_model, iso_forest

def load_encoders(encoder_path, target_encoder_path):
    encoder = joblib.load(encoder_path)
    target_encoder = joblib.load(target_encoder_path)
    return encoder, target_encoder

