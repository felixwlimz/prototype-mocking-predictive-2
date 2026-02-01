"""
VIVA COSMETICS - PREDICTIVE AI MODELS
======================================
Complete implementation of Traditional ML & Deep Learning
for sales prediction

Dataset: viva_cosmetics_dl_dataset.csv (57,457 samples)
Target: unit_terjual (sales prediction)

Models:
1. XGBoost (Recommended)
2. LightGBM (Recommended)
3. Random Forest
4. Deep Neural Network
5. Ensemble (Stacking)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from model import PredictorNet
import torch.nn as nn

warnings.filterwarnings('ignore')


# ============================================================================
# 1. LOAD & PREPROCESSING DATA
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load dataset and perform preprocessing"""
    print("=" * 80)
    print("STEP 1: LOADING & PREPROCESSING DATA")
    print("=" * 80)

    # Load data
    df = pd.read_csv(filepath)
    print(f"\n✓ Dataset loaded: {len(df):,} samples")
    print(f"✓ Features: {len(df.columns)} columns")
    print(f"✓ Period: {df['tahun'].min()}-{df['tahun'].max()}")

    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠ Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values")

    # Separate features and target
    X = df.drop('unit_terjual', axis=1)
    y = df['unit_terjual']

    # Identify categorical and numerical columns
    categorical_cols = ['provinsi', 'kota', 'produk', 'kategori', 'usia_target',
                        'gender_target', 'tingkat_urbanisasi', 'intensitas_iklan',
                        'promo_aktif', 'musim_spesial']

    numerical_cols = ['bulan', 'tahun', 'harga', 'kepadatan_penduduk',
                      'daya_beli_index', 'jumlah_toko_retail', 'kompetitor_count',
                      'rating_produk', 'stok_tersedia']

    print(f"\n✓ Categorical features: {len(categorical_cols)}")
    print(f"✓ Numerical features: {len(numerical_cols)}")

    # Label encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    print(f"\n✓ Categorical encoding completed")
    print(f"✓ Total features after encoding: {X_encoded.shape[1]}")

    return X_encoded, y, label_encoders, categorical_cols, numerical_cols


# ============================================================================
# 2. SPLIT DATA
# ============================================================================

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING DATA")
    print("=" * 80)

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"\n✓ Training set:   {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"✓ Validation set: {len(X_val):,} samples ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"✓ Test set:       {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# 3. EVALUATION METRICS
# ============================================================================

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate and print evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{model_name} Performance:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  RMSE:      {rmse:.2f}")
    print(f"  MAE:       {mae:.2f}")
    print(f"  MAPE:      {mape:.2f}%")

    return {
        'model': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


# ============================================================================
# 4. XGBOOST MODEL
# ============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost model"""
    print("\n" + "=" * 80)
    print("STEP 3A: TRAINING XGBOOST MODEL")
    print("=" * 80)

    try:
        import xgboost as xgb
    except ImportError:
        print("\n⚠ XGBoost not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'xgboost', '--break-system-packages'])
        import xgboost as xgb

    # Train model
    print("\n🚀 Training XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20,
        n_jobs=-1

    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Evaluate
    print("\n📊 XGBOOST RESULTS:")
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")

    return model, test_metrics


# ============================================================================
# 5. LIGHTGBM MODEL
# ============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train LightGBM model"""
    print("\n" + "=" * 80)
    print("STEP 3B: TRAINING LIGHTGBM MODEL")
    print("=" * 80)

    try:
        import lightgbm as lgb
    except ImportError:
        print("\n⚠ LightGBM not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'lightgbm', '--break-system-packages'])
        import lightgbm as lgb

    # Train model
    print("\n🚀 Training LightGBM...")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Evaluate
    print("\n📊 LIGHTGBM RESULTS:")
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")

    return model, test_metrics


# ============================================================================
# 6. RANDOM FOREST MODEL
# ============================================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model"""
    print("\n" + "=" * 80)
    print("STEP 3C: TRAINING RANDOM FOREST MODEL")
    print("=" * 80)

    from sklearn.ensemble import RandomForestRegressor

    print("\n🚀 Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate
    print("\n📊 RANDOM FOREST RESULTS:")
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")

    return model, test_metrics


# ============================================================================
# 7. DEEP LEARNING MODEL (NEURAL NETWORK)
# ============================================================================

def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Deep Neural Network using PyTorch"""
    print("\n" + "=" * 80)
    print("STEP 3D: TRAINING DEEP NEURAL NETWORK (PYTORCH)")
    print("=" * 80)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # 1. Scale features
    print("\n🔄 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 2. Convert to PyTorch Tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    X_val_t = torch.FloatTensor(X_val_scaled).to(device)
    y_val_t = torch.FloatTensor(y_val.values).view(-1, 1).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = PredictorNet(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning Rate Scheduler (Equivalent to ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    print("\n🚀 Training neural network...")
    print("   Architecture: [256 → 128 → 64 → 1]")

    # 4. Training Loop (with Early Stopping)
    epochs = 100
    patience = 20
    best_loss = float('inf')
    counter = 0
    best_model_state = None

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            v_loss = criterion(val_outputs, y_val_t).item()

        avg_train_loss = np.mean(batch_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(v_loss)

        scheduler.step(v_loss)

        # Early Stopping Logic (Equivalent to restore_best_weights)
        if v_loss < best_loss:
            best_loss = v_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    # Load best weights
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"✓ Training completed")

    # 5. Predictions
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_t).cpu().numpy().flatten()
        y_val_pred = model(X_val_t).cpu().numpy().flatten()
        y_test_pred = model(X_test_t).cpu().numpy().flatten()

    # 6. Evaluate
    print("\n📊 NEURAL NETWORK RESULTS:")
    train_metrics = evaluate_model(y_train, y_train_pred, "Training")
    val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
    test_metrics = evaluate_model(y_test, y_test_pred, "Test")

    return model, scaler, test_metrics

# ============================================================================
# 8. COMPARE ALL MODELS
# ============================================================================

def compare_models(results):
    """Compare all models and visualize results"""
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('r2', ascending=False)

    print("\n📊 Test Set Performance Ranking:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Model':<25} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<10}")
    print("-" * 80)

    for idx, row in df_results.iterrows():
        rank = "🥇" if idx == df_results.index[0] else "🥈" if idx == df_results.index[1] else "🥉" if idx == \
                                                                                                    df_results.index[
                                                                                                        2] else "  "
        print(
            f"{rank:<6} {row['model']:<25} {row['r2']:<10.4f} {row['rmse']:<12.2f} {row['mae']:<12.2f} {row['mape']:<10.2f}%")

    print("-" * 80)

    # Best model
    best_model = df_results.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['model']}")
    print(f"   R² Score: {best_model['r2']:.4f}")
    print(f"   RMSE: {best_model['rmse']:.2f}")

    return df_results


# ============================================================================
# 9. FEATURE IMPORTANCE (FOR TREE MODELS)
# ============================================================================

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Plot feature importance for tree-based models"""
    print(f"\n📊 Feature Importance - {model_name}")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}.png', dpi=300,
                    bbox_inches='tight')
        print(f"✓ Plot saved: feature_importance_{model_name.lower().replace(' ', '_')}.png")

        print(f"\nTop {min(10, top_n)} Features:")
        for i in range(min(10, top_n)):
            print(f"  {i + 1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n")
    print("=" * 80)
    print(" " * 15 + "V COSMETICS - PREDICTIVE AI")
    print(" " * 20 + "Sales Forecasting Models")
    print("=" * 80)

    # Load data
    filepath = 'v_cosmetics_dataset.csv'
    X, y, label_encoders, categorical_cols, numerical_cols = load_and_preprocess_data(filepath)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Store results
    results = []

    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(xgb_metrics)
    plot_feature_importance(xgb_model, X.columns.tolist(), "XGBoost")

    # Train LightGBM
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(lgb_metrics)
    plot_feature_importance(lgb_model, X.columns.tolist(), "LightGBM")

    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_metrics)
    plot_feature_importance(rf_model, X.columns.tolist(), "Random Forest")

    # Train Neural Network
    nn_model, scaler, nn_metrics = train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test)
    results.append(nn_metrics)

    # Compare all models
    comparison_df = compare_models(results)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print("\n✅ All models trained successfully")
    print("✅ Feature importance plots saved")
    print("\n💡 Recommendation: Use the best performing model for production")

    return {
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'rf_model': rf_model,
        'nn_model': nn_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'comparison': comparison_df
    }


if __name__ == "__main__":
    models = main()