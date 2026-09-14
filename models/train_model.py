import os
import sys
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.isotonic import IsotonicRegression


def create_training_matrix(df):
    """Converts the SQL dataframe into the [time x sector] matrix format."""
    print("Creating training matrix...")
    
    df['time_idx'] = (df['year'] - 2019) * 12 + df['month']
    
    matrix = df.pivot_table(
        index='time_idx', 
        columns='sector', 
        values='amount_new_house_transactions',
        fill_value=0
    )
    
    expected_sectors = [f'sector {i}' for i in range(1, 97)]
    for sector in expected_sectors:
        if sector not in matrix.columns:
            matrix[sector] = 0
            
    matrix = matrix.reindex(sorted(matrix.columns, key=lambda x: int(x.split()[-1])), axis=1)
    return matrix

def create_features_at_time(train_matrix: pd.DataFrame, time_idx: int, max_history=18):
    """Creates time-series and rolling volatility features for all sectors at time t."""
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    df_panel.index.name = 'time'
    
    features_list, sector_list = [], []
    
    for sector in df_panel.columns:
        series = df_panel[sector].dropna()
        hist = series[series.index <= time_idx]
        if len(hist) < max_history:
            continue
            
        f = {}
        f['sector_id'] = int(str(sector).replace('sector ', '')) / 100.0
        
        month_t = ((time_idx - 1) % 12) + 1
        f['month_sin'] = np.sin(2*np.pi*month_t/12)
        f['month_cos'] = np.cos(2*np.pi*month_t/12)
        f['quarter'] = ((month_t - 1) // 3 + 1) / 4.0
        
        for lag in [1, 2, 3, 6, 12]:
            if len(hist) >= lag:
                val = float(hist.iloc[-lag])
                f[f'lag_{lag}_log1p'] = np.log1p(val)
                f[f'lag_{lag}_zero'] = int(val == 0)
            else:
                f[f'lag_{lag}_log1p'] = 0.0
                f[f'lag_{lag}_zero'] = 1
                
        for w in [3, 6, 12]:
            if len(hist) >= w:
                roll = hist.tail(w)
                f[f'roll_{w}_mean_log1p'] = np.log1p(roll.mean())
                f[f'roll_{w}_zero_rate'] = float((roll == 0).mean())
            else:
                f[f'roll_{w}_mean_log1p'] = 0.0
                f[f'roll_{w}_zero_rate'] = 1.0
                
        features_list.append(f)
        sector_list.append(sector)
        
    features_df = pd.DataFrame(features_list)
    features_df['sector'] = sector_list
    features_df['time'] = int(time_idx)
    return features_df

def create_training_data_for_horizon(train_matrix, horizon, max_history=18, max_train_time=None):
    """Aligns features at time t with targets at time t + horizon."""
    df_panel = train_matrix.apply(pd.to_numeric, errors='coerce').sort_index()
    min_t = int(df_panel.index.min() + max_history)
    max_t_possible = int(df_panel.index.max() - horizon)
    max_t = int(min(max_t_possible, max_train_time)) if max_train_time is not None else max_t_possible
    
    valid_times = [int(t) for t in df_panel.index if min_t <= t <= max_t]
    frames = []
    
    for t in valid_times:
        feat_t = create_features_at_time(df_panel, t, max_history=max_history)
        if feat_t.empty:
            continue
        tgt_time = t + horizon
        feat_t['target'] = df_panel.loc[tgt_time, feat_t['sector']].values.astype(float)
        frames.append(feat_t)
        
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class HorizonSpecificModel:
    def __init__(self, horizon, ridge_alpha=0.73, logistic_C=0.07, use_isotonic=True):
        self.horizon = int(horizon)
        self.ridge_alpha = float(ridge_alpha)
        self.logistic_C = float(logistic_C)
        self.use_isotonic = bool(use_isotonic)
        self.scaler = None
        self.zero_model = None
        self.count_model = None
        self.iso_ = None
        self.is_fitted = False
        self.zero_threshold_ = 0.5
        self.mu_cap_ = 1e12

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X)

        self.zero_model = LogisticRegression(C=self.logistic_C, class_weight='balanced', max_iter=500, random_state=42)
        self.zero_model.fit(Xs, (y == 0).astype(int))

        if self.use_isotonic:
            pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
            self.iso_ = IsotonicRegression(out_of_bounds='clip').fit(pi_raw, (y == 0).astype(int))

        mask_pos = y > 0
        if mask_pos.sum() > 10:
            y_log = np.log1p(y[mask_pos])
            self.count_model = Ridge(alpha=self.ridge_alpha, solver='svd')
            self.count_model.fit(Xs[mask_pos], y_log)
            self.mu_cap_ = float(np.quantile(y[mask_pos], 0.995))
        else:
            self.count_model = None
            self.mu_cap_ = 1e12

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        pi_raw = self.zero_model.predict_proba(Xs)[:, 1]
        p0 = self.iso_.transform(pi_raw) if self.iso_ is not None else pi_raw
        
        if self.count_model is not None:
            mu = np.clip(np.expm1(self.count_model.predict(Xs)), 0.0, self.mu_cap_)
        else:
            mu = np.zeros(len(Xs), dtype=float)
            
        yhat = (1.0 - p0) * mu
        yhat[p0 >= self.zero_threshold_] = 0.0
        return np.maximum(yhat, 0.0)

class MultiHorizonForecaster:
    def __init__(self, horizons=12, max_history=18):
        self.horizons = list(range(1, horizons + 1))
        self.max_history = int(max_history)
        self.models = {}
        self.feature_cols = None

    def fit(self, train_matrix: pd.DataFrame):
        max_train_time = int(train_matrix.index.max())
        
        for h in self.horizons:
            dfh = create_training_data_for_horizon(train_matrix, h, max_history=self.max_history, max_train_time=max_train_time)
            if dfh.empty:
                continue
                
            if self.feature_cols is None:
                self.feature_cols = [c for c in dfh.columns if c not in ['sector', 'time', 'target', 'target_time']]
                
            X = dfh[self.feature_cols].astype(float).values
            y = dfh['target'].astype(float).values
            
            model = HorizonSpecificModel(horizon=h)
            model.fit(X, y)
            self.models[h] = model
            print(f"✅ Trained Horizon {h} model.")
            
        return self


def save_forecast_for_streamlit(forecaster, train_matrix):
    """Generates 12-month forecast and saves to CSV for Streamlit app."""
    print("Generating predictions for Streamlit...")
    
    last_train = int(train_matrix.index.max())
    
    preds = {}
    
    Xh = create_features_at_time(train_matrix, last_train, max_history=16)
    
    for c in forecaster.feature_cols:
        if c not in Xh.columns:
            Xh[c] = 0.0
            
    X = Xh[forecaster.feature_cols].astype(float).values
    
    for h in range(1, 13):
        preds[last_train + h] = forecaster.models[h].predict(X)
        
    sectors = Xh['sector'].values
    pred_df = pd.DataFrame(preds, index=sectors)
    
    pred_long = pred_df.reset_index().melt(id_vars='index', var_name='time_idx', value_name='prediction')
    pred_long.rename(columns={'index': 'sector'}, inplace=True)
    
    start_date = pd.to_datetime('2019-01-01')
    pred_long['date'] = pred_long['time_idx'].apply(lambda x: start_date + pd.DateOffset(months=int(x)-1))
    
    os.makedirs("/app/data/", exist_ok=True)
    pred_long.to_csv("/app/data/forecast_68_79.csv", index=False)
    print("🎉 Predictions saved to /app/data/forecast_68_79.csv")



def train_models():
    print("Starting data extraction...")
    db_path = "/app/src/database/real_estate.db"
    
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )

    sql = """
    WITH BaseMonthSector AS (
        SELECT month, sector FROM land_transactions
        UNION SELECT month, sector FROM land_transactions_nearby_sectors
        UNION SELECT month, sector FROM new_house_transactions
        UNION SELECT month, sector FROM new_house_transactions_nearby_sectors
        UNION SELECT month, sector FROM pre_owned_house_transactions
        UNION SELECT month, sector FROM pre_owned_house_transactions_nearby_sectors
    )
    SELECT
        CAST(SUBSTR(b.month, 1, 4) AS INTEGER) AS year,
        CASE SUBSTR(b.month, 6)
            WHEN 'Jan' THEN 1 WHEN 'Feb' THEN 2 WHEN 'Mar' THEN 3 WHEN 'Apr' THEN 4
            WHEN 'May' THEN 5 WHEN 'Jun' THEN 6 WHEN 'Jul' THEN 7 WHEN 'Aug' THEN 8
            WHEN 'Sep' THEN 9 WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11 WHEN 'Dec' THEN 12
            ELSE CAST(SUBSTR(b.month, 6) AS INTEGER)
        END AS month,
        b.sector,
        nht.amount_new_house_transactions
    FROM BaseMonthSector b
    LEFT JOIN new_house_transactions nht ON b.month = nht.month AND b.sector = nht.sector;
    """
    
    df = pd.read_sql(sql, engine)
    print(f"Loaded {len(df):,} rows from database.")
    
    train_matrix = create_training_matrix(df)
    
    print("Training Multi-Horizon Forecaster...")
    forecaster = MultiHorizonForecaster(horizons=12, max_history=16)
    forecaster.fit(train_matrix)
    
    model_dir = "/app/models/"
    os.makedirs(model_dir, exist_ok=True)
    
    save_path = os.path.join(model_dir, "horizon_ridge_forecaster.joblib")
    joblib.dump(forecaster, save_path)
    print(f"✅ Model successfully saved to {save_path}")
    
    save_forecast_for_streamlit(forecaster, train_matrix)

if __name__ == "__main__":
    train_models()