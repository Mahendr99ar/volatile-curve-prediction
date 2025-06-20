import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class CompetitionSubmission:
    """
    Complete submission pipeline for NIFTY50 volatility prediction competition
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.predictor = None
        self.deep_predictor = None
        self.ensemble_predictor = None
        self.validation_results = {}
        
    def load_competition_data(self, train_file=None, test_file=None):
        """
        Load competition data files
        """
        print("📥 Loading competition data...")
        
        if train_file:
            self.train_df = pd.read_csv(train_file)
            print(f"✅ Training data loaded: {self.train_df.shape}")
        
        if test_file:
            self.test_df = pd.read_csv(test_file)
            print(f"✅ Test data loaded: {self.test_df.shape}")
            
        return self.train_df if train_file else None, self.test_df if test_file else None
    
    def exploratory_data_analysis(self, df):
        """
        Comprehensive EDA for the competition data
        """
        print("🔍 Performing Exploratory Data Analysis...")
        
        # Basic statistics
        print("\n📊 Dataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Create EDA plots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # 1. Implied Volatility distribution
        if 'implied_volatility' in df.columns:
            axes[0].hist(df['implied_volatility'].dropna(), bins=50, alpha=0.7)
            axes[0].set_title('Implied Volatility Distribution')
            axes[0].set_xlabel('Implied Volatility')
        
        # 2. Strike vs Underlying Price
        if all(col in df.columns for col in ['strike', 'underlying_price']):
            axes[1].scatter(df['underlying_price'], df['strike'], alpha=0.5)
            axes[1].set_title('Strike vs Underlying Price')
            axes[1].set_xlabel('Underlying Price')
            axes[1].set_ylabel('Strike Price')
        
        # 3. Time to Expiry distribution
        if 'time_to_expiry' in df.columns:
            axes[2].hist(df['time_to_expiry'].dropna(), bins=50, alpha=0.7)
            axes[2].set_title('Time to Expiry Distribution')
            axes[2].set_xlabel('Time to Expiry (Years)')
        elif 'days_to_expiry' in df.columns:
            axes[2].hist(df['days_to_expiry'].dropna(), bins=50, alpha=0.7)
            axes[2].set_title('Days to Expiry Distribution')
            axes[2].set_xlabel('Days to Expiry')
        
        # 4. Volume distribution
        if 'volume' in df.columns:
            axes[3].hist(np.log1p(df['volume']), bins=50, alpha=0.7)
            axes[3].set_title('Log Volume Distribution')
            axes[3].set_xlabel('Log(Volume + 1)')
        
        # 5. Open Interest distribution
        if 'open_interest' in df.columns:
            axes[4].hist(np.log1p(df['open_interest']), bins=50, alpha=0.7)
            axes[4].set_title('Log Open Interest Distribution')
            axes[4].set_xlabel('Log(Open Interest + 1)')
        
        # 6. Option Type distribution
        if 'option_type' in df.columns:
            df['option_type'].value_counts().plot(kind='bar', ax=axes[5])
            axes[5].set_title('Option Type Distribution')
            axes[5].tick_params(axis='x', rotation=45)
        
        # 7. IV vs Moneyness (if available)
        if all(col in df.columns for col in ['implied_volatility', 'moneyness']):
            scatter = axes[6].scatter(df['moneyness'], df['implied_volatility'], 
                                    c=df['time_to_expiry'] if 'time_to_expiry' in df.columns else 'blue', 
                                    alpha=0.5, cmap='viridis')
            axes[6].set_title('IV vs Moneyness')
            axes[6].set_xlabel('Moneyness')
            axes[6].set_ylabel('Implied Volatility')
            if 'time_to_expiry' in df.columns:
                plt.colorbar(scatter, ax=axes[6], label='Time to Expiry')
        
        # 8. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[7])
            axes[7].set_title('Correlation Matrix')
        
        # 9. Time series plot (if timestamp available)
        if 'timestamp' in df.columns and 'implied_volatility' in df.columns:
            df_ts = df.set_index('timestamp')['implied_volatility'].resample('1H').mean()
            axes[8].plot(df_ts.index[:100], df_ts.values[:100])  # First 100 hours
            axes[8].set_title('IV Time Series (First 100 Hours)')
            axes[8].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def prepare_competition_features(self, df):
        """
        Prepare features specifically for the competition
        """
        print("🔧 Preparing competition features...")
        
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Handle datetime columns
        datetime_cols = ['timestamp', 'expiry']
        for col in datetime_cols:
            if col in df_features.columns:
                df_features[col] = pd.to_datetime(df_features[col])
        
        # Calculate moneyness if not present
        if 'moneyness' not in df_features.columns and all(col in df_features.columns for col in ['strike', 'underlying_price']):
            df_features['moneyness'] = df_features['strike'] / df_features['underlying_price']
            df_features['log_moneyness'] = np.log(df_features['moneyness'])
        
        # Calculate time to expiry if not present
        if 'time_to_expiry' not in df_features.columns:
            if 'days_to_expiry' in df_features.columns:
                df_features['time_to_expiry'] = df_features['days_to_expiry'] / 365.25
            elif all(col in df_features.columns for col in ['timestamp', 'expiry']):
                df_features['days_to_expiry'] = (df_features['expiry'] - df_features['timestamp']).dt.days
                df_features['time_to_expiry'] = df_features['days_to_expiry'] / 365.25
        
        # Advanced feature engineering
        df_features = self._advanced_competition_features(df_features)
        
        return df_features
    
    def _advanced_competition_features(self, df):
        """
        Advanced feature engineering for competition
        """
        # Volatility smile features
        if all(col in df.columns for col in ['moneyness', 'time_to_expiry']):
            # Distance from ATM
            df['atm_distance'] = np.abs(df['moneyness'] - 1.0)
            
            # Smile curvature proxy
            df['smile_proxy'] = (df['moneyness'] - 1.0) ** 2
            
            # Term structure proxy
            df['term_structure'] = np.sqrt(df['time_to_expiry'])
        
        # Market microstructure features
        if 'bid' in df.columns and 'ask' in df.columns:
            df['bid_ask_spread'] = df['ask'] - df['bid']
            df['bid_ask_mid'] = (df['bid'] + df['ask']) / 2
            df['spread_pct'] = df['bid_ask_spread'] / df['bid_ask_mid']
        
        # Liquidity features
        if 'volume' in df.columns and 'open_interest' in df.columns:
            df['volume_oi_ratio'] = df['volume'] / (df['open_interest'] + 1)
            df['liquidity_score'] = np.log1p(df['volume']) * np.log1p(df['open_interest'])
        
        # Option Greeks proxies
        if all(col in df.columns for col in ['moneyness', 'time_to_expiry']):
            # Delta proxy
            df['delta_proxy'] = np.where(
                df['option_type'] == 'Call',
                1 / (1 + np.exp(-5 * (df['moneyness'] - 1))),
                1 / (1 + np.exp(5 * (df['moneyness'] - 1)))
            )
            
            # Gamma proxy
            df['gamma_proxy'] = np.exp(-2 * (df['moneyness'] - 1)**2) / np.sqrt(df['time_to_expiry'])
            
            # Vega proxy
            df['vega_proxy'] = np.sqrt(df['time_to_expiry']) * np.exp(-0.5 * (df['moneyness'] - 1)**2)
        
        # Temporal features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Market session features
            df['is_opening'] = (df['hour'] == 9) & (df['minute'] < 30)
            df['is_closing'] = (df['hour'] == 15) & (df['minute'] > 30)
            df['is_lunch'] = (df['hour'] == 12) | (df['hour'] == 13)
        
        # Cross-sectional features (if timestamp available)
        if 'timestamp' in df.columns:
            df = self._add_cross_sectional_features(df)
        
        return df
    
    def _add_cross_sectional_features(self, df):
        """
        Add cross-sectional features across timestamps
        """
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Rolling statistics for underlying price
        if 'underlying_price' in df.columns:
            df['underlying_ma5'] = df['underlying_price'].rolling(window=5).mean()
            df['underlying_ma20'] = df['underlying_price'].rolling(window=20).mean()
            df['underlying_volatility'] = df['underlying_price'].rolling(window=20).std()
        
        # ATM implied volatility (VIX proxy)
        if 'implied_volatility' in df.columns and 'moneyness' in df.columns:
            # Find ATM options for each timestamp
            df['atm_distance'] = np.abs(df['moneyness'] - 1.0)
            
            # Get ATM IV for each timestamp (closest to moneyness = 1.0)
            atm_iv = df.groupby('timestamp').apply(
                lambda x: x.loc[x['atm_distance'].idxmin(), 'implied_volatility'] 
                if len(x) > 0 else np.nan
            ).to_dict()
            df['atm_iv'] = df['timestamp'].map(atm_iv)
            
            # IV term structure
            df['iv_vs_atm'] = df['implied_volatility'] - df['atm_iv']
        
        return df
    
    def build_ensemble_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Build ensemble model for volatility prediction
        """
        print("🏗️ Building ensemble model...")
        
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.svm import SVR
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        
        # Base models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'ridge': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ]),
            'elastic_net': Pipeline([
                ('scaler', StandardScaler()),
                ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
            ]),
            'svr': Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, gamma='scale'))
            ]),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42
                ))
            ])
        }
        
        # Train and evaluate base models
        model_scores = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='neg_mean_squared_error', n_jobs=-1)
            model_scores[name] = -cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Validation score if available
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_mse = np.mean((y_val - val_pred) ** 2)
                print(f"{name} - CV MSE: {model_scores[name]:.6f}, Val MSE: {val_mse:.6f}")
            else:
                print(f"{name} - CV MSE: {model_scores[name]:.6f}")
        
        # Meta-learner (stacking)
        print("Training meta-learner...")
        
        # Create meta-features using cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        meta_features = np.zeros((X_train.shape[0], len(models)))
        
        for i, (name, model) in enumerate(models.items()):
            fold_predictions = np.zeros(X_train.shape[0])
            
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                model.fit(X_fold_train, y_fold_train)
                fold_predictions[val_idx] = model.predict(X_fold_val)
            
            meta_features[:, i] = fold_predictions
        
        # Train meta-learner
        meta_learner = Ridge(alpha=0.1)
        meta_learner.fit(meta_features, y_train)
        
        # Store ensemble components
        self.ensemble_predictor = {
            'base_models': trained_models,
            'meta_learner': meta_learner,
            'model_scores': model_scores
        }
        
        return self.ensemble_predictor
    
    def predict_ensemble(self, X_test):
        """
        Make predictions using ensemble model
        """
        if self.ensemble_predictor is None:
            raise ValueError("Ensemble model not trained. Call build_ensemble_model first.")
        
        base_models = self.ensemble_predictor['base_models']
        meta_learner = self.ensemble_predictor['meta_learner']
        
        # Get base model predictions
        base_predictions = np.zeros((X_test.shape[0], len(base_models)))
        
        for i, (name, model) in enumerate(base_models.items()):
            base_predictions[:, i] = model.predict(X_test)
        
        # Meta-learner prediction
        final_predictions = meta_learner.predict(base_predictions)
        
        return final_predictions
    
    def train_deep_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train deep learning model for volatility prediction
        """
        print("🧠 Training deep learning model...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            if X_val is not None:
                X_val_scaled = scaler.transform(X_val)
            
            # Build model
            model = Sequential([
                Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(64, activation='relu'),
                Dropout(0.2),
                
                Dense(32, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
            ]
            
            # Train model
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=200,
                    batch_size=128,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=128,
                    callbacks=callbacks,
                    verbose=1
                )
            
            self.deep_predictor = {
                'model': model,
                'scaler': scaler,
                'history': history
            }
            
            print("✅ Deep learning model training completed!")
            return self.deep_predictor
            
        except ImportError:
            print("⚠️ TensorFlow not available. Skipping deep learning model.")
            return None
    
    def predict_deep(self, X_test):
        """
        Make predictions using deep learning model
        """
        if self.deep_predictor is None:
            raise ValueError("Deep model not trained or not available.")
        
        model = self.deep_predictor['model']
        scaler = self.deep_predictor['scaler']
        
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled).flatten()
        
        return predictions
    
    def validate_model(self, X_train, y_train, test_size=0.2):
        """
        Validate model performance using train-validation split
        """
        print("🔬 Validating model performance...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=test_size, random_state=42
        )
        
        # Train models
        self.build_ensemble_model(X_tr, y_tr, X_val, y_val)
        self.train_deep_model(X_tr, y_tr, X_val, y_val)
        
        # Make predictions
        ensemble_pred = self.predict_ensemble(X_val)
        
        # Calculate metrics
        metrics = {
            'ensemble': {
                'mse': mean_squared_error(y_val, ensemble_pred),
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'r2': r2_score(y_val, ensemble_pred)
            }
        }
        
        # Deep model predictions (if available)
        if self.deep_predictor is not None:
            deep_pred = self.predict_deep(X_val)
            metrics['deep'] = {
                'mse': mean_squared_error(y_val, deep_pred),
                'mae': mean_absolute_error(y_val, deep_pred),
                'r2': r2_score(y_val, deep_pred)
            }
            
            # Hybrid prediction (ensemble + deep)
            hybrid_pred = 0.7 * ensemble_pred + 0.3 * deep_pred
            metrics['hybrid'] = {
                'mse': mean_squared_error(y_val, hybrid_pred),
                'mae': mean_absolute_error(y_val, hybrid_pred),
                'r2': r2_score(y_val, hybrid_pred)
            }
        
        self.validation_results = metrics
        
        # Print results
        print("\n📊 Validation Results:")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  MSE: {model_metrics['mse']:.6f}")
            print(f"  MAE: {model_metrics['mae']:.6f}")
            print(f"  R²:  {model_metrics['r2']:.6f}")
        
        return metrics
    
    def create_submission(self, test_data, submission_file='submission.csv'):
        """
        Create final submission file
        """
        print("📤 Creating submission file...")
        
        # Prepare test features
        X_test = self.prepare_competition_features(test_data)
        
        # Select features (remove non-predictive columns)
        feature_cols = [col for col in X_test.columns if col not in [
            'timestamp', 'expiry', 'option_id', 'implied_volatility'
        ] and X_test[col].dtype in ['int64', 'float64']]
        
        X_test_features = X_test[feature_cols].fillna(0)
        
        # Make predictions
        if self.ensemble_predictor is not None and self.deep_predictor is not None:
            # Hybrid prediction
            ensemble_pred = self.predict_ensemble(X_test_features)
            deep_pred = self.predict_deep(X_test_features)
            final_predictions = 0.7 * ensemble_pred + 0.3 * deep_pred
            print("Using hybrid prediction (ensemble + deep)")
        elif self.ensemble_predictor is not None:
            # Ensemble only
            final_predictions = self.predict_ensemble(X_test_features)
            print("Using ensemble prediction")
        else:
            raise ValueError("No trained model available for predictions")
        
        # Create submission DataFrame
        if 'option_id' in test_data.columns:
            submission_df = pd.DataFrame({
                'option_id': test_data['option_id'],
                'implied_volatility': final_predictions
            })
        else:
            submission_df = pd.DataFrame({
                'implied_volatility': final_predictions
            })
        
        # Ensure predictions are within reasonable bounds
        submission_df['implied_volatility'] = np.clip(
            submission_df['implied_volatility'], 0.01, 2.0
        )
        
        # Save submission
        submission_df.to_csv(submission_file, index=False)
        print(f"✅ Submission saved to {submission_file}")
        
        return submission_df
    
    def save_model(self, filepath='competition_model.joblib'):
        """
        Save trained models to disk
        """
        model_data = {
            'ensemble_predictor': self.ensemble_predictor,
            'deep_predictor': self.deep_predictor,
            'validation_results': self.validation_results
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath='competition_model.joblib'):
        """
        Load trained models from disk
        """
        model_data = joblib.load(filepath)
        self.ensemble_predictor = model_data.get('ensemble_predictor')
        self.deep_predictor = model_data.get('deep_predictor')
        self.validation_results = model_data.get('validation_results', {})
        print(f"✅ Model loaded from {filepath}")
    
    def plot_validation_results(self):
        """
        Plot validation results and model performance
        """
        if not self.validation_results:
            print("No validation results available. Run validate_model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # MSE comparison
        models = list(self.validation_results.keys())
        mse_scores = [self.validation_results[model]['mse'] for model in models]
        
        axes[0, 0].bar(models, mse_scores, color=['blue', 'green', 'red'][:len(models)])
        axes[0, 0].set_title('Model MSE Comparison')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        mae_scores = [self.validation_results[model]['mae'] for model in models]
        axes[0, 1].bar(models, mae_scores, color=['blue', 'green', 'red'][:len(models)])
        axes[0, 1].set_title('Model MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_scores = [self.validation_results[model]['r2'] for model in models]
        axes[1, 0].bar(models, r2_scores, color=['blue', 'green', 'red'][:len(models)])
        axes[1, 0].set_title('Model R² Comparison')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Deep learning training history (if available)
        if self.deep_predictor and 'history' in self.deep_predictor:
            history = self.deep_predictor['history']
            axes[1, 1].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[1, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_title('Deep Learning Training History')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def run_complete_pipeline(self, train_file, test_file, submission_file='submission.csv'):
        """
        Run the complete competition pipeline
        """
        print("🚀 Running complete competition pipeline...")
        
        # 1. Load data
        train_df, test_df = self.load_competition_data(train_file, test_file)
        
        # 2. EDA
        self.exploratory_data_analysis(train_df)
        
        # 3. Prepare features
        train_features = self.prepare_competition_features(train_df)
        
        # 4. Prepare training data
        feature_cols = [col for col in train_features.columns if col not in [
            'timestamp', 'expiry', 'option_id', 'implied_volatility'
        ] and train_features[col].dtype in ['int64', 'float64']]
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features['implied_volatility']
        
        print(f"Training with {len(feature_cols)} features and {len(X_train)} samples")
        print(f"Feature columns: {feature_cols}")
        
        # 5. Validate models
        self.validate_model(X_train, y_train)
        
        # 6. Plot validation results
        self.plot_validation_results()
        
        # 7. Train final models on full data
        print("\n🏁 Training final models on full dataset...")
        self.build_ensemble_model(X_train, y_train)
        self.train_deep_model(X_train, y_train)
        
        # 8. Save models
        self.save_model()
        
        # 9. Create submission
        submission_df = self.create_submission(test_df, submission_file)
        
        print("🎉 Competition pipeline completed successfully!")
        return submission_df


class AdvancedVolatilityPredictor:
    """
    Advanced volatility prediction with additional methods
    """
    
    def __init__(self):
        self.volatility_surface_model = None
        self.term_structure_model = None
        self.smile_model = None
    
    def build_volatility_surface(self, df):
        """
        Build volatility surface model
        """
        print("🌊 Building volatility surface model...")
        
        from scipy.interpolate import RBFInterpolator
        from sklearn.preprocessing import StandardScaler
        
        # Prepare surface data
        if all(col in df.columns for col in ['moneyness', 'time_to_expiry', 'implied_volatility']):
            # Remove outliers
            q99 = df['implied_volatility'].quantile(0.99)
            q01 = df['implied_volatility'].quantile(0.01)
            surface_data = df[(df['implied_volatility'] >= q01) & (df['implied_volatility'] <= q99)].copy()
            
            # Create surface
            X_surface = surface_data[['moneyness', 'time_to_expiry']].values
            y_surface = surface_data['implied_volatility'].values
            
            # Fit RBF interpolator
            self.volatility_surface_model = RBFInterpolator(
                X_surface, y_surface, 
                kernel='thin_plate_spline',
                smoothing=0.1
            )
            
            print("✅ Volatility surface model built")
            return self.volatility_surface_model
        else:
            print("⚠️ Required columns not available for surface modeling")
            return None
    
    def predict_surface_iv(self, moneyness, time_to_expiry):
        """
        Predict implied volatility using surface model
        """
        if self.volatility_surface_model is None:
            raise ValueError("Surface model not built")
        
        X_pred = np.column_stack([moneyness, time_to_expiry])
        return self.volatility_surface_model(X_pred)
    
    def analyze_volatility_smile(self, df, timestamp=None):
        """
        Analyze volatility smile for a specific timestamp
        """
        print("😊 Analyzing volatility smile...")
        
        if timestamp is not None:
            smile_data = df[df['timestamp'] == timestamp].copy()
        else:
            # Use most recent timestamp
            latest_time = df['timestamp'].max()
            smile_data = df[df['timestamp'] == latest_time].copy()
        
        if len(smile_data) == 0:
            print("No data available for smile analysis")
            return None
        
        # Plot smile
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Smile by moneyness
        if 'moneyness' in smile_data.columns:
            call_data = smile_data[smile_data['option_type'] == 'Call']
            put_data = smile_data[smile_data['option_type'] == 'Put']
            
            if len(call_data) > 0:
                axes[0].scatter(call_data['moneyness'], call_data['implied_volatility'], 
                              label='Calls', alpha=0.7, c='blue')
            if len(put_data) > 0:
                axes[0].scatter(put_data['moneyness'], put_data['implied_volatility'], 
                              label='Puts', alpha=0.7, c='red')
            
            axes[0].set_xlabel('Moneyness (K/S)')
            axes[0].set_ylabel('Implied Volatility')
            axes[0].set_title('Volatility Smile')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Term structure
        if 'time_to_expiry' in smile_data.columns:
            atm_data = smile_data[abs(smile_data['moneyness'] - 1.0) < 0.05]  # Near ATM
            
            if len(atm_data) > 0:
                axes[1].scatter(atm_data['time_to_expiry'], atm_data['implied_volatility'], 
                              alpha=0.7, c='green')
                axes[1].set_xlabel('Time to Expiry (Years)')
                axes[1].set_ylabel('Implied Volatility')
                axes[1].set_title('Volatility Term Structure (ATM)')
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def calculate_greeks(self, df):
        """
        Calculate option Greeks using Black-Scholes approximation
        """
        print("🔢 Calculating option Greeks...")
        
        from scipy.stats import norm
        
        df_greeks = df.copy()
        
        required_cols = ['underlying_price', 'strike', 'time_to_expiry', 'implied_volatility']
        if not all(col in df_greeks.columns for col in required_cols):
            print("Missing required columns for Greeks calculation")
            return df_greeks
        
        # Risk-free rate approximation (you should use actual rate)
        r = 0.05
        
        S = df_greeks['underlying_price']
        K = df_greeks['strike']
        T = df_greeks['time_to_expiry']
        sigma = df_greeks['implied_volatility']
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        df_greeks['delta'] = np.where(
            df_greeks['option_type'] == 'Call',
            norm.cdf(d1),
            norm.cdf(d1) - 1
        )
        
        # Gamma
        df_greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (per day)
        theta_call = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        theta_put = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        df_greeks['theta'] = np.where(
            df_greeks['option_type'] == 'Call',
            theta_call / 365,
            theta_put / 365
        )
        
        # Vega
        df_greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        df_greeks['rho'] = np.where(
            df_greeks['option_type'] == 'Call',
            rho_call,
            rho_put
        )
        
        print("✅ Greeks calculated")
        return df_greeks


def create_sample_data(n_samples=10000):
    """
    Create sample NIFTY50 options data for testing
    """
    print(f"📊 Generating sample data with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Base parameters
    underlying_prices = np.random.normal(18000, 500, n_samples)
    
    # Generate timestamps (last 30 days of trading)
    base_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    timestamps = pd.date_range(base_date, periods=n_samples//100, freq='15min')
    timestamps = np.random.choice(timestamps, n_samples)
    
    # Generate strikes around underlying
    moneyness_range = np.random.uniform(0.8, 1.2, n_samples)
    strikes = underlying_prices * moneyness_range
    
    # Generate expiry dates
    days_to_expiry = np.random.exponential(30, n_samples)
    days_to_expiry = np.clip(days_to_expiry, 1, 365)
    
    # Option types
    option_types = np.random.choice(['Call', 'Put'], n_samples)
    
    # Generate implied volatility with realistic patterns
    base_iv = 0.2
    moneyness = strikes / underlying_prices
    time_to_expiry = days_to_expiry / 365.25
    
    # Volatility smile effect
    smile_effect = 0.05 * (moneyness - 1.0) ** 2
    
    # Term structure effect
    term_effect = 0.02 * np.sqrt(time_to_expiry)
    
    # Random noise
    noise = np.random.normal(0, 0.02, n_samples)
    
    implied_volatility = base_iv + smile_effect + term_effect + noise
    implied_volatility = np.clip(implied_volatility, 0.05, 0.8)
    
    # Volume and Open Interest
    volume = np.random.exponential(100, n_samples).astype(int)
    open_interest = np.random.exponential(500, n_samples).astype(int)
    
    # Bid-Ask spreads
    bid_ask_spread = np.random.exponential(2, n_samples)
    mid_price = np.random.uniform(10, 500, n_samples)
    bid = mid_price - bid_ask_spread/2
    ask = mid_price + bid_ask_spread/2
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'option_id': range(n_samples),
        'timestamp': timestamps,
        'underlying_price': underlying_prices,
        'strike': strikes,
        'days_to_expiry': days_to_expiry,
        'option_type': option_types,
        'implied_volatility': implied_volatility,
        'volume': volume,
        'open_interest': open_interest,
        'bid': bid,
        'ask': ask,
        'moneyness': moneyness
    })
    
    # Add expiry dates
    sample_data['expiry'] = sample_data['timestamp'] + pd.to_timedelta(sample_data['days_to_expiry'], unit='D')
    
    print("✅ Sample data generated successfully!")
    return sample_data


# Example usage and testing
if __name__ == "__main__":
    print("🚀 NIFTY50 Volatility Prediction Competition")
    print("=" * 50)
    
    # Create sample data for demonstration
    print("\n1. Creating sample data...")
    sample_data = create_sample_data(5000)
    
    # Split into train and test
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size].copy()
    test_data = sample_data[train_size:].copy()
    
    # Remove target from test data (as in real competition)
    test_data_submission = test_data.drop('implied_volatility', axis=1)
    
    # Save sample files
    train_data.to_csv('sample_train.csv', index=False)
    test_data_submission.to_csv('sample_test.csv', index=False)
    
    print(f"Sample train data: {train_data.shape}")
    print(f"Sample test data: {test_data_submission.shape}")
    
    # Initialize competition submission
    print("\n2. Initializing competition framework...")
    competition = CompetitionSubmission()
    
    # Initialize advanced predictor
    advanced_predictor = AdvancedVolatilityPredictor()
    
    try:
        print("\n3. Running basic pipeline...")
        
        # Load data
        train_df, test_df = competition.load_competition_data('sample_train.csv', 'sample_test.csv')
        
        # Quick EDA
        print("\n4. Basic data exploration...")
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Training columns: {list(train_df.columns)}")
        
        # Prepare features
        print("\n5. Feature engineering...")
        train_features = competition.prepare_competition_features(train_df)
        
        # Select features for modeling
        feature_cols = [col for col in train_features.columns if col not in [
            'timestamp', 'expiry', 'option_id', 'implied_volatility'
        ] and train_features[col].dtype in ['int64', 'float64']]
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features['implied_volatility']
        
        print(f"Features selected: {len(feature_cols)}")
        print(f"Training samples: {len(X_train)}")
        
        # Quick model training (smaller scale for demo)
        print("\n6. Training models...")
        
        # Train ensemble model
        ensemble_model = competition.build_ensemble_model(X_train, y_train)
        
        # Validate performance
        print("\n7. Model validation...")
        validation_results = competition.validate_model(X_train, y_train, test_size=0.2)
        
        # Create submission
        print("\n8. Creating submission...")
        submission = competition.create_submission(test_df, 'sample_submission.csv')
        
        print("\n📈 Submission Statistics:")
        print(submission['implied_volatility'].describe())
        
        # Advanced analysis
        print("\n9. Advanced volatility analysis...")
        
        # Build volatility surface
        surface_model = advanced_predictor.build_volatility_surface(train_df)
        
        # Analyze volatility smile
        smile_analysis = advanced_predictor.analyze_volatility_smile(train_df)
        
        # Calculate Greeks
        train_with_greeks = advanced_predictor.calculate_greeks(train_df)
        
        print("\n✅ Demo completed successfully!")
        print(f"Submission file created: sample_submission.csv")
        print(f"Model validation results available in competition.validation_results")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nThis is a demonstration. In real usage:")
        print("1. Replace sample data with actual competition files")
        print("2. Ensure all required libraries are installed")
        print("3. Adjust hyperparameters based on your data")
        
    print("\n" + "=" * 50)
    print("Competition framework ready for use!")
    print("📚 Key features:")
    print("  • Comprehensive EDA and visualization")
    print("  • Advanced feature engineering for options")
    print("  • Ensemble modeling with multiple algorithms")
    print("  • Deep learning with TensorFlow/Keras")
    print("  • Volatility surface modeling")
    print("  • Options Greeks calculation")
    print("  • Model validation and performance metrics")
    print("  • Automated submission generation")
        