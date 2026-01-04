"""
Training Diagnostics and Metadata System for AI Risk Oracle
Provides advanced feature importance analysis, learning curves, and model diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Optional imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings

# ML libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.ml_trainer import XGBoostRiskModel

@dataclass
class FeatureImportanceAnalysis:
    """Comprehensive feature importance analysis results"""
    model_version: str
    analysis_timestamp: datetime
    
    # Basic feature importance (from model)
    feature_importance_scores: Dict[str, float]
    feature_importance_ranking: List[Tuple[str, float]]
    
    # SHAP values analysis
    shap_values_available: bool = False
    shap_feature_importance: Optional[Dict[str, float]] = None
    shap_summary_plot_path: Optional[str] = None
    
    # Feature correlation analysis
    feature_correlations: Optional[Dict[str, Dict[str, float]]] = None
    highly_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Feature categories analysis
    category_importance: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)

@dataclass
class LearningCurveAnalysis:
    """Learning curve analysis results"""
    model_version: str
    analysis_timestamp: datetime
    
    # Learning curve data
    train_sizes: List[int]
    train_scores_mean: List[float]
    train_scores_std: List[float]
    validation_scores_mean: List[float]
    validation_scores_std: List[float]
    
    # Convergence analysis
    is_converged: bool
    convergence_point: Optional[int] = None
    final_gap: float = 0.0
    
    # Learning curve plot path
    plot_path: Optional[str] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ModelDiagnostics:
    """Comprehensive model diagnostic results"""
    model_version: str
    analysis_timestamp: datetime
    
    # Overfitting/Underfitting detection
    overfitting_detected: bool
    underfitting_detected: bool
    overfitting_severity: str  # "none", "mild", "moderate", "severe"
    underfitting_severity: str
    
    # Validation curve analysis
    validation_curve_analysis: Optional[Dict[str, Any]] = None
    
    # Performance stability
    performance_stability: Dict[str, float] = field(default_factory=dict)
    
    # Diagnostic recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Supporting plots
    diagnostic_plots: List[str] = field(default_factory=list)

class AdvancedFeatureImportanceAnalyzer:
    """Advanced feature importance analysis with SHAP values and correlations"""
    
    def __init__(self, output_dir: str = "models/diagnostics"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib for non-interactive backend
        plt.switch_backend('Agg')
        
    def analyze_feature_importance(self, 
                                 model: XGBoostRiskModel,
                                 X_train: np.ndarray,
                                 feature_names: List[str],
                                 X_sample: Optional[np.ndarray] = None) -> FeatureImportanceAnalysis:
        """
        Comprehensive feature importance analysis
        
        Args:
            model: Trained XGBoost model
            X_train: Training feature matrix
            feature_names: List of feature names
            X_sample: Sample data for SHAP analysis (optional)
        
        Returns:
            FeatureImportanceAnalysis object with comprehensive results
        """
        start_time = time.time()
        
        try:
            analysis_timestamp = datetime.now()
            
            # Basic feature importance from model
            basic_importance = self._get_basic_feature_importance(model, feature_names)
            
            # SHAP analysis if available
            shap_analysis = self._perform_shap_analysis(
                model, X_train, feature_names, X_sample, analysis_timestamp
            )
            
            # Feature correlation analysis
            correlation_analysis = self._analyze_feature_correlations(
                X_train, feature_names
            )
            
            # Category-based importance analysis
            category_importance = self._analyze_category_importance(basic_importance)
            
            # Generate recommendations
            recommendations = self._generate_feature_recommendations(
                basic_importance, shap_analysis, correlation_analysis
            )
            
            # Create comprehensive analysis result
            analysis = FeatureImportanceAnalysis(
                model_version=model.model_version,
                analysis_timestamp=analysis_timestamp,
                feature_importance_scores=basic_importance,
                feature_importance_ranking=sorted(basic_importance.items(), 
                                                key=lambda x: x[1], reverse=True),
                shap_values_available=shap_analysis["available"],
                shap_feature_importance=shap_analysis.get("importance"),
                shap_summary_plot_path=shap_analysis.get("plot_path"),
                feature_correlations=correlation_analysis["correlations"],
                highly_correlated_pairs=correlation_analysis["high_correlations"],
                category_importance=category_importance,
                recommendations=recommendations
            )
            
            # Save analysis results
            self._save_feature_analysis(analysis)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "feature_importance_analysis", duration_ms, True
            )
            
            return analysis
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "feature_importance_analysis", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "feature_analysis", "comprehensive", 0, False, str(e)
            )
            raise
    
    def _get_basic_feature_importance(self, model: XGBoostRiskModel, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """Extract basic feature importance from trained model"""
        try:
            if not model.feature_importance:
                # Fallback to model's built-in feature importance
                if hasattr(model.model, 'feature_importances_'):
                    importance_values = model.model.feature_importances_
                    return dict(zip(feature_names, importance_values))
                else:
                    return {}
            
            return model.feature_importance.copy()
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "basic_feature_importance", "extraction", 0, False, str(e)
            )
            return {}
    
    def _perform_shap_analysis(self, model: XGBoostRiskModel, X_train: np.ndarray,
                             feature_names: List[str], X_sample: Optional[np.ndarray],
                             timestamp: datetime) -> Dict[str, Any]:
        """Perform SHAP analysis for feature importance"""
        try:
            if not SHAP_AVAILABLE:
                return {"available": False, "reason": "SHAP not installed"}
            
            if X_train.size == 0:
                return {"available": False, "reason": "No training data"}
            
            # Use sample data or subset of training data
            if X_sample is not None:
                analysis_data = X_sample
            else:
                # Use a sample of training data for efficiency
                sample_size = min(100, X_train.shape[0])
                indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                analysis_data = X_train[indices]
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(analysis_data)
            
            # Calculate mean absolute SHAP values for feature importance
            if isinstance(shap_values, list):
                # For multi-class (though we have binary)
                shap_importance = np.mean(np.abs(shap_values[1]), axis=0)
            else:
                # For binary classification
                shap_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance dictionary
            shap_feature_importance = dict(zip(feature_names, shap_importance))
            
            # Create and save SHAP summary plot
            plot_path = self._create_shap_summary_plot(
                shap_values, analysis_data, feature_names, timestamp
            )
            
            return {
                "available": True,
                "importance": shap_feature_importance,
                "plot_path": plot_path,
                "sample_size": analysis_data.shape[0]
            }
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "shap_analysis", "computation", 0, False, str(e)
            )
            return {"available": False, "reason": f"SHAP analysis failed: {str(e)}"}
    
    def _create_shap_summary_plot(self, shap_values, X_data: np.ndarray,
                                feature_names: List[str], timestamp: datetime) -> str:
        """Create and save SHAP summary plot"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Create SHAP summary plot
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_data, feature_names=feature_names, 
                                show=False, max_display=20)
            else:
                shap.summary_plot(shap_values, X_data, feature_names=feature_names,
                                show=False, max_display=20)
            
            # Save plot
            plot_filename = f"shap_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "shap_plot", "creation", 0, False, str(e)
            )
            plt.close()  # Ensure plot is closed on error
            return ""
    
    def _analyze_feature_correlations(self, X_train: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Analyze correlations between features"""
        try:
            if X_train.size == 0:
                return {"correlations": {}, "high_correlations": []}
            
            # Create DataFrame for correlation analysis
            df = pd.DataFrame(X_train, columns=feature_names)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            # Convert to dictionary format
            correlations = {}
            for feature in feature_names:
                correlations[feature] = correlation_matrix[feature].to_dict()
            
            # Find highly correlated pairs (>0.8 correlation)
            high_correlations = []
            for i, feature1 in enumerate(feature_names):
                for j, feature2 in enumerate(feature_names[i+1:], i+1):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_correlations.append((feature1, feature2, corr_value))
            
            return {
                "correlations": correlations,
                "high_correlations": high_correlations,
                "correlation_matrix": correlation_matrix
            }
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "correlation_analysis", "computation", 0, False, str(e)
            )
            return {"correlations": {}, "high_correlations": []}
    
    def _analyze_category_importance(self, feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Analyze importance by feature categories"""
        try:
            # Define feature categories
            categories = {
                "price_features": ["current_price", "price_lag", "price_change"],
                "volume_features": ["current_volume", "volume_lag", "volume_anomaly"],
                "technical_indicators": ["rsi", "macd", "bb_position"],
                "correlation_features": ["btc_correlation", "gold_correlation", "market_correlation"],
                "volatility_features": ["current_volatility", "volatility_percentile"],
                "momentum_features": ["price_momentum", "momentum_score"]
            }
            
            category_importance = {}
            
            for category, keywords in categories.items():
                total_importance = 0.0
                feature_count = 0
                
                for feature, importance in feature_importance.items():
                    if any(keyword in feature.lower() for keyword in keywords):
                        total_importance += importance
                        feature_count += 1
                
                # Average importance for this category
                if feature_count > 0:
                    category_importance[category] = total_importance / feature_count
                else:
                    category_importance[category] = 0.0
            
            return category_importance
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "category_analysis", "computation", 0, False, str(e)
            )
            return {}
    
    def _generate_feature_recommendations(self, basic_importance: Dict[str, float],
                                        shap_analysis: Dict[str, Any],
                                        correlation_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feature analysis"""
        recommendations = []
        
        try:
            # Check for low-importance features
            if basic_importance:
                sorted_features = sorted(basic_importance.items(), key=lambda x: x[1])
                low_importance_features = [f for f, imp in sorted_features[:5] if imp < 0.01]
                
                if low_importance_features:
                    recommendations.append(
                        f"Consider removing low-importance features: {', '.join(low_importance_features)}"
                    )
            
            # Check for highly correlated features
            high_correlations = correlation_analysis.get("high_correlations", [])
            if high_correlations:
                recommendations.append(
                    f"Found {len(high_correlations)} highly correlated feature pairs. "
                    "Consider feature selection to reduce multicollinearity."
                )
            
            # SHAP-specific recommendations
            if shap_analysis.get("available"):
                recommendations.append(
                    "SHAP analysis available - use for detailed feature interaction analysis"
                )
            else:
                recommendations.append(
                    f"SHAP analysis unavailable: {shap_analysis.get('reason', 'Unknown reason')}"
                )
            
            # General recommendations
            if len(basic_importance) > 50:
                recommendations.append(
                    "Large number of features detected. Consider dimensionality reduction."
                )
            
            return recommendations
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "feature_recommendations", "generation", 0, False, str(e)
            )
            return ["Error generating recommendations"]
    
    def _save_feature_analysis(self, analysis: FeatureImportanceAnalysis):
        """Save feature importance analysis results"""
        try:
            # Convert analysis to dictionary for JSON serialization
            analysis_dict = {
                "model_version": analysis.model_version,
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                "feature_importance_scores": analysis.feature_importance_scores,
                "feature_importance_ranking": analysis.feature_importance_ranking,
                "shap_values_available": analysis.shap_values_available,
                "shap_feature_importance": analysis.shap_feature_importance,
                "shap_summary_plot_path": analysis.shap_summary_plot_path,
                "feature_correlations": analysis.feature_correlations,
                "highly_correlated_pairs": analysis.highly_correlated_pairs,
                "category_importance": analysis.category_importance,
                "recommendations": analysis.recommendations
            }
            
            # Save to JSON file
            filename = f"feature_analysis_{analysis.model_version}_{analysis.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(analysis_dict, f, indent=2, default=str)
            
            self.data_logger.log_data_collection(
                "feature_analysis", "save", 1, True, f"Saved to {filepath}"
            )
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "feature_analysis", "save", 0, False, str(e)
            )

class LearningCurveAnalyzer:
    """Learning curve generation and convergence analysis"""
    
    def __init__(self, output_dir: str = "models/diagnostics"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib
        plt.switch_backend('Agg')
    
    def generate_learning_curves(self, model_class, X: np.ndarray, y: np.ndarray,
                                cv_folds: int = 5, train_sizes: Optional[List[float]] = None,
                                model_params: Optional[Dict] = None) -> LearningCurveAnalysis:
        """
        Generate learning curves and analyze convergence
        
        Args:
            model_class: Model class to instantiate
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            train_sizes: Training sizes to evaluate (as fractions)
            model_params: Model parameters
        
        Returns:
            LearningCurveAnalysis object with results
        """
        start_time = time.time()
        
        try:
            analysis_timestamp = datetime.now()
            
            if train_sizes is None:
                train_sizes = np.linspace(0.1, 1.0, 10)
            
            # Create model instance
            if model_params:
                model = model_class(model_params)
            else:
                model = model_class()
            
            # Generate learning curves
            train_sizes_abs, train_scores, validation_scores = learning_curve(
                model.model if hasattr(model, 'model') else model,
                X, y,
                train_sizes=train_sizes,
                cv=cv_folds,
                scoring='f1',
                n_jobs=1,  # Changed from -1 to avoid issues
                random_state=42
            )
            
            # Calculate means and standard deviations
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            validation_scores_mean = np.mean(validation_scores, axis=1)
            validation_scores_std = np.std(validation_scores, axis=1)
            
            # Analyze convergence
            convergence_analysis = self._analyze_convergence(
                train_scores_mean, validation_scores_mean, train_sizes_abs
            )
            
            # Create learning curve plot
            plot_path = self._create_learning_curve_plot(
                train_sizes_abs, train_scores_mean, train_scores_std,
                validation_scores_mean, validation_scores_std, analysis_timestamp
            )
            
            # Generate recommendations
            recommendations = self._generate_learning_curve_recommendations(
                convergence_analysis, train_scores_mean, validation_scores_mean
            )
            
            # Create analysis result
            analysis = LearningCurveAnalysis(
                model_version=getattr(model, 'model_version', 'unknown'),
                analysis_timestamp=analysis_timestamp,
                train_sizes=train_sizes_abs.tolist(),
                train_scores_mean=train_scores_mean.tolist(),
                train_scores_std=train_scores_std.tolist(),
                validation_scores_mean=validation_scores_mean.tolist(),
                validation_scores_std=validation_scores_std.tolist(),
                is_converged=convergence_analysis["is_converged"],
                convergence_point=convergence_analysis["convergence_point"],
                final_gap=convergence_analysis["final_gap"],
                plot_path=plot_path,
                recommendations=recommendations
            )
            
            # Save analysis results
            self._save_learning_curve_analysis(analysis)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "learning_curve_analysis", duration_ms, True
            )
            
            return analysis
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "learning_curve_analysis", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "learning_curve", "generation", 0, False, str(e)
            )
            raise
    
    def _analyze_convergence(self, train_scores: np.ndarray, val_scores: np.ndarray,
                           train_sizes: np.ndarray) -> Dict[str, Any]:
        """Analyze learning curve convergence"""
        try:
            # Check if curves have converged (stable performance)
            convergence_threshold = 0.01  # 1% change threshold
            window_size = 3  # Look at last 3 points
            
            is_converged = False
            convergence_point = None
            
            if len(val_scores) >= window_size:
                # Check if validation score is stable in the last window
                recent_val_scores = val_scores[-window_size:]
                val_stability = np.std(recent_val_scores) < convergence_threshold
                
                # Check if training and validation scores are close
                final_gap = abs(train_scores[-1] - val_scores[-1])
                
                if val_stability and final_gap < 0.1:  # 10% gap threshold
                    is_converged = True
                    # Find approximate convergence point
                    for i in range(len(val_scores) - window_size + 1):
                        window_scores = val_scores[i:i + window_size]
                        if np.std(window_scores) < convergence_threshold:
                            convergence_point = int(train_sizes[i])
                            break
            
            final_gap = abs(train_scores[-1] - val_scores[-1]) if len(train_scores) > 0 else 0.0
            
            return {
                "is_converged": is_converged,
                "convergence_point": convergence_point,
                "final_gap": float(final_gap)
            }
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "convergence_analysis", "computation", 0, False, str(e)
            )
            return {"is_converged": False, "convergence_point": None, "final_gap": 0.0}
    
    def _create_learning_curve_plot(self, train_sizes: np.ndarray,
                                  train_scores_mean: np.ndarray, train_scores_std: np.ndarray,
                                  val_scores_mean: np.ndarray, val_scores_std: np.ndarray,
                                  timestamp: datetime) -> str:
        """Create and save learning curve plot"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot training scores
            plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1, color='blue')
            
            # Plot validation scores
            plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                           val_scores_mean + val_scores_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('F1 Score')
            plt.title('Learning Curves')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_filename = f"learning_curves_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "learning_curve_plot", "creation", 0, False, str(e)
            )
            plt.close()
            return ""
    
    def _generate_learning_curve_recommendations(self, convergence_analysis: Dict[str, Any],
                                               train_scores: np.ndarray,
                                               val_scores: np.ndarray) -> List[str]:
        """Generate recommendations based on learning curve analysis"""
        recommendations = []
        
        try:
            # Convergence recommendations
            if convergence_analysis["is_converged"]:
                recommendations.append(
                    f"Model has converged at approximately {convergence_analysis['convergence_point']} samples"
                )
            else:
                recommendations.append(
                    "Model has not converged - consider collecting more training data"
                )
            
            # Performance gap analysis
            final_gap = convergence_analysis["final_gap"]
            if final_gap > 0.15:  # 15% gap
                recommendations.append(
                    f"Large gap between training and validation performance ({final_gap:.3f}) - "
                    "possible overfitting detected"
                )
            elif final_gap < 0.05:  # 5% gap
                recommendations.append(
                    "Good generalization - training and validation scores are close"
                )
            
            # Performance level analysis
            if len(val_scores) > 0:
                final_val_score = val_scores[-1]
                if final_val_score < 0.6:
                    recommendations.append(
                        f"Low validation performance ({final_val_score:.3f}) - "
                        "consider feature engineering or model tuning"
                    )
                elif final_val_score > 0.8:
                    recommendations.append(
                        f"Good validation performance ({final_val_score:.3f})"
                    )
            
            # Training efficiency
            if len(train_scores) > 5:
                early_score = train_scores[len(train_scores)//3]  # Score at 1/3 of data
                final_score = train_scores[-1]
                improvement = final_score - early_score
                
                if improvement < 0.05:
                    recommendations.append(
                        "Limited improvement with additional data - model may be saturated"
                    )
            
            return recommendations
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "learning_curve_recommendations", "generation", 0, False, str(e)
            )
            return ["Error generating learning curve recommendations"]
    
    def _save_learning_curve_analysis(self, analysis: LearningCurveAnalysis):
        """Save learning curve analysis results"""
        try:
            # Convert analysis to dictionary
            analysis_dict = {
                "model_version": analysis.model_version,
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                "train_sizes": analysis.train_sizes,
                "train_scores_mean": analysis.train_scores_mean,
                "train_scores_std": analysis.train_scores_std,
                "validation_scores_mean": analysis.validation_scores_mean,
                "validation_scores_std": analysis.validation_scores_std,
                "is_converged": analysis.is_converged,
                "convergence_point": analysis.convergence_point,
                "final_gap": analysis.final_gap,
                "plot_path": analysis.plot_path,
                "recommendations": analysis.recommendations
            }
            
            # Save to JSON file
            filename = f"learning_curve_analysis_{analysis.model_version}_{analysis.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(analysis_dict, f, indent=2, default=str)
            
            self.data_logger.log_data_collection(
                "learning_curve_analysis", "save", 1, True, f"Saved to {filepath}"
            )
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "learning_curve_analysis", "save", 0, False, str(e)
            )

class ModelDiagnosticSystem:
    """Comprehensive model diagnostic system for overfitting/underfitting detection"""
    
    def __init__(self, output_dir: str = "models/diagnostics"):
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib
        plt.switch_backend('Agg')
    
    def diagnose_model_performance(self, model_class, X: np.ndarray, y: np.ndarray,
                                 model_params: Optional[Dict] = None,
                                 param_name: str = 'max_depth',
                                 param_range: Optional[List] = None) -> ModelDiagnostics:
        """
        Comprehensive model diagnostics including overfitting/underfitting detection
        
        Args:
            model_class: Model class to diagnose
            X: Feature matrix
            y: Target labels
            model_params: Base model parameters
            param_name: Parameter to vary for validation curve
            param_range: Range of parameter values to test
        
        Returns:
            ModelDiagnostics object with comprehensive results
        """
        start_time = time.time()
        
        try:
            analysis_timestamp = datetime.now()
            
            if param_range is None:
                if param_name == 'max_depth':
                    param_range = [3, 4, 5, 6, 7, 8, 10, 12]
                elif param_name == 'n_estimators':
                    param_range = [50, 100, 150, 200, 300, 400, 500]
                else:
                    param_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
            
            # Generate validation curves
            validation_analysis = self._generate_validation_curves(
                model_class, X, y, model_params, param_name, param_range, analysis_timestamp
            )
            
            # Detect overfitting and underfitting
            overfitting_analysis = self._detect_overfitting(validation_analysis)
            underfitting_analysis = self._detect_underfitting(validation_analysis)
            
            # Analyze performance stability
            stability_analysis = self._analyze_performance_stability(validation_analysis)
            
            # Generate diagnostic recommendations
            recommendations = self._generate_diagnostic_recommendations(
                overfitting_analysis, underfitting_analysis, stability_analysis
            )
            
            # Create diagnostic plots
            diagnostic_plots = self._create_diagnostic_plots(
                validation_analysis, analysis_timestamp
            )
            
            # Create comprehensive diagnostics result
            diagnostics = ModelDiagnostics(
                model_version=getattr(model_class(), 'model_version', 'unknown'),
                analysis_timestamp=analysis_timestamp,
                overfitting_detected=overfitting_analysis["detected"],
                underfitting_detected=underfitting_analysis["detected"],
                overfitting_severity=overfitting_analysis["severity"],
                underfitting_severity=underfitting_analysis["severity"],
                validation_curve_analysis=validation_analysis,
                performance_stability=stability_analysis,
                recommendations=recommendations,
                diagnostic_plots=diagnostic_plots
            )
            
            # Save diagnostic results
            self._save_diagnostic_results(diagnostics)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "model_diagnostics", duration_ms, True
            )
            
            return diagnostics
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time(
                "model_diagnostics", duration_ms, False
            )
            self.data_logger.log_data_collection(
                "model_diagnostics", "comprehensive", 0, False, str(e)
            )
            raise
    
    def _generate_validation_curves(self, model_class, X: np.ndarray, y: np.ndarray,
                                  model_params: Optional[Dict], param_name: str,
                                  param_range: List, timestamp: datetime) -> Dict[str, Any]:
        """Generate validation curves for parameter analysis"""
        try:
            # Create base model
            base_params = model_params.copy() if model_params else {}
            model = model_class(base_params)
            
            # Generate validation curve
            train_scores, validation_scores = validation_curve(
                model.model if hasattr(model, 'model') else model,
                X, y,
                param_name=param_name,
                param_range=param_range,
                cv=5,
                scoring='f1',
                n_jobs=1  # Changed from -1 to avoid issues
            )
            
            # Calculate means and standard deviations
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            validation_scores_mean = np.mean(validation_scores, axis=1)
            validation_scores_std = np.std(validation_scores, axis=1)
            
            return {
                "param_name": param_name,
                "param_range": param_range,
                "train_scores_mean": train_scores_mean.tolist(),
                "train_scores_std": train_scores_std.tolist(),
                "validation_scores_mean": validation_scores_mean.tolist(),
                "validation_scores_std": validation_scores_std.tolist(),
                "best_param_idx": int(np.argmax(validation_scores_mean)),
                "best_param_value": param_range[np.argmax(validation_scores_mean)],
                "best_validation_score": float(np.max(validation_scores_mean))
            }
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "validation_curves", "generation", 0, False, str(e)
            )
            return {}
    
    def _detect_overfitting(self, validation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect overfitting from validation curves"""
        try:
            if not validation_analysis:
                return {"detected": False, "severity": "none", "evidence": []}
            
            train_scores = np.array(validation_analysis["train_scores_mean"])
            val_scores = np.array(validation_analysis["validation_scores_mean"])
            
            # Calculate performance gaps
            performance_gaps = train_scores - val_scores
            max_gap = np.max(performance_gaps)
            avg_gap = np.mean(performance_gaps)
            
            # Detect overfitting patterns
            evidence = []
            severity = "none"
            detected = False
            
            # Large performance gap
            if max_gap > 0.2:  # 20% gap
                evidence.append(f"Large performance gap detected: {max_gap:.3f}")
                detected = True
                severity = "severe"
            elif max_gap > 0.15:  # 15% gap
                evidence.append(f"Moderate performance gap detected: {max_gap:.3f}")
                detected = True
                severity = "moderate"
            elif max_gap > 0.1:  # 10% gap
                evidence.append(f"Mild performance gap detected: {max_gap:.3f}")
                detected = True
                severity = "mild"
            
            # Decreasing validation performance with increasing complexity
            if len(val_scores) > 3:
                # Check if validation score decreases in latter half
                mid_point = len(val_scores) // 2
                early_val_avg = np.mean(val_scores[:mid_point])
                late_val_avg = np.mean(val_scores[mid_point:])
                
                if late_val_avg < early_val_avg - 0.05:  # 5% decrease
                    evidence.append("Validation performance decreases with model complexity")
                    detected = True
                    if severity == "none":
                        severity = "mild"
            
            return {
                "detected": detected,
                "severity": severity,
                "evidence": evidence,
                "max_gap": float(max_gap),
                "avg_gap": float(avg_gap)
            }
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "overfitting_detection", "analysis", 0, False, str(e)
            )
            return {"detected": False, "severity": "none", "evidence": []}
    
    def _detect_underfitting(self, validation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect underfitting from validation curves"""
        try:
            if not validation_analysis:
                return {"detected": False, "severity": "none", "evidence": []}
            
            train_scores = np.array(validation_analysis["train_scores_mean"])
            val_scores = np.array(validation_analysis["validation_scores_mean"])
            
            evidence = []
            severity = "none"
            detected = False
            
            # Low absolute performance
            max_train_score = np.max(train_scores)
            max_val_score = np.max(val_scores)
            
            if max_val_score < 0.6:  # 60% performance threshold
                evidence.append(f"Low validation performance: {max_val_score:.3f}")
                detected = True
                if max_val_score < 0.4:
                    severity = "severe"
                elif max_val_score < 0.5:
                    severity = "moderate"
                else:
                    severity = "mild"
            
            # Both training and validation scores are low and similar
            if max_train_score < 0.7 and abs(max_train_score - max_val_score) < 0.05:
                evidence.append("Both training and validation scores are low and similar")
                detected = True
                if severity == "none":
                    severity = "mild"
            
            # Continuously improving performance (not plateaued)
            if len(val_scores) > 3:
                # Check if performance is still improving at the end
                recent_trend = val_scores[-3:] - val_scores[-4:-1]
                if np.mean(recent_trend) > 0.01:  # Still improving by 1%
                    evidence.append("Model performance still improving - may benefit from more complexity")
                    detected = True
                    if severity == "none":
                        severity = "mild"
            
            return {
                "detected": detected,
                "severity": severity,
                "evidence": evidence,
                "max_train_score": float(max_train_score),
                "max_val_score": float(max_val_score)
            }
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "underfitting_detection", "analysis", 0, False, str(e)
            )
            return {"detected": False, "severity": "none", "evidence": []}
    
    def _analyze_performance_stability(self, validation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Analyze performance stability across parameter values"""
        try:
            if not validation_analysis:
                return {}
            
            val_scores = np.array(validation_analysis["validation_scores_mean"])
            val_stds = np.array(validation_analysis["validation_scores_std"])
            
            # Calculate stability metrics
            stability_metrics = {
                "score_variance": float(np.var(val_scores)),
                "score_std": float(np.std(val_scores)),
                "avg_cv_std": float(np.mean(val_stds)),
                "max_cv_std": float(np.max(val_stds)),
                "score_range": float(np.max(val_scores) - np.min(val_scores))
            }
            
            return stability_metrics
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "stability_analysis", "computation", 0, False, str(e)
            )
            return {}
    
    def _generate_diagnostic_recommendations(self, overfitting_analysis: Dict[str, Any],
                                           underfitting_analysis: Dict[str, Any],
                                           stability_analysis: Dict[str, float]) -> List[str]:
        """Generate diagnostic recommendations"""
        recommendations = []
        
        try:
            # Overfitting recommendations
            if overfitting_analysis["detected"]:
                severity = overfitting_analysis["severity"]
                if severity == "severe":
                    recommendations.append(
                        "Severe overfitting detected. Consider: reducing model complexity, "
                        "adding regularization, collecting more training data"
                    )
                elif severity == "moderate":
                    recommendations.append(
                        "Moderate overfitting detected. Consider: cross-validation, "
                        "early stopping, or regularization techniques"
                    )
                else:
                    recommendations.append(
                        "Mild overfitting detected. Monitor performance on unseen data"
                    )
            
            # Underfitting recommendations
            if underfitting_analysis["detected"]:
                severity = underfitting_analysis["severity"]
                if severity == "severe":
                    recommendations.append(
                        "Severe underfitting detected. Consider: increasing model complexity, "
                        "feature engineering, or different model architecture"
                    )
                elif severity == "moderate":
                    recommendations.append(
                        "Moderate underfitting detected. Consider: tuning hyperparameters "
                        "or adding more features"
                    )
                else:
                    recommendations.append(
                        "Mild underfitting detected. Model may benefit from slight complexity increase"
                    )
            
            # Stability recommendations
            if stability_analysis:
                score_variance = stability_analysis.get("score_variance", 0)
                if score_variance > 0.01:  # High variance
                    recommendations.append(
                        "High performance variance detected. Consider more robust "
                        "cross-validation or parameter tuning"
                    )
            
            # General recommendations
            if not overfitting_analysis["detected"] and not underfitting_analysis["detected"]:
                recommendations.append(
                    "Model appears well-balanced. Consider fine-tuning for optimal performance"
                )
            
            return recommendations
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "diagnostic_recommendations", "generation", 0, False, str(e)
            )
            return ["Error generating diagnostic recommendations"]
    
    def _create_diagnostic_plots(self, validation_analysis: Dict[str, Any],
                               timestamp: datetime) -> List[str]:
        """Create diagnostic plots"""
        plot_paths = []
        
        try:
            if not validation_analysis:
                return plot_paths
            
            # Validation curve plot
            plot_path = self._create_validation_curve_plot(validation_analysis, timestamp)
            if plot_path:
                plot_paths.append(plot_path)
            
            return plot_paths
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "diagnostic_plots", "creation", 0, False, str(e)
            )
            return []
    
    def _create_validation_curve_plot(self, validation_analysis: Dict[str, Any],
                                    timestamp: datetime) -> str:
        """Create validation curve plot"""
        try:
            param_range = validation_analysis["param_range"]
            train_scores_mean = np.array(validation_analysis["train_scores_mean"])
            train_scores_std = np.array(validation_analysis["train_scores_std"])
            val_scores_mean = np.array(validation_analysis["validation_scores_mean"])
            val_scores_std = np.array(validation_analysis["validation_scores_std"])
            param_name = validation_analysis["param_name"]
            
            plt.figure(figsize=(10, 6))
            
            # Plot training scores
            plt.plot(param_range, train_scores_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std, alpha=0.1, color='blue')
            
            # Plot validation scores
            plt.plot(param_range, val_scores_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(param_range, val_scores_mean - val_scores_std,
                           val_scores_mean + val_scores_std, alpha=0.1, color='red')
            
            plt.xlabel(param_name.replace('_', ' ').title())
            plt.ylabel('F1 Score')
            plt.title(f'Validation Curve - {param_name}')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Mark best parameter
            best_idx = validation_analysis["best_param_idx"]
            plt.axvline(x=param_range[best_idx], color='green', linestyle='--', alpha=0.7,
                       label=f'Best {param_name}: {param_range[best_idx]}')
            plt.legend(loc='best')
            
            # Save plot
            plot_filename = f"validation_curve_{param_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = self.output_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "validation_curve_plot", "creation", 0, False, str(e)
            )
            plt.close()
            return ""
    
    def _save_diagnostic_results(self, diagnostics: ModelDiagnostics):
        """Save model diagnostic results"""
        try:
            # Convert diagnostics to dictionary
            diagnostics_dict = {
                "model_version": diagnostics.model_version,
                "analysis_timestamp": diagnostics.analysis_timestamp.isoformat(),
                "overfitting_detected": diagnostics.overfitting_detected,
                "underfitting_detected": diagnostics.underfitting_detected,
                "overfitting_severity": diagnostics.overfitting_severity,
                "underfitting_severity": diagnostics.underfitting_severity,
                "validation_curve_analysis": diagnostics.validation_curve_analysis,
                "performance_stability": diagnostics.performance_stability,
                "recommendations": diagnostics.recommendations,
                "diagnostic_plots": diagnostics.diagnostic_plots
            }
            
            # Save to JSON file
            filename = f"model_diagnostics_{diagnostics.model_version}_{diagnostics.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(diagnostics_dict, f, indent=2, default=str)
            
            self.data_logger.log_data_collection(
                "model_diagnostics", "save", 1, True, f"Saved to {filepath}"
            )
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "model_diagnostics", "save", 0, False, str(e)
            )

# Comprehensive Training Diagnostics Manager
class TrainingDiagnosticsManager:
    """Main manager for all training diagnostics and metadata"""
    
    def __init__(self, output_dir: str = "models/diagnostics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers
        self.feature_analyzer = AdvancedFeatureImportanceAnalyzer(str(self.output_dir))
        self.learning_curve_analyzer = LearningCurveAnalyzer(str(self.output_dir))
        self.diagnostic_system = ModelDiagnosticSystem(str(self.output_dir))
        
        self.data_logger = get_data_logger()
    
    def run_comprehensive_diagnostics(self, model: XGBoostRiskModel, 
                                    X_train: np.ndarray, y_train: np.ndarray,
                                    feature_names: List[str],
                                    model_class=None, model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive training diagnostics
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            feature_names: Feature names
            model_class: Model class for learning curves
            model_params: Model parameters
        
        Returns:
            Dictionary with all diagnostic results
        """
        try:
            results = {}
            
            # 1. Feature importance analysis
            print("Running feature importance analysis...")
            feature_analysis = self.feature_analyzer.analyze_feature_importance(
                model, X_train, feature_names
            )
            results["feature_importance"] = feature_analysis
            
            # 2. Learning curve analysis (if model class provided)
            if model_class is not None:
                print("Generating learning curves...")
                learning_analysis = self.learning_curve_analyzer.generate_learning_curves(
                    model_class, X_train, y_train, model_params=model_params
                )
                results["learning_curves"] = learning_analysis
            
            # 3. Model diagnostics
            if model_class is not None:
                print("Running model diagnostics...")
                diagnostic_analysis = self.diagnostic_system.diagnose_model_performance(
                    model_class, X_train, y_train, model_params=model_params
                )
                results["model_diagnostics"] = diagnostic_analysis
            
            # 4. Generate comprehensive metadata
            metadata = self._generate_comprehensive_metadata(results, model)
            results["comprehensive_metadata"] = metadata
            
            # 5. Save comprehensive report
            report_path = self._save_comprehensive_report(results)
            results["report_path"] = report_path
            
            return results
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "comprehensive_diagnostics", "execution", 0, False, str(e)
            )
            raise
    
    def _generate_comprehensive_metadata(self, diagnostic_results: Dict[str, Any],
                                       model: XGBoostRiskModel) -> Dict[str, Any]:
        """Generate comprehensive training metadata"""
        try:
            metadata = {
                "model_version": model.model_version,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_parameters": model.model_params,
                
                # Feature analysis summary
                "feature_analysis_summary": {},
                
                # Learning curve summary
                "learning_curve_summary": {},
                
                # Diagnostic summary
                "diagnostic_summary": {},
                
                # Overall recommendations
                "overall_recommendations": []
            }
            
            # Extract feature analysis summary
            if "feature_importance" in diagnostic_results:
                fa = diagnostic_results["feature_importance"]
                metadata["feature_analysis_summary"] = {
                    "total_features": len(fa.feature_importance_scores),
                    "shap_available": fa.shap_values_available,
                    "high_correlation_pairs": len(fa.highly_correlated_pairs),
                    "top_features": fa.feature_importance_ranking[:5]
                }
            
            # Extract learning curve summary
            if "learning_curves" in diagnostic_results:
                lc = diagnostic_results["learning_curves"]
                metadata["learning_curve_summary"] = {
                    "converged": lc.is_converged,
                    "convergence_point": lc.convergence_point,
                    "final_gap": lc.final_gap,
                    "final_validation_score": lc.validation_scores_mean[-1] if lc.validation_scores_mean else None
                }
            
            # Extract diagnostic summary
            if "model_diagnostics" in diagnostic_results:
                md = diagnostic_results["model_diagnostics"]
                metadata["diagnostic_summary"] = {
                    "overfitting_detected": md.overfitting_detected,
                    "overfitting_severity": md.overfitting_severity,
                    "underfitting_detected": md.underfitting_detected,
                    "underfitting_severity": md.underfitting_severity
                }
            
            # Compile overall recommendations
            all_recommendations = []
            for analysis_type, analysis_result in diagnostic_results.items():
                if hasattr(analysis_result, 'recommendations'):
                    all_recommendations.extend(analysis_result.recommendations)
            
            metadata["overall_recommendations"] = all_recommendations
            
            return metadata
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "comprehensive_metadata", "generation", 0, False, str(e)
            )
            return {}
    
    def _save_comprehensive_report(self, diagnostic_results: Dict[str, Any]) -> str:
        """Save comprehensive diagnostic report"""
        try:
            timestamp = datetime.now()
            report_filename = f"comprehensive_diagnostics_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            report_path = self.output_dir / report_filename
            
            # Convert all results to serializable format
            serializable_results = {}
            for key, value in diagnostic_results.items():
                if hasattr(value, '__dict__'):
                    # Convert dataclass to dict
                    serializable_results[key] = {
                        attr: getattr(value, attr) for attr in dir(value)
                        if not attr.startswith('_') and not callable(getattr(value, attr))
                    }
                else:
                    serializable_results[key] = value
            
            # Save to JSON
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.data_logger.log_data_collection(
                "comprehensive_report", "save", 1, True, f"Saved to {report_path}"
            )
            
            return str(report_path)
            
        except Exception as e:
            self.data_logger.log_data_collection(
                "comprehensive_report", "save", 0, False, str(e)
            )
            return ""