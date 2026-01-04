"""
Threshold Optimization Engine for AI Risk Oracle Model Training
Implements grid search for optimal threshold selection and F1 score optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.ml_trainer import XGBoostRiskModel, FeaturePreprocessor

@dataclass
class ThresholdMetrics:
    """Performance metrics for a specific threshold value"""
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    
    # Confusion matrix components
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Risk-specific metrics
    risk_events_detected: int = 0
    risk_events_missed: int = 0
    false_alarms: int = 0
    
    # Statistical confidence
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    sample_size: int = 0
    
    # Additional metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OptimalThreshold:
    """Optimal threshold recommendation with confidence metrics"""
    value: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    auc_score: Optional[float] = None
    
    # Confidence intervals
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    confidence_level: float = 0.95
    
    # Validation method and metadata
    validation_method: str = "grid_search"
    optimization_criterion: str = "f1_score"
    total_evaluations: int = 0
    best_threshold_rank: int = 1
    
    # Performance comparison
    baseline_performance: Optional[Dict[str, float]] = None
    improvement_over_baseline: Optional[Dict[str, float]] = None
    
    # Recommendation metadata
    recommendation_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "unknown"
    validation_data_period: Optional[Tuple[datetime, datetime]] = None

class ThresholdOptimizer:
    """Grid search optimizer for optimal threshold selection"""
    
    def __init__(self, 
                 threshold_range: Tuple[float, float] = (0.1, 0.95),
                 threshold_step: float = 0.05,
                 optimization_metric: str = "f1_score",
                 cross_validation_folds: int = 5,
                 confidence_level: float = 0.95):
        """
        Initialize threshold optimizer
        
        Args:
            threshold_range: (min, max) threshold values to test
            threshold_step: Step size for threshold grid search
            optimization_metric: Metric to optimize ('f1_score', 'precision', 'recall', 'accuracy')
            cross_validation_folds: Number of CV folds for confidence estimation
            confidence_level: Confidence level for statistical intervals
        """
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Optimization parameters
        self.threshold_range = threshold_range
        self.threshold_step = threshold_step
        self.optimization_metric = optimization_metric
        self.cross_validation_folds = cross_validation_folds
        self.confidence_level = confidence_level
        
        # Generate threshold grid
        self.threshold_grid = self._generate_threshold_grid()
        
        # Results storage
        self.evaluation_results: List[ThresholdMetrics] = []
        self.optimal_threshold: Optional[OptimalThreshold] = None
        
    def _generate_threshold_grid(self) -> List[float]:
        """Generate grid of threshold values to evaluate"""
        min_thresh, max_thresh = self.threshold_range
        
        # Create base grid with specified step size
        base_grid = np.arange(min_thresh, max_thresh + self.threshold_step, self.threshold_step)
        
        # Add commonly used thresholds from requirements
        common_thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        
        # Combine and deduplicate
        all_thresholds = list(set(list(base_grid) + common_thresholds))
        
        # Filter to range and sort
        filtered_thresholds = [t for t in all_thresholds if min_thresh <= t <= max_thresh]
        filtered_thresholds.sort()
        
        return filtered_thresholds
    
    def optimize_threshold(self, 
                          model: XGBoostRiskModel,
                          X_validation: np.ndarray,
                          y_validation: np.ndarray,
                          validation_period: Optional[Tuple[datetime, datetime]] = None) -> OptimalThreshold:
        """
        Find optimal threshold using grid search on validation data
        
        Args:
            model: Trained XGBoost model
            X_validation: Validation feature matrix
            y_validation: Validation labels
            validation_period: Optional period info for metadata
            
        Returns:
            OptimalThreshold object with recommendation and confidence metrics
        """
        start_time = time.time()
        
        try:
            if not model.is_trained:
                raise ValueError("Model must be trained before threshold optimization")
            
            if X_validation.size == 0 or len(y_validation) == 0:
                raise ValueError("Empty validation data provided")
            
            # Get model predictions (probabilities)
            _, probabilities = model.predict(X_validation, return_probabilities=True)
            
            # Evaluate each threshold
            self.evaluation_results = []
            
            for threshold in self.threshold_grid:
                metrics = self._evaluate_threshold(
                    threshold, probabilities, y_validation
                )
                self.evaluation_results.append(metrics)
                
                self.data_logger.log_data_collection(
                    "threshold_evaluation", f"threshold_{threshold:.3f}", 
                    1, True, f"F1: {metrics.f1_score:.3f}, Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}"
                )
            
            # Find optimal threshold based on optimization metric
            optimal_metrics = self._select_optimal_threshold()
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(
                optimal_metrics, X_validation, y_validation, model
            )
            
            # Create optimal threshold recommendation
            self.optimal_threshold = OptimalThreshold(
                value=optimal_metrics.threshold,
                f1_score=optimal_metrics.f1_score,
                precision=optimal_metrics.precision,
                recall=optimal_metrics.recall,
                accuracy=optimal_metrics.accuracy,
                auc_score=optimal_metrics.auc_score,
                confidence_interval=confidence_interval,
                confidence_level=self.confidence_level,
                validation_method="grid_search",
                optimization_criterion=self.optimization_metric,
                total_evaluations=len(self.evaluation_results),
                best_threshold_rank=1,
                model_version=model.model_version,
                validation_data_period=validation_period
            )
            
            # Calculate baseline comparison (threshold = 0.5)
            baseline_metrics = self._get_baseline_performance(probabilities, y_validation)
            if baseline_metrics:
                self.optimal_threshold.baseline_performance = baseline_metrics
                self.optimal_threshold.improvement_over_baseline = self._calculate_improvement(
                    optimal_metrics, baseline_metrics
                )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("threshold_optimization", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "threshold_optimization", "complete", 1, True,
                f"Optimal threshold: {self.optimal_threshold.value:.3f} with {self.optimization_metric}: {getattr(self.optimal_threshold, self.optimization_metric):.3f}"
            )
            
            return self.optimal_threshold
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("threshold_optimization", duration_ms, False)
            self.data_logger.log_data_collection("threshold_optimization", "error", 0, False, str(e))
            raise
    
    def _evaluate_threshold(self, threshold: float, probabilities: np.ndarray, y_true: np.ndarray) -> ThresholdMetrics:
        """Evaluate performance metrics for a specific threshold"""
        try:
            # Convert probabilities to binary predictions
            y_pred = (probabilities >= threshold).astype(int)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # AUC score (independent of threshold)
            auc = None
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, probabilities)
            
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Risk-specific metrics
            risk_events_detected = tp
            risk_events_missed = fn
            false_alarms = fp
            
            return ThresholdMetrics(
                threshold=threshold,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                auc_score=float(auc) if auc is not None else None,
                true_positives=int(tp),
                false_positives=int(fp),
                true_negatives=int(tn),
                false_negatives=int(fn),
                risk_events_detected=int(risk_events_detected),
                risk_events_missed=int(risk_events_missed),
                false_alarms=int(false_alarms),
                sample_size=len(y_true)
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "threshold_evaluation", f"threshold_{threshold:.3f}", 0, False, str(e)
            )
            # Return zero metrics on error
            return ThresholdMetrics(
                threshold=threshold,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                sample_size=len(y_true) if len(y_true) > 0 else 0
            )
    
    def _select_optimal_threshold(self) -> ThresholdMetrics:
        """Select optimal threshold based on optimization metric"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        # Sort by optimization metric (descending)
        sorted_results = sorted(
            self.evaluation_results,
            key=lambda x: getattr(x, self.optimization_metric),
            reverse=True
        )
        
        return sorted_results[0]
    
    def _calculate_confidence_interval(self, 
                                     optimal_metrics: ThresholdMetrics,
                                     X_validation: np.ndarray,
                                     y_validation: np.ndarray,
                                     model: XGBoostRiskModel) -> Tuple[float, float]:
        """Calculate confidence interval for optimal threshold using bootstrap"""
        try:
            if len(y_validation) < 30:  # Too small for reliable CI
                return (optimal_metrics.threshold - 0.05, optimal_metrics.threshold + 0.05)
            
            # Bootstrap sampling for confidence interval
            n_bootstrap = 100
            bootstrap_thresholds = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_validation), size=len(X_validation), replace=True)
                X_boot = X_validation[indices]
                y_boot = y_validation[indices]
                
                # Get predictions for bootstrap sample
                _, prob_boot = model.predict(X_boot, return_probabilities=True)
                
                # Find optimal threshold for this bootstrap sample
                boot_results = []
                for threshold in self.threshold_grid:
                    metrics = self._evaluate_threshold(threshold, prob_boot, y_boot)
                    boot_results.append(metrics)
                
                # Select best threshold for this bootstrap
                best_boot = max(boot_results, key=lambda x: getattr(x, self.optimization_metric))
                bootstrap_thresholds.append(best_boot.threshold)
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_thresholds, lower_percentile)
            ci_upper = np.percentile(bootstrap_thresholds, upper_percentile)
            
            return (float(ci_lower), float(ci_upper))
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "confidence_interval", "calculation", 0, False, str(e)
            )
            # Return wide interval on error
            return (max(0.0, optimal_metrics.threshold - 0.1), 
                   min(1.0, optimal_metrics.threshold + 0.1))
    
    def _get_baseline_performance(self, probabilities: np.ndarray, y_true: np.ndarray) -> Optional[Dict[str, float]]:
        """Get baseline performance using threshold = 0.5"""
        try:
            baseline_metrics = self._evaluate_threshold(0.5, probabilities, y_true)
            return {
                'threshold': 0.5,
                'accuracy': baseline_metrics.accuracy,
                'precision': baseline_metrics.precision,
                'recall': baseline_metrics.recall,
                'f1_score': baseline_metrics.f1_score,
                'auc_score': baseline_metrics.auc_score
            }
        except Exception:
            return None
    
    def _calculate_improvement(self, optimal_metrics: ThresholdMetrics, baseline: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement over baseline performance"""
        improvements = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            optimal_value = getattr(optimal_metrics, metric)
            baseline_value = baseline.get(metric, 0.0)
            
            if baseline_value > 0:
                improvement = (optimal_value - baseline_value) / baseline_value
                improvements[f'{metric}_improvement'] = improvement
            else:
                improvements[f'{metric}_improvement'] = 0.0
        
        return improvements
    
    def get_threshold_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all threshold evaluations"""
        if not self.evaluation_results:
            return {}
        
        # Extract metrics for analysis
        thresholds = [r.threshold for r in self.evaluation_results]
        f1_scores = [r.f1_score for r in self.evaluation_results]
        precisions = [r.precision for r in self.evaluation_results]
        recalls = [r.recall for r in self.evaluation_results]
        accuracies = [r.accuracy for r in self.evaluation_results]
        
        return {
            'total_thresholds_evaluated': len(self.evaluation_results),
            'threshold_range': {
                'min': min(thresholds),
                'max': max(thresholds),
                'step_size': self.threshold_step
            },
            'performance_summary': {
                'f1_score': {
                    'min': min(f1_scores),
                    'max': max(f1_scores),
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores)
                },
                'precision': {
                    'min': min(precisions),
                    'max': max(precisions),
                    'mean': np.mean(precisions),
                    'std': np.std(precisions)
                },
                'recall': {
                    'min': min(recalls),
                    'max': max(recalls),
                    'mean': np.mean(recalls),
                    'std': np.std(recalls)
                },
                'accuracy': {
                    'min': min(accuracies),
                    'max': max(accuracies),
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies)
                }
            },
            'optimization_details': {
                'optimization_metric': self.optimization_metric,
                'confidence_level': self.confidence_level,
                'cross_validation_folds': self.cross_validation_folds
            }
        }
    
    def get_top_thresholds(self, top_k: int = 5) -> List[ThresholdMetrics]:
        """Get top K performing thresholds"""
        if not self.evaluation_results:
            return []
        
        sorted_results = sorted(
            self.evaluation_results,
            key=lambda x: getattr(x, self.optimization_metric),
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to file"""
        try:
            results_data = {
                'optimization_parameters': {
                    'threshold_range': self.threshold_range,
                    'threshold_step': self.threshold_step,
                    'optimization_metric': self.optimization_metric,
                    'confidence_level': self.confidence_level
                },
                'threshold_grid': self.threshold_grid,
                'evaluation_results': [
                    {
                        'threshold': r.threshold,
                        'accuracy': r.accuracy,
                        'precision': r.precision,
                        'recall': r.recall,
                        'f1_score': r.f1_score,
                        'auc_score': r.auc_score,
                        'true_positives': r.true_positives,
                        'false_positives': r.false_positives,
                        'true_negatives': r.true_negatives,
                        'false_negatives': r.false_negatives,
                        'sample_size': r.sample_size
                    }
                    for r in self.evaluation_results
                ],
                'optimal_threshold': {
                    'value': self.optimal_threshold.value,
                    'f1_score': self.optimal_threshold.f1_score,
                    'precision': self.optimal_threshold.precision,
                    'recall': self.optimal_threshold.recall,
                    'accuracy': self.optimal_threshold.accuracy,
                    'confidence_interval': self.optimal_threshold.confidence_interval,
                    'validation_method': self.optimal_threshold.validation_method,
                    'optimization_criterion': self.optimal_threshold.optimization_criterion,
                    'model_version': self.optimal_threshold.model_version
                } if self.optimal_threshold else None,
                'evaluation_summary': self.get_threshold_evaluation_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.data_logger.log_data_collection("optimization_results", "save", 1, True, f"Saved to {filepath}")
        
        except Exception as e:
            self.data_logger.log_data_collection("optimization_results", "save", 0, False, str(e))
            raise
    
    def load_optimization_results(self, filepath: str) -> None:
        """Load optimization results from file"""
        try:
            with open(filepath, 'r') as f:
                results_data = json.load(f)
            
            # Restore optimization parameters
            params = results_data.get('optimization_parameters', {})
            self.threshold_range = tuple(params.get('threshold_range', self.threshold_range))
            self.threshold_step = params.get('threshold_step', self.threshold_step)
            self.optimization_metric = params.get('optimization_metric', self.optimization_metric)
            self.confidence_level = params.get('confidence_level', self.confidence_level)
            
            # Restore threshold grid
            self.threshold_grid = results_data.get('threshold_grid', [])
            
            # Restore evaluation results
            self.evaluation_results = []
            for result_data in results_data.get('evaluation_results', []):
                metrics = ThresholdMetrics(
                    threshold=result_data['threshold'],
                    accuracy=result_data['accuracy'],
                    precision=result_data['precision'],
                    recall=result_data['recall'],
                    f1_score=result_data['f1_score'],
                    auc_score=result_data.get('auc_score'),
                    true_positives=result_data.get('true_positives', 0),
                    false_positives=result_data.get('false_positives', 0),
                    true_negatives=result_data.get('true_negatives', 0),
                    false_negatives=result_data.get('false_negatives', 0),
                    sample_size=result_data.get('sample_size', 0)
                )
                self.evaluation_results.append(metrics)
            
            # Restore optimal threshold
            optimal_data = results_data.get('optimal_threshold')
            if optimal_data:
                self.optimal_threshold = OptimalThreshold(
                    value=optimal_data['value'],
                    f1_score=optimal_data['f1_score'],
                    precision=optimal_data['precision'],
                    recall=optimal_data['recall'],
                    accuracy=optimal_data['accuracy'],
                    confidence_interval=tuple(optimal_data.get('confidence_interval', (0.0, 1.0))),
                    validation_method=optimal_data.get('validation_method', 'grid_search'),
                    optimization_criterion=optimal_data.get('optimization_criterion', 'f1_score'),
                    model_version=optimal_data.get('model_version', 'unknown')
                )
            
            self.data_logger.log_data_collection("optimization_results", "load", 1, True, f"Loaded from {filepath}")
        
        except Exception as e:
            self.data_logger.log_data_collection("optimization_results", "load", 0, False, str(e))
            raise