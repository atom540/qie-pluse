"""
F1 Score Optimization for AI Risk Oracle Model Training
Specialized optimizer for F1 score maximization with crash event focus
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.ml_trainer import XGBoostRiskModel
from services.timeseries.threshold_optimizer import OptimalThreshold

class F1ScoreOptimizer:
    """Specialized optimizer for F1 score maximization with crash event focus"""
    
    def __init__(self, 
                 crash_event_weight: float = 2.0,
                 precision_recall_balance: float = 1.0,
                 confidence_level: float = 0.95):
        """
        Initialize F1 score optimizer
        
        Args:
            crash_event_weight: Weight multiplier for crash event periods
            precision_recall_balance: Balance between precision and recall (1.0 = equal weight)
            confidence_level: Confidence level for statistical intervals
        """
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        self.crash_event_weight = crash_event_weight
        self.precision_recall_balance = precision_recall_balance
        self.confidence_level = confidence_level
        
        # F1 optimization results
        self.f1_optimization_results: List[Dict[str, Any]] = []
        self.optimal_f1_threshold: Optional[OptimalThreshold] = None
    
    def optimize_f1_score(self, 
                         model: XGBoostRiskModel,
                         X_validation: np.ndarray,
                         y_validation: np.ndarray,
                         crash_event_labels: Optional[np.ndarray] = None,
                         threshold_candidates: Optional[List[float]] = None) -> OptimalThreshold:
        """
        Optimize threshold for maximum F1 score with crash event focus
        
        Args:
            model: Trained XGBoost model
            X_validation: Validation feature matrix
            y_validation: Validation labels
            crash_event_labels: Optional labels indicating crash event periods
            threshold_candidates: Optional list of thresholds to evaluate
            
        Returns:
            OptimalThreshold with F1 score optimization results
        """
        start_time = time.time()
        
        try:
            if not model.is_trained:
                raise ValueError("Model must be trained before F1 optimization")
            
            # Get model predictions
            _, probabilities = model.predict(X_validation, return_probabilities=True)
            
            # Use default threshold candidates if not provided
            if threshold_candidates is None:
                threshold_candidates = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
            
            # Evaluate F1 score for each threshold
            f1_results = []
            
            for threshold in threshold_candidates:
                f1_metrics = self._evaluate_f1_threshold(
                    threshold, probabilities, y_validation, crash_event_labels
                )
                f1_results.append(f1_metrics)
                
                self.data_logger.log_data_collection(
                    "f1_optimization", f"threshold_{threshold:.3f}",
                    1, True, f"F1: {f1_metrics['f1_score']:.3f}, Weighted F1: {f1_metrics['weighted_f1_score']:.3f}"
                )
            
            self.f1_optimization_results = f1_results
            
            # Select optimal threshold based on weighted F1 score
            optimal_result = max(f1_results, key=lambda x: x['weighted_f1_score'])
            
            # Calculate confidence interval for optimal F1 score
            f1_confidence_interval = self._calculate_f1_confidence_interval(
                optimal_result, X_validation, y_validation, model, crash_event_labels
            )
            
            # Create optimal threshold recommendation
            self.optimal_f1_threshold = OptimalThreshold(
                value=optimal_result['threshold'],
                f1_score=optimal_result['f1_score'],
                precision=optimal_result['precision'],
                recall=optimal_result['recall'],
                accuracy=optimal_result['accuracy'],
                auc_score=optimal_result.get('auc_score'),
                confidence_interval=f1_confidence_interval,
                confidence_level=self.confidence_level,
                validation_method="f1_score_optimization",
                optimization_criterion="weighted_f1_score",
                total_evaluations=len(f1_results),
                model_version=model.model_version
            )
            
            # Add F1-specific metadata
            self.optimal_f1_threshold.baseline_performance = self._get_f1_baseline_performance(
                probabilities, y_validation, crash_event_labels
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("f1_score_optimization", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "f1_optimization", "complete", 1, True,
                f"Optimal F1 threshold: {self.optimal_f1_threshold.value:.3f} with F1: {self.optimal_f1_threshold.f1_score:.3f}"
            )
            
            return self.optimal_f1_threshold
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("f1_score_optimization", duration_ms, False)
            self.data_logger.log_data_collection("f1_optimization", "error", 0, False, str(e))
            raise
    
    def _evaluate_f1_threshold(self, 
                              threshold: float,
                              probabilities: np.ndarray,
                              y_true: np.ndarray,
                              crash_event_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate F1 score and related metrics for a specific threshold"""
        try:
            # Convert probabilities to predictions
            y_pred = (probabilities >= threshold).astype(int)
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # AUC score
            auc = None
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, probabilities)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate weighted F1 score for crash events
            weighted_f1 = f1
            crash_period_f1 = f1
            normal_period_f1 = f1
            
            if crash_event_labels is not None:
                # Separate crash and normal periods
                crash_mask = crash_event_labels == 1
                normal_mask = crash_event_labels == 0
                
                if np.any(crash_mask):
                    crash_y_true = y_true[crash_mask]
                    crash_y_pred = y_pred[crash_mask]
                    crash_period_f1 = f1_score(crash_y_true, crash_y_pred, zero_division=0)
                
                if np.any(normal_mask):
                    normal_y_true = y_true[normal_mask]
                    normal_y_pred = y_pred[normal_mask]
                    normal_period_f1 = f1_score(normal_y_true, normal_y_pred, zero_division=0)
                
                # Calculate weighted F1 (emphasize crash events)
                crash_weight = self.crash_event_weight
                normal_weight = 1.0
                
                total_weight = crash_weight + normal_weight
                weighted_f1 = (crash_period_f1 * crash_weight + normal_period_f1 * normal_weight) / total_weight
            
            # Precision-recall balance adjustment
            if self.precision_recall_balance != 1.0:
                # Adjust F1 score based on precision-recall preference
                if precision > 0 and recall > 0:
                    beta = self.precision_recall_balance
                    adjusted_f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                    weighted_f1 = (weighted_f1 + adjusted_f1) / 2
            
            return {
                'threshold': threshold,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'weighted_f1_score': float(weighted_f1),
                'crash_period_f1': float(crash_period_f1),
                'normal_period_f1': float(normal_period_f1),
                'auc_score': float(auc) if auc is not None else None,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'sample_size': len(y_true)
            }
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "f1_threshold_evaluation", f"threshold_{threshold:.3f}", 0, False, str(e)
            )
            return {
                'threshold': threshold,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'weighted_f1_score': 0.0,
                'error': str(e)
            }
    
    def _calculate_f1_confidence_interval(self, 
                                        optimal_result: Dict[str, Any],
                                        X_validation: np.ndarray,
                                        y_validation: np.ndarray,
                                        model: XGBoostRiskModel,
                                        crash_event_labels: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Calculate confidence interval for optimal F1 threshold using bootstrap"""
        try:
            if len(y_validation) < 30:
                return (max(0.0, optimal_result['threshold'] - 0.05), 
                       min(1.0, optimal_result['threshold'] + 0.05))
            
            # Bootstrap sampling
            n_bootstrap = 100
            bootstrap_f1_thresholds = []
            
            # Threshold candidates for bootstrap
            threshold_candidates = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_validation), size=len(X_validation), replace=True)
                X_boot = X_validation[indices]
                y_boot = y_validation[indices]
                crash_boot = crash_event_labels[indices] if crash_event_labels is not None else None
                
                # Get predictions for bootstrap sample
                _, prob_boot = model.predict(X_boot, return_probabilities=True)
                
                # Find optimal F1 threshold for this bootstrap sample
                boot_f1_results = []
                for threshold in threshold_candidates:
                    f1_metrics = self._evaluate_f1_threshold(threshold, prob_boot, y_boot, crash_boot)
                    boot_f1_results.append(f1_metrics)
                
                # Select best F1 threshold for this bootstrap
                best_f1_result = max(boot_f1_results, key=lambda x: x['weighted_f1_score'])
                bootstrap_f1_thresholds.append(best_f1_result['threshold'])
            
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_f1_thresholds, lower_percentile)
            ci_upper = np.percentile(bootstrap_f1_thresholds, upper_percentile)
            
            return (float(ci_lower), float(ci_upper))
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "f1_confidence_interval", "calculation", 0, False, str(e)
            )
            return (max(0.0, optimal_result['threshold'] - 0.1), 
                   min(1.0, optimal_result['threshold'] + 0.1))
    
    def _get_f1_baseline_performance(self, 
                                   probabilities: np.ndarray,
                                   y_true: np.ndarray,
                                   crash_event_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get baseline F1 performance using threshold = 0.5"""
        try:
            baseline_result = self._evaluate_f1_threshold(0.5, probabilities, y_true, crash_event_labels)
            return {
                'threshold': 0.5,
                'f1_score': baseline_result['f1_score'],
                'weighted_f1_score': baseline_result['weighted_f1_score'],
                'precision': baseline_result['precision'],
                'recall': baseline_result['recall'],
                'accuracy': baseline_result['accuracy']
            }
        except Exception:
            return {
                'threshold': 0.5,
                'f1_score': 0.0,
                'weighted_f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }
    
    def get_f1_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of F1 score optimization results"""
        if not self.f1_optimization_results:
            return {}
        
        f1_scores = [r['f1_score'] for r in self.f1_optimization_results]
        weighted_f1_scores = [r['weighted_f1_score'] for r in self.f1_optimization_results]
        thresholds = [r['threshold'] for r in self.f1_optimization_results]
        
        return {
            'total_thresholds_evaluated': len(self.f1_optimization_results),
            'f1_score_statistics': {
                'min': min(f1_scores),
                'max': max(f1_scores),
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores)
            },
            'weighted_f1_score_statistics': {
                'min': min(weighted_f1_scores),
                'max': max(weighted_f1_scores),
                'mean': np.mean(weighted_f1_scores),
                'std': np.std(weighted_f1_scores)
            },
            'optimal_threshold': {
                'value': self.optimal_f1_threshold.value if self.optimal_f1_threshold else None,
                'f1_score': self.optimal_f1_threshold.f1_score if self.optimal_f1_threshold else None,
                'confidence_interval': self.optimal_f1_threshold.confidence_interval if self.optimal_f1_threshold else None
            },
            'optimization_parameters': {
                'crash_event_weight': self.crash_event_weight,
                'precision_recall_balance': self.precision_recall_balance,
                'confidence_level': self.confidence_level
            }
        }