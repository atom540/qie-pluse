"""
Threshold Optimization Integration Module
Integrates threshold optimization, F1 optimization, and configuration management
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.ml_trainer import XGBoostRiskModel
from services.timeseries.threshold_optimizer import ThresholdOptimizer, OptimalThreshold
from services.timeseries.f1_optimizer import F1ScoreOptimizer
from services.timeseries.threshold_config import ThresholdConfigurationManager, ThresholdConfiguration

class IntegratedThresholdOptimizer:
    """Integrated threshold optimization system combining all optimization approaches"""
    
    def __init__(self, config_dir: str = "config/thresholds"):
        """
        Initialize integrated threshold optimizer
        
        Args:
            config_dir: Directory for threshold configurations
        """
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Load threshold optimization parameters
        self.threshold_params = self._load_threshold_parameters()
        
        # Initialize optimizers
        self.grid_optimizer = ThresholdOptimizer(
            threshold_range=tuple(self.threshold_params.get('default_threshold_range', [0.1, 0.95])),
            threshold_step=self.threshold_params.get('default_threshold_step', 0.05),
            optimization_metric=self.threshold_params.get('optimization_metric', 'f1_score'),
            confidence_level=self.threshold_params.get('confidence_level', 0.95)
        )
        
        self.f1_optimizer = F1ScoreOptimizer(
            crash_event_weight=self.threshold_params.get('crash_event_weight', 2.0),
            precision_recall_balance=self.threshold_params.get('precision_recall_balance', 1.0),
            confidence_level=self.threshold_params.get('confidence_level', 0.95)
        )
        
        # Configuration manager
        self.config_manager = ThresholdConfigurationManager(config_dir)
        
        # Results storage
        self.optimization_results: Dict[str, Any] = {}
    
    def _load_threshold_parameters(self) -> Dict[str, Any]:
        """Load threshold optimization parameters from configuration"""
        try:
            config_path = Path("config/model_params.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return config_data.get('threshold_optimization', {})
            return {}
        except Exception as e:
            self.data_logger.log_data_collection(
                "threshold_params", "load", 0, False, str(e)
            )
            return {}
    
    def optimize_threshold_comprehensive(self,
                                       model: XGBoostRiskModel,
                                       X_validation: np.ndarray,
                                       y_validation: np.ndarray,
                                       crash_event_labels: Optional[np.ndarray] = None,
                                       business_context: Optional[Dict[str, Any]] = None,
                                       validation_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive threshold optimization using multiple approaches
        
        Args:
            model: Trained XGBoost model
            X_validation: Validation feature matrix
            y_validation: Validation labels
            crash_event_labels: Optional crash event period labels
            business_context: Optional business context for optimization
            validation_period: Optional validation data period info
            
        Returns:
            Comprehensive optimization results
        """
        start_time = time.time()
        
        try:
            self.data_logger.log_data_collection(
                "comprehensive_optimization", "start", 1, True,
                f"Starting comprehensive threshold optimization for model {model.model_version}"
            )
            
            # Step 1: Grid search optimization
            self.data_logger.log_data_collection(
                "comprehensive_optimization", "grid_search", 1, True, "Running grid search optimization"
            )
            
            grid_optimal = self.grid_optimizer.optimize_threshold(
                model, X_validation, y_validation, validation_period
            )
            
            # Step 2: F1 score optimization
            self.data_logger.log_data_collection(
                "comprehensive_optimization", "f1_optimization", 1, True, "Running F1 score optimization"
            )
            
            # Get threshold candidates based on business context
            threshold_candidates = self._get_threshold_candidates(business_context)
            
            f1_optimal = self.f1_optimizer.optimize_f1_score(
                model, X_validation, y_validation, crash_event_labels, threshold_candidates
            )
            
            # Step 3: Compare and select best approach
            best_threshold = self._select_best_threshold(grid_optimal, f1_optimal, business_context)
            
            # Step 4: Generate comprehensive results
            optimization_results = {
                'best_threshold': best_threshold,
                'grid_search_results': {
                    'optimal_threshold': grid_optimal,
                    'evaluation_summary': self.grid_optimizer.get_threshold_evaluation_summary(),
                    'top_thresholds': self.grid_optimizer.get_top_thresholds(5)
                },
                'f1_optimization_results': {
                    'optimal_threshold': f1_optimal,
                    'optimization_summary': self.f1_optimizer.get_f1_optimization_summary()
                },
                'comparison_analysis': self._compare_optimization_approaches(grid_optimal, f1_optimal),
                'business_context_analysis': self._analyze_business_context_impact(
                    best_threshold, business_context
                ) if business_context else None,
                'optimization_metadata': {
                    'model_version': model.model_version,
                    'validation_sample_size': len(y_validation),
                    'validation_period': validation_period,
                    'crash_events_included': crash_event_labels is not None,
                    'optimization_timestamp': datetime.now().isoformat(),
                    'total_duration_ms': 0  # Will be updated below
                }
            }
            
            # Step 5: Save configuration
            validation_info = {
                'sample_size': len(y_validation),
                'data_period': validation_period
            }
            
            config_path = self.config_manager.save_optimal_threshold(
                best_threshold, model.model_version, validation_info, business_context
            )
            
            optimization_results['configuration_saved_to'] = config_path
            
            # Step 6: Generate recommendation report
            recommendation_report = self.config_manager.generate_threshold_recommendation_report(
                model.model_version, include_history=True
            )
            optimization_results['recommendation_report'] = recommendation_report
            
            # Update duration
            duration_ms = (time.time() - start_time) * 1000
            optimization_results['optimization_metadata']['total_duration_ms'] = duration_ms
            
            # Store results
            self.optimization_results = optimization_results
            
            self.performance_logger.log_inference_time("comprehensive_optimization", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "comprehensive_optimization", "complete", 1, True,
                f"Completed comprehensive optimization. Best threshold: {best_threshold.value:.3f}"
            )
            
            return optimization_results
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("comprehensive_optimization", duration_ms, False)
            self.data_logger.log_data_collection("comprehensive_optimization", "error", 0, False, str(e))
            raise
    
    def _get_threshold_candidates(self, business_context: Optional[Dict[str, Any]]) -> List[float]:
        """Get threshold candidates based on business context"""
        try:
            # Load threshold candidates from configuration
            config_path = Path("config/model_params.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                candidates_config = config_data.get('threshold_candidates', {})
            else:
                candidates_config = {}
            
            # Select candidates based on business context
            if business_context:
                risk_tolerance = business_context.get('risk_tolerance', 'medium')
                
                if risk_tolerance == 'low':
                    # More aggressive thresholds for higher recall
                    return candidates_config.get('aggressive', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                elif risk_tolerance == 'high':
                    # More conservative thresholds for higher precision
                    return candidates_config.get('conservative', [0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
                else:
                    # Standard thresholds for balanced approach
                    return candidates_config.get('standard', [0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
            
            # Default to standard candidates
            return candidates_config.get('standard', [0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
        
        except Exception as e:
            self.data_logger.log_data_collection("threshold_candidates", "selection", 0, False, str(e))
            return [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    def _select_best_threshold(self, 
                             grid_optimal: OptimalThreshold,
                             f1_optimal: OptimalThreshold,
                             business_context: Optional[Dict[str, Any]]) -> OptimalThreshold:
        """Select the best threshold from optimization approaches"""
        try:
            # Compare F1 scores
            grid_f1 = grid_optimal.f1_score
            f1_f1 = f1_optimal.f1_score
            
            # If F1 scores are very close (within 0.01), consider other factors
            if abs(grid_f1 - f1_f1) < 0.01:
                # Consider confidence interval width (narrower is better)
                grid_ci_width = grid_optimal.confidence_interval[1] - grid_optimal.confidence_interval[0]
                f1_ci_width = f1_optimal.confidence_interval[1] - f1_optimal.confidence_interval[0]
                
                if grid_ci_width < f1_ci_width:
                    selected = grid_optimal
                    selection_reason = "Similar F1 scores, grid search has narrower confidence interval"
                else:
                    selected = f1_optimal
                    selection_reason = "Similar F1 scores, F1 optimization has narrower confidence interval"
            
            elif grid_f1 > f1_f1:
                selected = grid_optimal
                selection_reason = f"Grid search achieved higher F1 score ({grid_f1:.3f} vs {f1_f1:.3f})"
            
            else:
                selected = f1_optimal
                selection_reason = f"F1 optimization achieved higher F1 score ({f1_f1:.3f} vs {grid_f1:.3f})"
            
            # Apply business context adjustments if provided
            if business_context:
                selected = self._apply_business_context_adjustments(selected, business_context)
            
            # Add selection metadata
            selected.optimization_criterion = "comprehensive_selection"
            selected.validation_method = "comprehensive_optimization"
            
            self.data_logger.log_data_collection(
                "threshold_selection", "best", 1, True, selection_reason
            )
            
            return selected
        
        except Exception as e:
            self.data_logger.log_data_collection("threshold_selection", "error", 0, False, str(e))
            # Fallback to grid search result
            return grid_optimal
    
    def _apply_business_context_adjustments(self, 
                                          threshold: OptimalThreshold,
                                          business_context: Dict[str, Any]) -> OptimalThreshold:
        """Apply business context adjustments to threshold"""
        try:
            # Load business context defaults
            config_path = Path("config/model_params.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                business_defaults = config_data.get('business_context_defaults', {})
            else:
                business_defaults = {}
            
            risk_tolerance = business_context.get('risk_tolerance', 'medium')
            context_key = f"risk_tolerance_{risk_tolerance}"
            
            if context_key in business_defaults:
                context_config = business_defaults[context_key]
                threshold_bias = context_config.get('threshold_bias', 0.0)
                
                # Apply threshold bias
                adjusted_value = max(0.0, min(1.0, threshold.value + threshold_bias))
                
                if adjusted_value != threshold.value:
                    self.data_logger.log_data_collection(
                        "business_adjustment", "threshold_bias", 1, True,
                        f"Applied business context bias: {threshold.value:.3f} -> {adjusted_value:.3f}"
                    )
                    
                    # Create adjusted threshold
                    adjusted_threshold = OptimalThreshold(
                        value=adjusted_value,
                        f1_score=threshold.f1_score,
                        precision=threshold.precision,
                        recall=threshold.recall,
                        accuracy=threshold.accuracy,
                        auc_score=threshold.auc_score,
                        confidence_interval=threshold.confidence_interval,
                        confidence_level=threshold.confidence_level,
                        validation_method=f"{threshold.validation_method}_business_adjusted",
                        optimization_criterion=threshold.optimization_criterion,
                        total_evaluations=threshold.total_evaluations,
                        model_version=threshold.model_version,
                        baseline_performance=threshold.baseline_performance,
                        improvement_over_baseline=threshold.improvement_over_baseline
                    )
                    
                    return adjusted_threshold
            
            return threshold
        
        except Exception as e:
            self.data_logger.log_data_collection("business_adjustment", "error", 0, False, str(e))
            return threshold
    
    def _compare_optimization_approaches(self, 
                                       grid_optimal: OptimalThreshold,
                                       f1_optimal: OptimalThreshold) -> Dict[str, Any]:
        """Compare different optimization approaches"""
        return {
            'threshold_difference': abs(grid_optimal.value - f1_optimal.value),
            'f1_score_difference': abs(grid_optimal.f1_score - f1_optimal.f1_score),
            'precision_difference': abs(grid_optimal.precision - f1_optimal.precision),
            'recall_difference': abs(grid_optimal.recall - f1_optimal.recall),
            'grid_search_advantages': [
                "Comprehensive threshold space exploration",
                "Statistical confidence intervals",
                "Multiple metric optimization"
            ],
            'f1_optimization_advantages': [
                "Crash event weighting",
                "Precision-recall balance control",
                "Business context integration"
            ],
            'recommendation': self._get_approach_recommendation(grid_optimal, f1_optimal)
        }
    
    def _get_approach_recommendation(self, 
                                   grid_optimal: OptimalThreshold,
                                   f1_optimal: OptimalThreshold) -> str:
        """Get recommendation on which approach to use"""
        f1_diff = abs(grid_optimal.f1_score - f1_optimal.f1_score)
        threshold_diff = abs(grid_optimal.value - f1_optimal.value)
        
        if f1_diff < 0.01 and threshold_diff < 0.05:
            return "Both approaches yield similar results. Grid search recommended for statistical rigor."
        elif grid_optimal.f1_score > f1_optimal.f1_score + 0.02:
            return "Grid search significantly outperforms F1 optimization. Use grid search results."
        elif f1_optimal.f1_score > grid_optimal.f1_score + 0.02:
            return "F1 optimization significantly outperforms grid search. Use F1 optimization results."
        else:
            return "Results are comparable. Consider business context and crash event importance."
    
    def _analyze_business_context_impact(self, 
                                       threshold: OptimalThreshold,
                                       business_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of business context on threshold selection"""
        analysis = {
            'risk_tolerance_alignment': 'unknown',
            'cost_benefit_analysis': {},
            'recommendations': []
        }
        
        try:
            risk_tolerance = business_context.get('risk_tolerance', 'medium')
            
            # Analyze risk tolerance alignment
            if risk_tolerance == 'low':
                if threshold.recall >= 0.85:
                    analysis['risk_tolerance_alignment'] = 'well_aligned'
                    analysis['recommendations'].append("Threshold aligns well with low risk tolerance")
                else:
                    analysis['risk_tolerance_alignment'] = 'misaligned'
                    analysis['recommendations'].append("Consider lowering threshold for better risk coverage")
            
            elif risk_tolerance == 'high':
                if threshold.precision >= 0.80:
                    analysis['risk_tolerance_alignment'] = 'well_aligned'
                    analysis['recommendations'].append("Threshold aligns well with high risk tolerance")
                else:
                    analysis['risk_tolerance_alignment'] = 'misaligned'
                    analysis['recommendations'].append("Consider raising threshold to reduce false alarms")
            
            else:  # medium
                if 0.7 <= threshold.precision <= 0.85 and 0.7 <= threshold.recall <= 0.85:
                    analysis['risk_tolerance_alignment'] = 'well_aligned'
                    analysis['recommendations'].append("Threshold provides good balance for medium risk tolerance")
                else:
                    analysis['risk_tolerance_alignment'] = 'partially_aligned'
                    analysis['recommendations'].append("Threshold may need adjustment for optimal balance")
            
            # Cost-benefit analysis if cost weights provided
            if 'false_alarm_cost' in business_context and 'missed_event_cost' in business_context:
                false_alarm_cost = business_context['false_alarm_cost']
                missed_event_cost = business_context['missed_event_cost']
                
                # Estimate expected cost
                fp_rate = 1 - threshold.precision if threshold.precision > 0 else 0.5
                fn_rate = 1 - threshold.recall if threshold.recall > 0 else 0.5
                
                expected_cost = (fp_rate * false_alarm_cost) + (fn_rate * missed_event_cost)
                
                analysis['cost_benefit_analysis'] = {
                    'expected_cost_per_prediction': expected_cost,
                    'false_alarm_cost_component': fp_rate * false_alarm_cost,
                    'missed_event_cost_component': fn_rate * missed_event_cost,
                    'cost_ratio': false_alarm_cost / missed_event_cost if missed_event_cost > 0 else 1.0
                }
                
                # Cost-based recommendations
                if missed_event_cost > false_alarm_cost * 3:
                    analysis['recommendations'].append("High miss cost suggests lowering threshold for better recall")
                elif false_alarm_cost > missed_event_cost * 3:
                    analysis['recommendations'].append("High false alarm cost suggests raising threshold for better precision")
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def get_threshold_for_prediction(self, model_version: str) -> Optional[float]:
        """
        Get the optimal threshold for making predictions with a specific model version
        
        Args:
            model_version: Version of the model
            
        Returns:
            Optimal threshold value or None if not found
        """
        return self.config_manager.get_threshold_for_model(model_version)
    
    def monitor_threshold_performance(self, 
                                    model_version: str,
                                    current_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor threshold performance and recommend reoptimization if needed
        
        Args:
            model_version: Model version to monitor
            current_performance: Current performance metrics
            
        Returns:
            Monitoring results and recommendations
        """
        try:
            config = self.config_manager.load_threshold_configuration(model_version)
            if not config:
                return {'error': f'No configuration found for model {model_version}'}
            
            monitoring_results = {
                'model_version': model_version,
                'current_performance': current_performance,
                'baseline_performance': {
                    'f1_score': config.f1_score,
                    'precision': config.precision,
                    'recall': config.recall,
                    'accuracy': config.accuracy
                },
                'performance_degradation': {},
                'recommendations': [],
                'reoptimization_needed': False
            }
            
            # Calculate performance degradation
            for metric in ['f1_score', 'precision', 'recall', 'accuracy']:
                if metric in current_performance:
                    baseline_value = getattr(config, metric, 0.0)
                    current_value = current_performance[metric]
                    
                    if baseline_value > 0:
                        degradation = (baseline_value - current_value) / baseline_value
                        monitoring_results['performance_degradation'][metric] = degradation
                        
                        # Check if degradation exceeds threshold
                        if degradation > (1 - config.performance_degradation_threshold):
                            monitoring_results['reoptimization_needed'] = True
                            monitoring_results['recommendations'].append(
                                f"{metric.upper()} degraded by {degradation:.1%}, consider reoptimization"
                            )
            
            # Check configuration age
            if config.created_timestamp:
                try:
                    created_date = datetime.fromisoformat(config.created_timestamp.replace('Z', '+00:00'))
                    age_days = (datetime.now() - created_date).days
                    
                    if age_days > config.reoptimization_frequency_days:
                        monitoring_results['recommendations'].append(
                            f"Configuration is {age_days} days old, consider reoptimization"
                        )
                        monitoring_results['reoptimization_needed'] = True
                
                except ValueError:
                    monitoring_results['recommendations'].append("Invalid timestamp, cannot determine age")
            
            return monitoring_results
        
        except Exception as e:
            return {'error': f'Monitoring failed: {str(e)}'}
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            return {'error': 'No optimization results available'}
        
        return {
            'optimization_summary': self.optimization_results,
            'configuration_status': self.config_manager.list_available_configurations(),
            'system_health': {
                'total_configurations': len(self.config_manager.list_available_configurations()),
                'optimization_approaches_available': ['grid_search', 'f1_optimization', 'comprehensive'],
                'last_optimization': self.optimization_results.get('optimization_metadata', {}).get('optimization_timestamp')
            }
        }