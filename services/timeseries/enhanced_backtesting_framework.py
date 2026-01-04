"""
Enhanced Backtesting Framework for AI Risk Oracle Model Training
Integrates crash event analysis, multi-horizon evaluation, and baseline comparisons
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.ml_trainer import XGBoostRiskModel, FeaturePreprocessor
from services.timeseries.oracle_processor import FeatureSet
from services.timeseries.training_pipeline import CrashEvent
from services.timeseries.crash_event_backtester import (
    CrashEventAnalyzer, MultiHorizonEvaluator, BaselineModelComparator, 
    ComprehensivePerformanceReporter, CrashEventMetrics, MultiHorizonMetrics, 
    BaselineModelResult
)

@dataclass
class EnhancedBacktestResult:
    """Comprehensive backtesting result with crash event analysis"""
    model_version: str
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    
    # Overall performance metrics
    overall_accuracy: float
    overall_precision: float
    overall_recall: float
    overall_f1_score: float
    overall_auc_score: Optional[float]
    
    # Crash event analysis results
    crash_event_results: List[CrashEventMetrics]
    crash_events_evaluated: int
    average_crash_detection_f1: float
    
    # Multi-horizon analysis results
    multi_horizon_results: Dict[str, MultiHorizonMetrics]
    best_horizon_performance: str
    
    # Baseline comparison results
    baseline_comparisons: Dict[str, BaselineModelResult]
    outperforms_baselines: bool
    
    # Comprehensive report
    performance_report: Dict[str, Any]
    recommendations: List[str]
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    total_evaluation_time_seconds: float = 0.0

class EnhancedBacktestingFramework:
    """
    Enhanced backtesting framework that provides comprehensive evaluation
    including crash event analysis, multi-horizon evaluation, and baseline comparisons
    """
    
    def __init__(self):
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        self.config = get_config()
        
        # Initialize component analyzers
        self.crash_analyzer = CrashEventAnalyzer()
        self.horizon_evaluator = MultiHorizonEvaluator()
        self.baseline_comparator = BaselineModelComparator()
        self.performance_reporter = ComprehensivePerformanceReporter()
    
    def run_enhanced_backtest(self,
                            model: XGBoostRiskModel,
                            historical_features: List[FeatureSet],
                            historical_outcomes: Dict[str, List[float]],
                            crash_events: List[CrashEvent],
                            evaluation_start: datetime,
                            evaluation_end: datetime,
                            include_baseline_comparison: bool = True,
                            include_multi_horizon: bool = True) -> EnhancedBacktestResult:
        """
        Run comprehensive enhanced backtesting with crash event analysis
        
        Args:
            model: Trained XGBoost model to evaluate
            historical_features: Historical feature data
            historical_outcomes: Historical outcome data
            crash_events: List of crash events to analyze
            evaluation_start: Start of evaluation period
            evaluation_end: End of evaluation period
            include_baseline_comparison: Whether to include baseline model comparisons
            include_multi_horizon: Whether to include multi-horizon evaluation
        
        Returns:
            EnhancedBacktestResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            self.data_logger.log_data_collection(
                "enhanced_backtesting", "start", 1, True,
                f"Starting enhanced backtest for {len(crash_events)} crash events"
            )
            
            # 1. Run crash event analysis
            crash_event_results = self._run_crash_event_analysis(
                model, historical_features, historical_outcomes, crash_events
            )
            
            # 2. Run multi-horizon evaluation (if requested)
            multi_horizon_results = {}
            if include_multi_horizon:
                multi_horizon_results = self._run_multi_horizon_evaluation(
                    model, historical_features, historical_outcomes, 
                    evaluation_start, evaluation_end
                )
            
            # 3. Run baseline comparisons (if requested)
            baseline_comparisons = {}
            if include_baseline_comparison:
                baseline_comparisons = self._run_baseline_comparisons(
                    model, historical_features, historical_outcomes,
                    evaluation_start, evaluation_end
                )
            
            # 4. Calculate overall performance metrics
            overall_metrics = self._calculate_overall_metrics(
                model, historical_features, historical_outcomes,
                evaluation_start, evaluation_end
            )
            
            # 5. Generate comprehensive performance report
            performance_report = self.performance_reporter.generate_comprehensive_report(
                crash_event_results, multi_horizon_results, baseline_comparisons,
                model.model_version, (evaluation_start, evaluation_end)
            )
            
            # 6. Create enhanced backtest result
            result = self._create_enhanced_result(
                model, evaluation_start, evaluation_end, overall_metrics,
                crash_event_results, multi_horizon_results, baseline_comparisons,
                performance_report, time.time() - start_time
            )
            
            self.data_logger.log_data_collection(
                "enhanced_backtesting", "complete", 1, True,
                f"Enhanced backtest completed with F1: {result.overall_f1_score:.3f}"
            )
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            self.data_logger.log_data_collection(
                "enhanced_backtesting", "error", 0, False, str(e)
            )
            self.performance_logger.log_inference_time("enhanced_backtesting", duration * 1000, False)
            raise
    
    def _run_crash_event_analysis(self,
                                model: XGBoostRiskModel,
                                historical_features: List[FeatureSet],
                                historical_outcomes: Dict[str, List[float]],
                                crash_events: List[CrashEvent]) -> List[CrashEventMetrics]:
        """Run crash event analysis for all provided crash events"""
        try:
            crash_results = []
            
            for crash_event in crash_events:
                try:
                    crash_result = self.crash_analyzer.evaluate_crash_event(
                        model, crash_event, historical_features, historical_outcomes
                    )
                    crash_results.append(crash_result)
                    
                    self.data_logger.log_data_collection(
                        "crash_event_analysis", crash_event.name, 1, True,
                        f"F1: {crash_result.f1_score:.3f}, Recall: {crash_result.recall:.3f}"
                    )
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "crash_event_analysis", crash_event.name, 0, False, str(e)
                    )
                    continue
            
            return crash_results
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "crash_event_analysis", "all_events", 0, False, str(e)
            )
            return []
    
    def _run_multi_horizon_evaluation(self,
                                    model: XGBoostRiskModel,
                                    historical_features: List[FeatureSet],
                                    historical_outcomes: Dict[str, List[float]],
                                    evaluation_start: datetime,
                                    evaluation_end: datetime) -> Dict[str, MultiHorizonMetrics]:
        """Run multi-horizon evaluation"""
        try:
            return self.horizon_evaluator.evaluate_multi_horizon_accuracy(
                model, historical_features, historical_outcomes,
                evaluation_start, evaluation_end
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "multi_horizon_evaluation", "all_horizons", 0, False, str(e)
            )
            return {}
    
    def _run_baseline_comparisons(self,
                                model: XGBoostRiskModel,
                                historical_features: List[FeatureSet],
                                historical_outcomes: Dict[str, List[float]],
                                evaluation_start: datetime,
                                evaluation_end: datetime) -> Dict[str, BaselineModelResult]:
        """Run baseline model comparisons"""
        try:
            return self.baseline_comparator.compare_with_baselines(
                model, historical_features, historical_outcomes,
                evaluation_start, evaluation_end
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "baseline_comparison", "all_baselines", 0, False, str(e)
            )
            return {}
    
    def _calculate_overall_metrics(self,
                                 model: XGBoostRiskModel,
                                 historical_features: List[FeatureSet],
                                 historical_outcomes: Dict[str, List[float]],
                                 evaluation_start: datetime,
                                 evaluation_end: datetime) -> Dict[str, float]:
        """Calculate overall performance metrics for the evaluation period"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Filter features for evaluation period
            period_features = [
                fs for fs in historical_features 
                if evaluation_start <= fs.timestamp <= evaluation_end
            ]
            
            if not period_features:
                return {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                    'f1_score': 0.0, 'auc_score': None
                }
            
            # Prepare features and labels
            preprocessor = FeaturePreprocessor()
            feature_df = preprocessor.prepare_features_from_feature_sets(period_features)
            labels = preprocessor.create_target_labels(feature_df, historical_outcomes)
            
            # Transform features
            X = preprocessor.transform(feature_df)
            y = labels.values
            
            if X.size == 0:
                return {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                    'f1_score': 0.0, 'auc_score': None
                }
            
            # Make predictions
            predictions, probabilities = model.predict(X, return_probabilities=True)
            
            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions, zero_division=0)
            recall = recall_score(y, predictions, zero_division=0)
            f1 = f1_score(y, predictions, zero_division=0)
            
            # AUC score if we have both classes
            auc = None
            if len(np.unique(y)) > 1:
                auc = roc_auc_score(y, probabilities)
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_score': float(auc) if auc is not None else None
            }
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "overall_metrics", "calculation", 0, False, str(e)
            )
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1_score': 0.0, 'auc_score': None
            }
    
    def _create_enhanced_result(self,
                              model: XGBoostRiskModel,
                              evaluation_start: datetime,
                              evaluation_end: datetime,
                              overall_metrics: Dict[str, float],
                              crash_event_results: List[CrashEventMetrics],
                              multi_horizon_results: Dict[str, MultiHorizonMetrics],
                              baseline_comparisons: Dict[str, BaselineModelResult],
                              performance_report: Dict[str, Any],
                              evaluation_time: float) -> EnhancedBacktestResult:
        """Create comprehensive enhanced backtest result"""
        try:
            # Calculate crash event summary metrics
            if crash_event_results:
                avg_crash_f1 = np.mean([r.f1_score for r in crash_event_results])
            else:
                avg_crash_f1 = 0.0
            
            # Determine best horizon performance
            best_horizon = "N/A"
            if multi_horizon_results:
                best_horizon = max(
                    multi_horizon_results.keys(),
                    key=lambda h: multi_horizon_results[h].f1_score
                )
            
            # Check if model outperforms baselines
            outperforms_baselines = True
            if baseline_comparisons:
                for result in baseline_comparisons.values():
                    if result.performance_difference <= 0:
                        outperforms_baselines = False
                        break
            
            # Extract recommendations from performance report
            recommendations = performance_report.get('performance_recommendations', [])
            
            return EnhancedBacktestResult(
                model_version=model.model_version,
                evaluation_period_start=evaluation_start,
                evaluation_period_end=evaluation_end,
                overall_accuracy=overall_metrics['accuracy'],
                overall_precision=overall_metrics['precision'],
                overall_recall=overall_metrics['recall'],
                overall_f1_score=overall_metrics['f1_score'],
                overall_auc_score=overall_metrics['auc_score'],
                crash_event_results=crash_event_results,
                crash_events_evaluated=len(crash_event_results),
                average_crash_detection_f1=float(avg_crash_f1),
                multi_horizon_results=multi_horizon_results,
                best_horizon_performance=best_horizon,
                baseline_comparisons=baseline_comparisons,
                outperforms_baselines=outperforms_baselines,
                performance_report=performance_report,
                recommendations=recommendations,
                total_evaluation_time_seconds=float(evaluation_time)
            )
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_result_creation", "error", 0, False, str(e)
            )
            # Return minimal result on error
            return EnhancedBacktestResult(
                model_version=model.model_version,
                evaluation_period_start=evaluation_start,
                evaluation_period_end=evaluation_end,
                overall_accuracy=0.0, overall_precision=0.0, overall_recall=0.0,
                overall_f1_score=0.0, overall_auc_score=None,
                crash_event_results=[], crash_events_evaluated=0,
                average_crash_detection_f1=0.0, multi_horizon_results={},
                best_horizon_performance="N/A", baseline_comparisons={},
                outperforms_baselines=False, performance_report={},
                recommendations=["Error in evaluation"], total_evaluation_time_seconds=0.0
            )
    
    def save_enhanced_backtest_result(self, 
                                    result: EnhancedBacktestResult,
                                    output_dir: str = "data/backtesting_results") -> str:
        """Save enhanced backtest result to file"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = result.evaluation_timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_backtest_{result.model_version}_{timestamp}.json"
            filepath = output_path / filename
            
            # Convert result to dictionary for JSON serialization
            result_dict = {
                'metadata': {
                    'model_version': result.model_version,
                    'evaluation_period_start': result.evaluation_period_start.isoformat(),
                    'evaluation_period_end': result.evaluation_period_end.isoformat(),
                    'evaluation_timestamp': result.evaluation_timestamp.isoformat(),
                    'total_evaluation_time_seconds': result.total_evaluation_time_seconds
                },
                'overall_performance': {
                    'accuracy': result.overall_accuracy,
                    'precision': result.overall_precision,
                    'recall': result.overall_recall,
                    'f1_score': result.overall_f1_score,
                    'auc_score': result.overall_auc_score
                },
                'crash_event_analysis': {
                    'crash_events_evaluated': result.crash_events_evaluated,
                    'average_crash_detection_f1': result.average_crash_detection_f1,
                    'individual_results': [
                        {
                            'event_name': r.crash_event_name,
                            'f1_score': r.f1_score,
                            'recall': r.recall,
                            'precision': r.precision,
                            'accuracy': r.accuracy,
                            'crash_detection_rate': r.crash_detection_rate,
                            'false_alarm_rate': r.false_alarm_rate
                        }
                        for r in result.crash_event_results
                    ]
                },
                'multi_horizon_analysis': {
                    'best_horizon_performance': result.best_horizon_performance,
                    'horizon_results': {
                        horizon: {
                            'accuracy': metrics.accuracy,
                            'f1_score': metrics.f1_score,
                            'average_lead_time_minutes': metrics.average_lead_time_minutes
                        }
                        for horizon, metrics in result.multi_horizon_results.items()
                    }
                },
                'baseline_comparison': {
                    'outperforms_baselines': result.outperforms_baselines,
                    'baseline_results': {
                        name: {
                            'f1_score': baseline.f1_score,
                            'performance_difference': baseline.performance_difference,
                            'statistical_significance': baseline.statistical_significance
                        }
                        for name, baseline in result.baseline_comparisons.items()
                    }
                },
                'performance_report': result.performance_report,
                'recommendations': result.recommendations
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.data_logger.log_data_collection(
                "enhanced_backtest_save", str(filepath), 1, True,
                f"Saved enhanced backtest result to {filepath}"
            )
            
            return str(filepath)
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "enhanced_backtest_save", "error", 0, False, str(e)
            )
            raise