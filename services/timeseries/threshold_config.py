"""
Threshold Configuration and Persistence for AI Risk Oracle Model Training
Handles saving, loading, and managing optimal threshold configurations
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import time

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.timeseries.threshold_optimizer import OptimalThreshold

@dataclass
class ThresholdConfiguration:
    """Configuration for model threshold settings"""
    model_version: str
    optimal_threshold: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    auc_score: Optional[float] = None
    
    # Confidence metrics
    confidence_interval: tuple = (0.0, 1.0)
    confidence_level: float = 0.95
    
    # Optimization metadata
    optimization_method: str = "grid_search"
    optimization_criterion: str = "f1_score"
    total_evaluations: int = 0
    
    # Validation data info
    validation_data_period: Optional[tuple] = None
    validation_sample_size: int = 0
    
    # Configuration metadata
    created_timestamp: str = ""
    last_updated_timestamp: str = ""
    configuration_version: str = "1.0"
    
    # Performance comparison
    baseline_performance: Optional[Dict[str, float]] = None
    improvement_over_baseline: Optional[Dict[str, float]] = None
    
    # Business context
    risk_tolerance_level: str = "medium"  # low, medium, high
    false_alarm_cost_weight: float = 1.0
    missed_event_cost_weight: float = 1.0
    
    # Monitoring settings
    performance_degradation_threshold: float = 0.9  # Trigger reoptimization at 90% of optimal
    reoptimization_frequency_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfiguration':
        """Create from dictionary"""
        return cls(**data)

class ThresholdConfigurationManager:
    """Manager for threshold configuration persistence and loading"""
    
    def __init__(self, config_dir: str = "config/thresholds"):
        """
        Initialize threshold configuration manager
        
        Args:
            config_dir: Directory to store threshold configurations
        """
        self.config = get_config()
        self.data_logger = get_data_logger()
        self.performance_logger = get_performance_logger()
        
        # Configuration storage
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Current configuration
        self.current_config: Optional[ThresholdConfiguration] = None
        self.config_history: List[ThresholdConfiguration] = []
    
    def save_optimal_threshold(self, 
                             optimal_threshold: OptimalThreshold,
                             model_version: str,
                             validation_info: Optional[Dict[str, Any]] = None,
                             business_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Save optimal threshold configuration to file
        
        Args:
            optimal_threshold: OptimalThreshold object from optimization
            model_version: Version of the model this threshold is for
            validation_info: Optional validation data information
            business_context: Optional business context settings
            
        Returns:
            Path to saved configuration file
        """
        start_time = time.time()
        
        try:
            # Create threshold configuration
            config = ThresholdConfiguration(
                model_version=model_version,
                optimal_threshold=optimal_threshold.value,
                f1_score=optimal_threshold.f1_score,
                precision=optimal_threshold.precision,
                recall=optimal_threshold.recall,
                accuracy=optimal_threshold.accuracy,
                auc_score=optimal_threshold.auc_score,
                confidence_interval=optimal_threshold.confidence_interval,
                confidence_level=optimal_threshold.confidence_level,
                optimization_method=optimal_threshold.validation_method,
                optimization_criterion=optimal_threshold.optimization_criterion,
                total_evaluations=optimal_threshold.total_evaluations,
                validation_data_period=optimal_threshold.validation_data_period,
                created_timestamp=datetime.now().isoformat(),
                last_updated_timestamp=datetime.now().isoformat(),
                baseline_performance=optimal_threshold.baseline_performance,
                improvement_over_baseline=optimal_threshold.improvement_over_baseline
            )
            
            # Add validation info if provided
            if validation_info:
                config.validation_sample_size = validation_info.get('sample_size', 0)
                if 'data_period' in validation_info:
                    config.validation_data_period = validation_info['data_period']
            
            # Add business context if provided
            if business_context:
                config.risk_tolerance_level = business_context.get('risk_tolerance', 'medium')
                config.false_alarm_cost_weight = business_context.get('false_alarm_cost_weight', 1.0)
                config.missed_event_cost_weight = business_context.get('missed_event_cost_weight', 1.0)
                config.performance_degradation_threshold = business_context.get('degradation_threshold', 0.9)
                config.reoptimization_frequency_days = business_context.get('reoptimization_days', 30)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threshold_config_{model_version}_{timestamp}.json"
            config_path = self.config_dir / filename
            
            # Save configuration
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2, default=str)
            
            # Update current configuration
            self.current_config = config
            self.config_history.append(config)
            
            # Create symlink to latest configuration
            latest_path = self.config_dir / f"latest_{model_version}.json"
            if latest_path.exists():
                latest_path.unlink()
            
            # Create relative symlink
            try:
                latest_path.symlink_to(filename)
            except OSError:
                # Fallback: copy file if symlink fails (Windows compatibility)
                import shutil
                shutil.copy2(config_path, latest_path)
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("threshold_config_save", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "threshold_config", "save", 1, True,
                f"Saved threshold config for {model_version} to {config_path}"
            )
            
            return str(config_path)
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("threshold_config_save", duration_ms, False)
            self.data_logger.log_data_collection("threshold_config", "save", 0, False, str(e))
            raise
    
    def load_threshold_configuration(self, 
                                   model_version: str,
                                   config_path: Optional[str] = None) -> Optional[ThresholdConfiguration]:
        """
        Load threshold configuration for a model version
        
        Args:
            model_version: Version of the model to load configuration for
            config_path: Optional specific path to configuration file
            
        Returns:
            ThresholdConfiguration object or None if not found
        """
        start_time = time.time()
        
        try:
            if config_path:
                # Load from specific path
                load_path = Path(config_path)
            else:
                # Load latest configuration for model version
                load_path = self.config_dir / f"latest_{model_version}.json"
            
            if not load_path.exists():
                self.data_logger.log_data_collection(
                    "threshold_config", "load", 0, False,
                    f"Configuration file not found: {load_path}"
                )
                return None
            
            # Load configuration
            with open(load_path, 'r') as f:
                config_data = json.load(f)
            
            # Convert to ThresholdConfiguration object
            config = ThresholdConfiguration.from_dict(config_data)
            
            # Update current configuration
            self.current_config = config
            
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("threshold_config_load", duration_ms, True)
            
            self.data_logger.log_data_collection(
                "threshold_config", "load", 1, True,
                f"Loaded threshold config for {model_version} from {load_path}"
            )
            
            return config
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_logger.log_inference_time("threshold_config_load", duration_ms, False)
            self.data_logger.log_data_collection("threshold_config", "load", 0, False, str(e))
            return None
    
    def get_threshold_for_model(self, model_version: str) -> Optional[float]:
        """
        Get the optimal threshold value for a specific model version
        
        Args:
            model_version: Version of the model
            
        Returns:
            Optimal threshold value or None if not found
        """
        try:
            config = self.load_threshold_configuration(model_version)
            if config:
                return config.optimal_threshold
            return None
        
        except Exception as e:
            self.data_logger.log_data_collection(
                "threshold_retrieval", model_version, 0, False, str(e)
            )
            return None
    
    def validate_threshold_configuration(self, config: ThresholdConfiguration) -> Dict[str, Any]:
        """
        Validate threshold configuration for completeness and consistency
        
        Args:
            config: ThresholdConfiguration to validate
            
        Returns:
            Validation results dictionary
        """
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'recommendations': []
            }
            
            # Check required fields
            if not config.model_version:
                validation_results['errors'].append("Model version is required")
                validation_results['is_valid'] = False
            
            if not (0.0 <= config.optimal_threshold <= 1.0):
                validation_results['errors'].append("Optimal threshold must be between 0.0 and 1.0")
                validation_results['is_valid'] = False
            
            # Check performance metrics
            if config.f1_score < 0.5:
                validation_results['warnings'].append("F1 score is below 0.5, consider reoptimization")
            
            if config.precision < 0.6:
                validation_results['warnings'].append("Low precision may cause excessive false alarms")
                validation_results['recommendations'].append("Consider raising threshold or improving model")
            
            if config.recall < 0.6:
                validation_results['warnings'].append("Low recall may miss critical risk events")
                validation_results['recommendations'].append("Consider lowering threshold or improving model")
            
            # Check confidence interval
            ci_width = config.confidence_interval[1] - config.confidence_interval[0]
            if ci_width > 0.2:
                validation_results['warnings'].append("Wide confidence interval indicates uncertainty")
                validation_results['recommendations'].append("Collect more validation data for better confidence")
            
            # Check age of configuration
            if config.created_timestamp:
                try:
                    created_date = datetime.fromisoformat(config.created_timestamp.replace('Z', '+00:00'))
                    age_days = (datetime.now() - created_date).days
                    
                    if age_days > config.reoptimization_frequency_days:
                        validation_results['warnings'].append(f"Configuration is {age_days} days old")
                        validation_results['recommendations'].append("Consider reoptimizing threshold")
                except ValueError:
                    validation_results['warnings'].append("Invalid timestamp format")
            
            # Business context validation
            if config.risk_tolerance_level not in ['low', 'medium', 'high']:
                validation_results['warnings'].append("Invalid risk tolerance level")
            
            return validation_results
        
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'recommendations': []
            }
    
    def list_available_configurations(self, model_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available threshold configurations
        
        Args:
            model_version: Optional filter by model version
            
        Returns:
            List of configuration summaries
        """
        try:
            configurations = []
            
            # Find all configuration files
            pattern = f"threshold_config_{model_version}_*.json" if model_version else "threshold_config_*.json"
            config_files = list(self.config_dir.glob(pattern))
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    summary = {
                        'filename': config_file.name,
                        'model_version': config_data.get('model_version', 'unknown'),
                        'optimal_threshold': config_data.get('optimal_threshold', 0.0),
                        'f1_score': config_data.get('f1_score', 0.0),
                        'created_timestamp': config_data.get('created_timestamp', ''),
                        'optimization_method': config_data.get('optimization_method', 'unknown'),
                        'file_path': str(config_file)
                    }
                    configurations.append(summary)
                
                except Exception as e:
                    self.data_logger.log_data_collection(
                        "config_listing", config_file.name, 0, False, str(e)
                    )
                    continue
            
            # Sort by creation timestamp (newest first)
            configurations.sort(key=lambda x: x['created_timestamp'], reverse=True)
            
            return configurations
        
        except Exception as e:
            self.data_logger.log_data_collection("config_listing", "all", 0, False, str(e))
            return []
    
    def update_threshold_configuration(self, 
                                     model_version: str,
                                     updates: Dict[str, Any]) -> Optional[str]:
        """
        Update existing threshold configuration
        
        Args:
            model_version: Model version to update
            updates: Dictionary of fields to update
            
        Returns:
            Path to updated configuration file or None if failed
        """
        try:
            # Load current configuration
            current_config = self.load_threshold_configuration(model_version)
            if not current_config:
                raise ValueError(f"No configuration found for model version {model_version}")
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
                else:
                    self.data_logger.log_data_collection(
                        "config_update", f"invalid_field_{key}", 0, False,
                        f"Invalid field {key} in update"
                    )
            
            # Update timestamp
            current_config.last_updated_timestamp = datetime.now().isoformat()
            
            # Save updated configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threshold_config_{model_version}_{timestamp}_updated.json"
            config_path = self.config_dir / filename
            
            with open(config_path, 'w') as f:
                json.dump(current_config.to_dict(), f, indent=2, default=str)
            
            # Update latest symlink
            latest_path = self.config_dir / f"latest_{model_version}.json"
            if latest_path.exists():
                latest_path.unlink()
            
            try:
                latest_path.symlink_to(filename)
            except OSError:
                import shutil
                shutil.copy2(config_path, latest_path)
            
            self.current_config = current_config
            
            self.data_logger.log_data_collection(
                "threshold_config", "update", 1, True,
                f"Updated threshold config for {model_version}"
            )
            
            return str(config_path)
        
        except Exception as e:
            self.data_logger.log_data_collection("threshold_config", "update", 0, False, str(e))
            return None
    
    def generate_threshold_recommendation_report(self, 
                                               model_version: str,
                                               include_history: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive threshold recommendation report
        
        Args:
            model_version: Model version to generate report for
            include_history: Whether to include configuration history
            
        Returns:
            Comprehensive recommendation report
        """
        try:
            config = self.load_threshold_configuration(model_version)
            if not config:
                return {'error': f'No configuration found for model version {model_version}'}
            
            # Validate current configuration
            validation_results = self.validate_threshold_configuration(config)
            
            # Generate report
            report = {
                'model_version': model_version,
                'current_configuration': {
                    'optimal_threshold': config.optimal_threshold,
                    'performance_metrics': {
                        'f1_score': config.f1_score,
                        'precision': config.precision,
                        'recall': config.recall,
                        'accuracy': config.accuracy,
                        'auc_score': config.auc_score
                    },
                    'confidence_metrics': {
                        'confidence_interval': config.confidence_interval,
                        'confidence_level': config.confidence_level
                    },
                    'optimization_details': {
                        'method': config.optimization_method,
                        'criterion': config.optimization_criterion,
                        'total_evaluations': config.total_evaluations
                    }
                },
                'validation_results': validation_results,
                'business_context': {
                    'risk_tolerance_level': config.risk_tolerance_level,
                    'false_alarm_cost_weight': config.false_alarm_cost_weight,
                    'missed_event_cost_weight': config.missed_event_cost_weight
                },
                'monitoring_settings': {
                    'performance_degradation_threshold': config.performance_degradation_threshold,
                    'reoptimization_frequency_days': config.reoptimization_frequency_days
                },
                'timestamps': {
                    'created': config.created_timestamp,
                    'last_updated': config.last_updated_timestamp
                },
                'recommendations': self._generate_actionable_recommendations(config, validation_results)
            }
            
            # Add baseline comparison if available
            if config.baseline_performance:
                report['baseline_comparison'] = {
                    'baseline_performance': config.baseline_performance,
                    'improvement_over_baseline': config.improvement_over_baseline
                }
            
            # Add configuration history if requested
            if include_history:
                history_configs = self.list_available_configurations(model_version)
                report['configuration_history'] = history_configs[:10]  # Last 10 configurations
            
            return report
        
        except Exception as e:
            return {'error': f'Failed to generate report: {str(e)}'}
    
    def _generate_actionable_recommendations(self, 
                                           config: ThresholdConfiguration,
                                           validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on configuration and validation"""
        recommendations = []
        
        # Add validation recommendations
        recommendations.extend(validation_results.get('recommendations', []))
        
        # Performance-based recommendations
        if config.f1_score < 0.7:
            recommendations.append("Consider model retraining to improve F1 score")
        
        if config.precision < 0.7 and config.recall > 0.8:
            recommendations.append("Consider raising threshold to reduce false alarms")
        elif config.recall < 0.7 and config.precision > 0.8:
            recommendations.append("Consider lowering threshold to catch more risk events")
        
        # Business context recommendations
        if config.risk_tolerance_level == 'low' and config.recall < 0.9:
            recommendations.append("Low risk tolerance requires higher recall - consider lowering threshold")
        elif config.risk_tolerance_level == 'high' and config.precision < 0.8:
            recommendations.append("High risk tolerance allows for current precision level")
        
        # Monitoring recommendations
        if config.performance_degradation_threshold > 0.95:
            recommendations.append("Consider more sensitive degradation threshold for earlier alerts")
        
        return recommendations
    
    def cleanup_old_configurations(self, 
                                 model_version: Optional[str] = None,
                                 keep_latest_n: int = 5,
                                 older_than_days: int = 90) -> int:
        """
        Clean up old threshold configuration files
        
        Args:
            model_version: Optional filter by model version
            keep_latest_n: Number of latest configurations to keep per model
            older_than_days: Delete configurations older than this many days
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            
            # Get all configurations
            all_configs = self.list_available_configurations(model_version)
            
            # Group by model version
            configs_by_model = {}
            for config in all_configs:
                model_ver = config['model_version']
                if model_ver not in configs_by_model:
                    configs_by_model[model_ver] = []
                configs_by_model[model_ver].append(config)
            
            # Clean up each model's configurations
            for model_ver, configs in configs_by_model.items():
                # Sort by creation timestamp (newest first)
                configs.sort(key=lambda x: x['created_timestamp'], reverse=True)
                
                # Keep latest N configurations
                configs_to_check = configs[keep_latest_n:]
                
                for config in configs_to_check:
                    try:
                        # Check if older than cutoff
                        created_date = datetime.fromisoformat(config['created_timestamp'].replace('Z', '+00:00'))
                        
                        if created_date < cutoff_date:
                            # Delete the file
                            config_path = Path(config['file_path'])
                            if config_path.exists():
                                config_path.unlink()
                                deleted_count += 1
                                
                                self.data_logger.log_data_collection(
                                    "config_cleanup", config['filename'], 1, True,
                                    f"Deleted old configuration file"
                                )
                    
                    except Exception as e:
                        self.data_logger.log_data_collection(
                            "config_cleanup", config['filename'], 0, False, str(e)
                        )
                        continue
            
            return deleted_count
        
        except Exception as e:
            self.data_logger.log_data_collection("config_cleanup", "all", 0, False, str(e))
            return 0