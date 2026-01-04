"""
Enhanced Training Configuration Management
Centralized configuration loading and validation for enhanced training features
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class ExtendedTrainingConfig:
    """Configuration for extended training periods"""
    enabled: bool = True
    min_training_months: int = 12
    max_training_months: int = 24
    default_training_months: int = 18
    enable_progress_tracking: bool = True
    resumable_collection: bool = True
    data_quality_threshold: float = 0.8

@dataclass
class CrashEventsConfig:
    """Configuration for crash event handling"""
    enabled: bool = True
    config_file: str = "config/crash_events.json"
    include_in_training: bool = True
    weight_crash_samples: bool = True
    default_weight_multiplier: float = 2.0
    validate_crash_periods: bool = True
    require_minimum_crash_data: bool = True
    minimum_crash_data_points: int = 10

@dataclass
class ThresholdOptimizationConfig:
    """Configuration for threshold optimization"""
    enabled: bool = False
    optimization_method: str = "grid_search"
    threshold_values: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    optimization_metric: str = "f1_score"
    cross_validation_folds: int = 5
    confidence_interval: float = 0.95
    save_optimization_results: bool = True
    use_crash_events_for_optimization: bool = True

@dataclass
class FeatureEngineeringConfig:
    """Configuration for enhanced feature engineering"""
    cross_asset_correlation: Dict[str, Any] = field(default_factory=dict)
    volatility_regimes: Dict[str, Any] = field(default_factory=dict)
    momentum_indicators: Dict[str, Any] = field(default_factory=dict)
    volume_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestingConfig:
    """Configuration for enhanced backtesting"""
    enhanced_backtesting: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_strategy: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AutomatedRetrainingConfig:
    """Configuration for automated retraining"""
    enabled: bool = False
    schedule: str = "weekly"
    performance_monitoring: Dict[str, Any] = field(default_factory=dict)
    data_freshness: Dict[str, Any] = field(default_factory=dict)
    incremental_training: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiagnosticsConfig:
    """Configuration for training diagnostics"""
    comprehensive_diagnostics: Dict[str, Any] = field(default_factory=dict)
    shap_analysis: Dict[str, Any] = field(default_factory=dict)
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    data_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnhancedTrainingConfiguration:
    """Complete enhanced training configuration"""
    extended_training: ExtendedTrainingConfig = field(default_factory=ExtendedTrainingConfig)
    crash_events: CrashEventsConfig = field(default_factory=CrashEventsConfig)
    threshold_optimization: ThresholdOptimizationConfig = field(default_factory=ThresholdOptimizationConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    automated_retraining: AutomatedRetrainingConfig = field(default_factory=AutomatedRetrainingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    output_configuration: Dict[str, Any] = field(default_factory=dict)
    resource_management: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)

class EnhancedTrainingConfigManager:
    """Manager for enhanced training configuration loading and validation"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.base_config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Default configuration files
        self.config_files = {
            "enhanced_training": self.config_dir / "enhanced_training.json",
            "crash_events": self.config_dir / "crash_events.json",
            "feature_engineering": self.config_dir / "feature_engineering.json",
            "backtesting": self.config_dir / "backtesting.json"
        }
    
    def load_enhanced_training_config(self, 
                                    config_file: Optional[str] = None,
                                    override_config: Optional[Dict[str, Any]] = None) -> EnhancedTrainingConfiguration:
        """
        Load enhanced training configuration from file and overrides
        
        Args:
            config_file: Path to enhanced training configuration file
            override_config: Dictionary of configuration overrides
            
        Returns:
            EnhancedTrainingConfiguration object
        """
        try:
            # Load base enhanced training configuration
            if config_file:
                config_path = Path(config_file)
            else:
                config_path = self.config_files["enhanced_training"]
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.logger.info(f"Loaded enhanced training configuration from {config_path}")
            else:
                self.logger.warning(f"Enhanced training config file not found: {config_path}, using defaults")
                config_data = {}
            
            # Apply overrides
            if override_config:
                config_data = self._merge_configs(config_data, override_config)
                self.logger.info("Applied configuration overrides")
            
            # Load additional configuration files
            additional_configs = self._load_additional_configs()
            config_data = self._merge_configs(config_data, additional_configs)
            
            # Create configuration object
            enhanced_config = self._create_enhanced_config(config_data)
            
            # Validate configuration
            self._validate_enhanced_config(enhanced_config)
            
            return enhanced_config
            
        except Exception as e:
            self.logger.error(f"Error loading enhanced training configuration: {str(e)}")
            # Return default configuration on error
            return EnhancedTrainingConfiguration()
    
    def _load_additional_configs(self) -> Dict[str, Any]:
        """Load additional configuration files"""
        additional_configs = {}
        
        # Load crash events configuration
        crash_events_path = self.config_files["crash_events"]
        if crash_events_path.exists():
            try:
                with open(crash_events_path, 'r') as f:
                    crash_events_data = json.load(f)
                additional_configs["crash_events_data"] = crash_events_data
                self.logger.info(f"Loaded crash events configuration from {crash_events_path}")
            except Exception as e:
                self.logger.warning(f"Error loading crash events config: {str(e)}")
        
        # Load feature engineering configuration
        feature_eng_path = self.config_files["feature_engineering"]
        if feature_eng_path.exists():
            try:
                with open(feature_eng_path, 'r') as f:
                    feature_eng_data = json.load(f)
                additional_configs["feature_engineering"] = feature_eng_data
                self.logger.info(f"Loaded feature engineering configuration from {feature_eng_path}")
            except Exception as e:
                self.logger.warning(f"Error loading feature engineering config: {str(e)}")
        
        # Load backtesting configuration
        backtesting_path = self.config_files["backtesting"]
        if backtesting_path.exists():
            try:
                with open(backtesting_path, 'r') as f:
                    backtesting_data = json.load(f)
                additional_configs["backtesting"] = backtesting_data
                self.logger.info(f"Loaded backtesting configuration from {backtesting_path}")
            except Exception as e:
                self.logger.warning(f"Error loading backtesting config: {str(e)}")
        
        return additional_configs
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _create_enhanced_config(self, config_data: Dict[str, Any]) -> EnhancedTrainingConfiguration:
        """Create EnhancedTrainingConfiguration from configuration data"""
        try:
            # Extract configuration sections
            extended_training_data = config_data.get("extended_training", {})
            crash_events_data = config_data.get("crash_events", {})
            threshold_opt_data = config_data.get("threshold_optimization", {})
            feature_eng_data = config_data.get("feature_engineering", {})
            backtesting_data = config_data.get("backtesting", {})
            auto_retrain_data = config_data.get("automated_retraining", {})
            diagnostics_data = config_data.get("diagnostics", {})
            
            # Create configuration objects
            extended_training = ExtendedTrainingConfig(**extended_training_data)
            crash_events = CrashEventsConfig(**crash_events_data)
            threshold_optimization = ThresholdOptimizationConfig(**threshold_opt_data)
            
            # Handle feature engineering config more carefully
            feature_engineering = FeatureEngineeringConfig()
            if feature_eng_data:
                for key, value in feature_eng_data.items():
                    if hasattr(feature_engineering, key):
                        setattr(feature_engineering, key, value)
            
            # Handle backtesting config more carefully  
            backtesting = BacktestingConfig()
            if backtesting_data:
                for key, value in backtesting_data.items():
                    if hasattr(backtesting, key):
                        setattr(backtesting, key, value)
            
            automated_retraining = AutomatedRetrainingConfig(**auto_retrain_data)
            
            # Handle diagnostics config more carefully
            diagnostics = DiagnosticsConfig()
            if diagnostics_data:
                for key, value in diagnostics_data.items():
                    if hasattr(diagnostics, key):
                        setattr(diagnostics, key, value)
            
            return EnhancedTrainingConfiguration(
                extended_training=extended_training,
                crash_events=crash_events,
                threshold_optimization=threshold_optimization,
                feature_engineering=feature_engineering,
                backtesting=backtesting,
                automated_retraining=automated_retraining,
                diagnostics=diagnostics,
                output_configuration=config_data.get("output_configuration", {}),
                resource_management=config_data.get("resource_management", {}),
                validation=config_data.get("validation", {})
            )
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced configuration object: {str(e)}")
            return EnhancedTrainingConfiguration()
    
    def _validate_enhanced_config(self, config: EnhancedTrainingConfiguration) -> None:
        """Validate enhanced training configuration"""
        errors = []
        
        # Validate extended training configuration
        if config.extended_training.min_training_months < 1:
            errors.append("Minimum training months must be at least 1")
        
        if config.extended_training.max_training_months < config.extended_training.min_training_months:
            errors.append("Maximum training months must be >= minimum training months")
        
        if not 0.0 <= config.extended_training.data_quality_threshold <= 1.0:
            errors.append("Data quality threshold must be between 0.0 and 1.0")
        
        # Validate crash events configuration
        if config.crash_events.default_weight_multiplier <= 0:
            errors.append("Default weight multiplier must be positive")
        
        if config.crash_events.minimum_crash_data_points < 1:
            errors.append("Minimum crash data points must be at least 1")
        
        # Validate threshold optimization configuration
        if config.threshold_optimization.enabled:
            if not config.threshold_optimization.threshold_values:
                errors.append("Threshold values list cannot be empty when optimization is enabled")
            
            for threshold in config.threshold_optimization.threshold_values:
                if not 0.0 <= threshold <= 1.0:
                    errors.append(f"Threshold value {threshold} must be between 0.0 and 1.0")
            
            if config.threshold_optimization.cross_validation_folds < 2:
                errors.append("Cross validation folds must be at least 2")
            
            if not 0.0 <= config.threshold_optimization.confidence_interval <= 1.0:
                errors.append("Confidence interval must be between 0.0 and 1.0")
        
        # Validate file paths
        if config.crash_events.enabled and config.crash_events.config_file:
            crash_events_path = Path(config.crash_events.config_file)
            if not crash_events_path.exists():
                errors.append(f"Crash events config file not found: {crash_events_path}")
        
        if errors:
            error_message = "Enhanced training configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            self.logger.error(error_message)
            raise ValueError(error_message)
        
        self.logger.info("Enhanced training configuration validation passed")
    
    def save_enhanced_config(self, config: EnhancedTrainingConfiguration, 
                           output_path: str) -> None:
        """Save enhanced training configuration to file"""
        try:
            config_dict = self._config_to_dict(config)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Enhanced training configuration saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced training configuration: {str(e)}")
            raise
    
    def _config_to_dict(self, config: EnhancedTrainingConfiguration) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        return {
            "extended_training": {
                "enabled": config.extended_training.enabled,
                "min_training_months": config.extended_training.min_training_months,
                "max_training_months": config.extended_training.max_training_months,
                "default_training_months": config.extended_training.default_training_months,
                "enable_progress_tracking": config.extended_training.enable_progress_tracking,
                "resumable_collection": config.extended_training.resumable_collection,
                "data_quality_threshold": config.extended_training.data_quality_threshold
            },
            "crash_events": {
                "enabled": config.crash_events.enabled,
                "config_file": config.crash_events.config_file,
                "include_in_training": config.crash_events.include_in_training,
                "weight_crash_samples": config.crash_events.weight_crash_samples,
                "default_weight_multiplier": config.crash_events.default_weight_multiplier,
                "validate_crash_periods": config.crash_events.validate_crash_periods,
                "require_minimum_crash_data": config.crash_events.require_minimum_crash_data,
                "minimum_crash_data_points": config.crash_events.minimum_crash_data_points
            },
            "threshold_optimization": {
                "enabled": config.threshold_optimization.enabled,
                "optimization_method": config.threshold_optimization.optimization_method,
                "threshold_values": config.threshold_optimization.threshold_values,
                "optimization_metric": config.threshold_optimization.optimization_metric,
                "cross_validation_folds": config.threshold_optimization.cross_validation_folds,
                "confidence_interval": config.threshold_optimization.confidence_interval,
                "save_optimization_results": config.threshold_optimization.save_optimization_results,
                "use_crash_events_for_optimization": config.threshold_optimization.use_crash_events_for_optimization
            },
            "feature_engineering": config.feature_engineering.__dict__,
            "backtesting": config.backtesting.__dict__,
            "automated_retraining": {
                "enabled": config.automated_retraining.enabled,
                "schedule": config.automated_retraining.schedule,
                "performance_monitoring": config.automated_retraining.performance_monitoring,
                "data_freshness": config.automated_retraining.data_freshness,
                "incremental_training": config.automated_retraining.incremental_training
            },
            "diagnostics": config.diagnostics.__dict__,
            "output_configuration": config.output_configuration,
            "resource_management": config.resource_management,
            "validation": config.validation
        }
    
    def get_crash_events_from_config(self, config: EnhancedTrainingConfiguration) -> Dict[str, Any]:
        """Load crash events data from configuration"""
        try:
            if not config.crash_events.enabled:
                return {}
            
            crash_events_path = Path(config.crash_events.config_file)
            if not crash_events_path.exists():
                self.logger.warning(f"Crash events config file not found: {crash_events_path}")
                return {}
            
            with open(crash_events_path, 'r') as f:
                crash_events_data = json.load(f)
            
            return crash_events_data
            
        except Exception as e:
            self.logger.error(f"Error loading crash events data: {str(e)}")
            return {}
    
    def create_training_summary(self, config: EnhancedTrainingConfiguration) -> str:
        """Create a human-readable summary of the training configuration"""
        summary_lines = []
        summary_lines.append("=== Enhanced Training Configuration Summary ===")
        summary_lines.append("")
        
        # Extended training
        summary_lines.append("Extended Training:")
        summary_lines.append(f"  Enabled: {config.extended_training.enabled}")
        if config.extended_training.enabled:
            summary_lines.append(f"  Training Period: {config.extended_training.min_training_months}-{config.extended_training.max_training_months} months")
            summary_lines.append(f"  Progress Tracking: {config.extended_training.enable_progress_tracking}")
        summary_lines.append("")
        
        # Crash events
        summary_lines.append("Crash Events:")
        summary_lines.append(f"  Enabled: {config.crash_events.enabled}")
        if config.crash_events.enabled:
            summary_lines.append(f"  Include in Training: {config.crash_events.include_in_training}")
            summary_lines.append(f"  Weight Multiplier: {config.crash_events.default_weight_multiplier}")
        summary_lines.append("")
        
        # Threshold optimization
        summary_lines.append("Threshold Optimization:")
        summary_lines.append(f"  Enabled: {config.threshold_optimization.enabled}")
        if config.threshold_optimization.enabled:
            summary_lines.append(f"  Method: {config.threshold_optimization.optimization_method}")
            summary_lines.append(f"  Metric: {config.threshold_optimization.optimization_metric}")
            summary_lines.append(f"  Threshold Values: {config.threshold_optimization.threshold_values}")
        summary_lines.append("")
        
        # Feature engineering
        summary_lines.append("Feature Engineering:")
        summary_lines.append(f"  Cross-Asset Correlation: {config.feature_engineering.cross_asset_correlation.get('enabled', False)}")
        summary_lines.append(f"  Volatility Regimes: {config.feature_engineering.volatility_regimes.get('enabled', False)}")
        summary_lines.append(f"  Momentum Indicators: {config.feature_engineering.momentum_indicators.get('enabled', False)}")
        summary_lines.append(f"  Volume Analysis: {config.feature_engineering.volume_analysis.get('enabled', False)}")
        summary_lines.append("")
        
        # Backtesting
        summary_lines.append("Enhanced Backtesting:")
        enhanced_bt = config.backtesting.enhanced_backtesting
        summary_lines.append(f"  Enabled: {enhanced_bt.get('enabled', False)}")
        if enhanced_bt.get('enabled', False):
            summary_lines.append(f"  Crash Event Analysis: {enhanced_bt.get('per_crash_event_analysis', False)}")
            summary_lines.append(f"  Multi-Horizon Evaluation: {enhanced_bt.get('multi_horizon_evaluation', False)}")
            summary_lines.append(f"  Baseline Comparisons: {enhanced_bt.get('baseline_comparisons', False)}")
        summary_lines.append("")
        
        # Automated retraining
        summary_lines.append("Automated Retraining:")
        summary_lines.append(f"  Enabled: {config.automated_retraining.enabled}")
        if config.automated_retraining.enabled:
            summary_lines.append(f"  Schedule: {config.automated_retraining.schedule}")
        summary_lines.append("")
        
        # Diagnostics
        summary_lines.append("Diagnostics:")
        comp_diag = config.diagnostics.comprehensive_diagnostics
        summary_lines.append(f"  Comprehensive Diagnostics: {comp_diag.get('enabled', False)}")
        summary_lines.append(f"  SHAP Analysis: {config.diagnostics.shap_analysis.get('enabled', False)}")
        
        return "\n".join(summary_lines)

# Global configuration manager instance
_config_manager = None

def get_enhanced_training_config_manager() -> EnhancedTrainingConfigManager:
    """Get the global enhanced training configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnhancedTrainingConfigManager()
    return _config_manager