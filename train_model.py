#!/usr/bin/env python3
"""
AI Risk Oracle Model Training Script
Main entry point for training the risk prediction model
"""

import asyncio
import argparse
import json
from datetime import datetime
from pathlib import Path

from config import get_config
from logging_config import get_data_logger
from services.timeseries.training_pipeline import TrainingPipeline

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train AI Risk Oracle model")
    parser.add_argument("--training-months", type=int, default=24,
                       help="Number of months of training data (default: 24)")
    parser.add_argument("--validation-months", type=int, default=6,
                       help="Number of months of validation data (default: 6)")
    parser.add_argument("--model-dir", type=str, default="models/trained",
                       help="Directory to save trained models (default: models/trained)")
    parser.add_argument("--model-params", type=str, default=None,
                       help="JSON file with model parameters")
    parser.add_argument("--quick-update", type=str, default=None,
                       help="Path to existing model for quick update")
    parser.add_argument("--update-days", type=int, default=7,
                       help="Days of recent data for quick update (default: 7)")
    
    # Enhanced training options
    parser.add_argument("--extended-training", action="store_true",
                       help="Enable extended training with 12-24 month periods")
    parser.add_argument("--min-training-months", type=int, default=12,
                       help="Minimum training period in months (default: 12)")
    parser.add_argument("--max-training-months", type=int, default=24,
                       help="Maximum training period in months (default: 24)")
    
    # Crash event options
    parser.add_argument("--include-crash-events", action="store_true", default=True,
                       help="Include crash event detection and labeling (default: True)")
    parser.add_argument("--crash-events-config", type=str, default=None,
                       help="JSON file with crash events configuration")
    parser.add_argument("--crash-event-weight", type=float, default=2.0,
                       help="Weight multiplier for crash event samples (default: 2.0)")
    
    # Threshold optimization options
    parser.add_argument("--optimize-threshold", action="store_true",
                       help="Enable automatic threshold optimization")
    parser.add_argument("--threshold-range", type=str, default="0.5,0.6,0.7,0.8,0.85,0.9",
                       help="Comma-separated threshold values to test (default: 0.5,0.6,0.7,0.8,0.85,0.9)")
    parser.add_argument("--threshold-metric", type=str, default="f1_score",
                       choices=["f1_score", "precision", "recall", "accuracy"],
                       help="Metric to optimize threshold for (default: f1_score)")
    
    # Enhanced feature engineering options
    parser.add_argument("--enable-correlation-features", action="store_true", default=True,
                       help="Enable cross-asset correlation features (default: True)")
    parser.add_argument("--correlation-windows", type=str, default="7,30,90",
                       help="Comma-separated correlation window sizes in days (default: 7,30,90)")
    parser.add_argument("--enable-volatility-regimes", action="store_true", default=True,
                       help="Enable volatility regime classification (default: True)")
    parser.add_argument("--enable-momentum-indicators", action="store_true", default=True,
                       help="Enable multi-timeframe momentum indicators (default: True)")
    parser.add_argument("--momentum-timeframes", type=str, default="1h,4h,24h",
                       help="Comma-separated momentum timeframes (default: 1h,4h,24h)")
    
    # Enhanced backtesting options
    parser.add_argument("--enable-enhanced-backtesting", action="store_true",
                       help="Enable enhanced backtesting with crash event analysis")
    parser.add_argument("--multi-horizon-evaluation", action="store_true",
                       help="Enable multi-horizon accuracy evaluation (1h, 4h, 24h)")
    parser.add_argument("--baseline-comparison", action="store_true",
                       help="Enable baseline model comparisons")
    
    # Automated retraining options
    parser.add_argument("--enable-auto-retraining", action="store_true",
                       help="Enable automated retraining system")
    parser.add_argument("--retraining-schedule", type=str, default="weekly",
                       choices=["daily", "weekly", "monthly"],
                       help="Automated retraining schedule (default: weekly)")
    parser.add_argument("--performance-threshold", type=float, default=0.05,
                       help="Performance degradation threshold for retraining trigger (default: 0.05)")
    
    # Enhanced diagnostics options
    parser.add_argument("--enable-diagnostics", action="store_true", default=True,
                       help="Enable comprehensive training diagnostics (default: True)")
    parser.add_argument("--enable-shap-analysis", action="store_true",
                       help="Enable SHAP feature importance analysis")
    parser.add_argument("--generate-learning-curves", action="store_true",
                       help="Generate learning curves for training progress analysis")
    parser.add_argument("--detect-overfitting", action="store_true", default=True,
                       help="Enable overfitting/underfitting detection (default: True)")
    
    # Configuration file option
    parser.add_argument("--enhanced-config", type=str, default=None,
                       help="JSON file with enhanced training configuration")
    
    args = parser.parse_args()
    
    # Initialize configuration and logging
    config = get_config()
    config.create_directories()
    data_logger = get_data_logger()
    
    # Load enhanced configuration if provided
    enhanced_config = {}
    if args.enhanced_config:
        with open(args.enhanced_config, 'r') as f:
            enhanced_config = json.load(f)
    
    # Load model parameters if provided
    model_params = None
    if args.model_params:
        with open(args.model_params, 'r') as f:
            model_params = json.load(f)
    
    # Load crash events configuration if provided
    crash_events_config = {}
    if args.crash_events_config:
        with open(args.crash_events_config, 'r') as f:
            crash_events_config = json.load(f)
    
    # Parse threshold range
    threshold_values = [float(x.strip()) for x in args.threshold_range.split(',')]
    
    # Parse correlation windows
    correlation_windows = [int(x.strip()) for x in args.correlation_windows.split(',')]
    
    # Parse momentum timeframes
    momentum_timeframes = [x.strip() for x in args.momentum_timeframes.split(',')]
    
    # Validate training period for extended training
    if args.extended_training:
        if args.training_months < args.min_training_months:
            print(f"Warning: Training period {args.training_months} months is below minimum {args.min_training_months} months for extended training")
            args.training_months = args.min_training_months
        elif args.training_months > args.max_training_months:
            print(f"Warning: Training period {args.training_months} months exceeds maximum {args.max_training_months} months")
            args.training_months = args.max_training_months
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(model_save_dir=args.model_dir)
    
    # Prepare enhanced training configuration
    enhanced_training_config = {
        "extended_training_enabled": args.extended_training,
        "include_crash_events": args.include_crash_events,
        "crash_events_config": crash_events_config,
        "crash_event_weight": args.crash_event_weight,
        "optimize_threshold": args.optimize_threshold,
        "threshold_values": threshold_values,
        "threshold_metric": args.threshold_metric,
        "enable_correlation_features": args.enable_correlation_features,
        "correlation_windows": correlation_windows,
        "enable_volatility_regimes": args.enable_volatility_regimes,
        "enable_momentum_indicators": args.enable_momentum_indicators,
        "momentum_timeframes": momentum_timeframes,
        "enable_enhanced_backtesting": args.enable_enhanced_backtesting,
        "multi_horizon_evaluation": args.multi_horizon_evaluation,
        "baseline_comparison": args.baseline_comparison,
        "enable_auto_retraining": args.enable_auto_retraining,
        "retraining_schedule": args.retraining_schedule,
        "performance_threshold": args.performance_threshold,
        "enable_diagnostics": args.enable_diagnostics,
        "enable_shap_analysis": args.enable_shap_analysis,
        "generate_learning_curves": args.generate_learning_curves,
        "detect_overfitting": args.detect_overfitting
    }
    
    # Merge with file-based enhanced config
    enhanced_training_config.update(enhanced_config)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(model_save_dir=args.model_dir)
    
    try:
        if args.quick_update:
            # Quick model update
            print(f"Performing quick model update with {args.update_days} days of recent data...")
            result = await pipeline.quick_model_update(
                args.quick_update, args.update_days
            )
            
            print(f"Model updated successfully!")
            print(f"Update metrics: {result['update_metrics']}")
            print(f"Samples used: {result['samples_used']}")
            
        else:
            # Determine which training pipeline to use
            use_enhanced_training = (
                args.extended_training or 
                args.optimize_threshold or 
                args.enable_enhanced_backtesting or 
                args.enable_diagnostics or
                args.enable_shap_analysis or
                any([
                    enhanced_training_config.get("extended_training_enabled"),
                    enhanced_training_config.get("optimize_threshold"),
                    enhanced_training_config.get("enable_enhanced_backtesting"),
                    enhanced_training_config.get("enable_diagnostics"),
                    enhanced_training_config.get("enable_shap_analysis")
                ])
            )
            
            if use_enhanced_training:
                # Enhanced training pipeline
                print(f"Starting enhanced training pipeline...")
                print(f"Training period: {args.training_months} months")
                print(f"Validation period: {args.validation_months} months")
                print(f"Model save directory: {args.model_dir}")
                print(f"Enhanced features enabled:")
                print(f"  - Extended training: {args.extended_training}")
                print(f"  - Crash events: {args.include_crash_events}")
                print(f"  - Threshold optimization: {args.optimize_threshold}")
                print(f"  - Enhanced backtesting: {args.enable_enhanced_backtesting}")
                print(f"  - Comprehensive diagnostics: {args.enable_diagnostics}")
                print(f"  - SHAP analysis: {args.enable_shap_analysis}")
                
                # Check if enhanced training method exists
                if hasattr(pipeline, 'run_enhanced_training_pipeline'):
                    # Use the new comprehensive enhanced training pipeline
                    result = await pipeline.run_enhanced_training_pipeline(
                        enhanced_config=None,  # Will load from config files
                        training_period_months=args.training_months,
                        validation_period_months=args.validation_months,
                        model_params=model_params
                    )
                    
                    # Print enhanced results
                    print("\n" + "="*60)
                    print("ENHANCED TRAINING COMPLETED SUCCESSFULLY")
                    print("="*60)
                    
                    print(f"\nModel Version: {result['model'].model_version}")
                    print(f"Enhanced metadata saved to: {result['enhanced_metadata_path']}")
                    
                    if 'enhanced_report_path' in result:
                        print(f"Enhanced report saved to: {result['enhanced_report_path']}")
                    
                    if 'diagnostic_results' in result and result['diagnostic_results']:
                        print(f"Diagnostic report available: {result['diagnostic_results'].get('report_path', 'N/A')}")
                    
                    # Print data quality information
                    if 'training_data_quality' in result and result['training_data_quality']:
                        quality_report = result['training_data_quality']
                        if hasattr(quality_report, 'quality_score'):
                            print(f"\nData Quality Score: {quality_report.quality_score:.2f}/1.0")
                            print(f"Data Coverage: {quality_report.coverage_percentage:.1%}")
                    
                    # Print crash events information
                    if 'crash_events_metadata' in result and result['crash_events_metadata']:
                        crash_events = result['crash_events_metadata']
                        print(f"\nCrash Events Included: {len(crash_events)}")
                        for event_name, event_info in crash_events.items():
                            print(f"  - {event_name}: {event_info.get('description', 'N/A')}")
                    
                    # Print optimal threshold information
                    if 'optimal_threshold' in result and result['optimal_threshold']:
                        threshold = result['optimal_threshold']
                        print(f"\nOptimal Threshold: {threshold.value:.3f} (F1: {threshold.f1_score:.3f})")
                    
                    # Print enhanced components status
                    if 'enhanced_components_status' in result:
                        print(f"\nEnhanced Components Status:")
                        for component, status in result['enhanced_components_status'].items():
                            status_icon = "✅" if status["available"] else "❌"
                            print(f"  {status_icon} {component}: {status['status']}")
                    
                    # Print comprehensive recommendations
                    if 'comprehensive_recommendations' in result:
                        print(f"\nRecommendations:")
                        for rec in result['comprehensive_recommendations']:
                            print(f"  • {rec}")
                    
                elif hasattr(pipeline, 'run_enhanced_training_with_diagnostics'):
                    # Fallback to existing enhanced training method
                    result = await pipeline.run_enhanced_training_with_diagnostics(
                        training_period_months=args.training_months,
                        validation_period_months=args.validation_months,
                        model_params=model_params,
                        enable_comprehensive_diagnostics=args.enable_diagnostics,
                        enable_shap_analysis=args.enable_shap_analysis
                    )
                    
                    # Print enhanced results
                    print("\n" + "="*60)
                    print("ENHANCED TRAINING COMPLETED SUCCESSFULLY")
                    print("="*60)
                    
                    print(f"\nModel Version: {result['model'].model_version}")
                    print(f"Enhanced metadata saved to: {result['enhanced_metadata_path']}")
                    
                    if 'diagnostic_results' in result and result['diagnostic_results']:
                        print(f"Diagnostic report available: {result['diagnostic_results'].get('report_path', 'N/A')}")
                    
                    # Print data quality information
                    if 'training_data_quality' in result and result['training_data_quality']:
                        quality_report = result['training_data_quality']
                        if hasattr(quality_report, 'quality_score'):
                            print(f"\nData Quality Score: {quality_report.quality_score:.2f}/1.0")
                            print(f"Data Coverage: {quality_report.coverage_percentage:.1%}")
                    
                    # Print crash events information
                    if 'crash_events_metadata' in result and result['crash_events_metadata']:
                        crash_events = result['crash_events_metadata']
                        print(f"\nCrash Events Included: {len(crash_events)}")
                        for event_name, event_info in crash_events.items():
                            print(f"  - {event_name}: {event_info['description']}")
                    
                    # Print comprehensive recommendations
                    if 'comprehensive_recommendations' in result:
                        print(f"\nRecommendations:")
                        for rec in result['comprehensive_recommendations']:
                            print(f"  • {rec}")
                    
                else:
                    # Fallback to standard training with enhanced configuration
                    print("Enhanced training method not available, using standard training with enhanced features...")
                    result = await pipeline.run_complete_training_pipeline(
                        training_period_months=args.training_months,
                        validation_period_months=args.validation_months,
                        model_params=model_params,
                        enable_enhanced_features=True
                    )
                    
                    # Print standard results with enhanced features
                    print("\n" + "="*60)
                    print("TRAINING COMPLETED SUCCESSFULLY (Enhanced Features Enabled)")
                    print("="*60)
                    
                    print(f"\nModel Version: {result['model'].model_version}")
                    print(f"Metadata saved to: {result['metadata_path']}")
                    
                    # Print enhanced components status
                    if 'enhanced_components_status' in result:
                        print(f"\nEnhanced Components Status:")
                        for component, status in result['enhanced_components_status'].items():
                            status_icon = "✅" if status["available"] else "❌"
                            print(f"  {status_icon} {component}: {status['status']}")
            
            else:
                # Standard training pipeline
                print(f"Starting standard training pipeline...")
                print(f"Training period: {args.training_months} months")
                print(f"Validation period: {args.validation_months} months")
                print(f"Model save directory: {args.model_dir}")
                
                result = await pipeline.run_complete_training_pipeline(
                    training_period_months=args.training_months,
                    validation_period_months=args.validation_months,
                    model_params=model_params,
                    enable_enhanced_features=False
                )
                
                # Print results
                print("\n" + "="*60)
                print("TRAINING COMPLETED SUCCESSFULLY")
                print("="*60)
                
                print(f"\nModel Version: {result['model'].model_version}")
                print(f"Metadata saved to: {result['metadata_path']}")
            
            # Print common training metrics
            print(f"\nTraining Metrics:")
            for metric, value in result['training_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nBacktest Results:")
            backtest = result['backtest_result']
            print(f"  Accuracy: {backtest.accuracy:.4f}")
            print(f"  Precision: {backtest.precision:.4f}")
            print(f"  Recall: {backtest.recall:.4f}")
            print(f"  F1 Score: {backtest.f1_score:.4f}")
            if backtest.auc_score:
                print(f"  AUC Score: {backtest.auc_score:.4f}")
            print(f"  Avg Latency: {backtest.average_prediction_latency_ms:.2f}ms")
            
            print(f"\nCross-Validation Summary:")
            cv_summary = result['performance_summary']
            print(f"  Avg Accuracy: {cv_summary.get('avg_accuracy', 0):.4f}")
            print(f"  Avg Precision: {cv_summary.get('avg_precision', 0):.4f}")
            print(f"  Avg Recall: {cv_summary.get('avg_recall', 0):.4f}")
            print(f"  Avg F1 Score: {cv_summary.get('avg_f1_score', 0):.4f}")
            print(f"  Total Predictions: {cv_summary.get('total_predictions', 0)}")
            
            print(f"\nTop Feature Importance:")
            feature_importance = result['model'].feature_importance
            if feature_importance:
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                for feature, importance in sorted_features:
                    print(f"  {feature}: {importance:.4f}")
            
            print(f"\nTraining Data Statistics:")
            # Handle both standard and enhanced metadata formats
            metadata = result.get('enhanced_metadata', result.get('metadata', {}))
            stats = metadata.get('training_data_stats', {})
            print(f"  Training Samples: {stats.get('training_samples', 'N/A')}")
            print(f"  Validation Samples: {stats.get('validation_samples', 'N/A')}")
            print(f"  Assets Covered: {', '.join(stats.get('assets_covered', []))}")
            
            # Save enhanced training configuration for future reference
            if enhanced_training_config:
                config_path = Path(args.model_dir) / f"enhanced_config_{result['model'].model_version}.json"
                with open(config_path, 'w') as f:
                    json.dump(enhanced_training_config, f, indent=2)
                print(f"\nEnhanced training configuration saved to: {config_path}")
        
        data_logger.log_data_collection("model_training", "complete", 1, True)
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        data_logger.log_data_collection("model_training", "complete", 0, False, str(e))
        raise

if __name__ == "__main__":
    asyncio.run(main())