#!/usr/bin/env python3
"""
Live Risk Monitoring Service
Starts the AI Risk Oracle with live blockchain integration
"""

import asyncio
import sys
import signal
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from services.inference.risk_inference_service import RiskInferenceService
from services.blockchain.blockchain_integration import BlockchainIntegrationService
from config import get_config
from logging_config import get_data_logger

class LiveRiskMonitor:
    """Live risk monitoring service with blockchain integration"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_data_logger()
        
        # Services
        self.inference_service = None
        self.blockchain_service = None
        
        # Monitoring state
        self.is_running = False
        self.monitoring_interval = 30  # seconds
        self.assets_to_monitor = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD"]
        
        # Statistics
        self.total_assessments = 0
        self.blockchain_updates = 0
        self.alerts_triggered = 0
        self.start_time = None
    
    async def initialize_services(self):
        """Initialize inference and blockchain services"""
        print("🔧 Initializing AI Risk Oracle services...")
        
        # Initialize inference service
        self.inference_service = RiskInferenceService()
        
        # Load latest model
        if self.inference_service.load_latest_model():
            print(f"✅ Model loaded: {self.inference_service.model_version}")
        else:
            raise Exception("Failed to load ML model")
        
        # Initialize blockchain service
        self.blockchain_service = BlockchainIntegrationService()
        
        if await self.blockchain_service.initialize():
            print("✅ Blockchain integration initialized")
            await self.blockchain_service.start_monitoring()
        else:
            print("⚠️  Blockchain integration failed - running in monitoring-only mode")
        
        print("🚀 All services initialized successfully!")
    
    async def perform_risk_assessment(self):
        """Perform risk assessment for all monitored assets"""
        try:
            print(f"📊 Performing risk assessment at {datetime.now().strftime('%H:%M:%S')}")
            
            # Get multi-asset assessment
            assessments = await self.inference_service.get_multi_asset_assessment(
                self.assets_to_monitor
            )
            
            if not assessments:
                print("⚠️  No assessments available")
                return
            
            # Process each assessment
            high_risk_assets = []
            blockchain_updates = 0
            
            for asset, assessment in assessments.items():
                self.total_assessments += 1
                
                print(f"   {asset}: {assessment.combined_risk_score:.3f} ({assessment.risk_level})")
                
                # Check for high risk
                if assessment.combined_risk_score >= 0.7:
                    high_risk_assets.append(asset)
                
                # Process blockchain update if service is available
                if self.blockchain_service:
                    update = await self.blockchain_service.process_risk_assessment(assessment)
                    if update and update.transaction_result and update.transaction_result.success:
                        blockchain_updates += 1
                        print(f"   🔗 Blockchain updated for {asset}: {update.transaction_result.tx_hash}")
            
            self.blockchain_updates += blockchain_updates
            
            # Generate alerts for high risk assets
            if high_risk_assets:
                self.alerts_triggered += 1
                await self.generate_risk_alert(high_risk_assets, assessments)
            
            print(f"✅ Assessment complete - {len(assessments)} assets processed")
            
        except Exception as e:
            print(f"❌ Risk assessment failed: {e}")
            self.logger.log_data_collection(
                "risk_assessment", "error", 0, False, str(e)
            )
    
    async def generate_risk_alert(self, high_risk_assets: list, assessments: dict):
        """Generate and log risk alerts"""
        try:
            max_risk = max(assessments[asset].combined_risk_score for asset in high_risk_assets)
            
            alert = {
                "timestamp": datetime.now().isoformat(),
                "alert_level": "HIGH" if max_risk >= 0.8 else "MEDIUM",
                "triggered_assets": high_risk_assets,
                "max_risk_score": max_risk,
                "assessments": {
                    asset: {
                        "risk_score": assessments[asset].combined_risk_score,
                        "risk_level": assessments[asset].risk_level,
                        "confidence": assessments[asset].confidence
                    }
                    for asset in high_risk_assets
                }
            }
            
            print(f"🚨 RISK ALERT: {alert['alert_level']} risk detected!")
            print(f"   Assets: {', '.join(high_risk_assets)}")
            print(f"   Max Risk: {max_risk:.3f}")
            
            # Save alert to file
            alerts_file = Path("risk_alerts.jsonl")
            with open(alerts_file, "a") as f:
                f.write(json.dumps(alert) + "\n")
            
            # Log alert
            self.logger.log_data_collection(
                "risk_alert", f"level_{alert['alert_level']}", len(high_risk_assets), True,
                f"High risk detected: {', '.join(high_risk_assets)}"
            )
            
        except Exception as e:
            print(f"⚠️  Alert generation failed: {e}")
    
    async def print_status(self):
        """Print current monitoring status"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print("\n" + "=" * 60)
        print(f"📈 AI Risk Oracle Live Monitoring Status")
        print(f"   Uptime: {uptime/3600:.1f} hours")
        print(f"   Total Assessments: {self.total_assessments}")
        print(f"   Blockchain Updates: {self.blockchain_updates}")
        print(f"   Alerts Triggered: {self.alerts_triggered}")
        print(f"   Monitoring: {', '.join(self.assets_to_monitor)}")
        
        # Get blockchain status if available
        if self.blockchain_service:
            blockchain_status = await self.blockchain_service.get_blockchain_status()
            connection_status = blockchain_status.get("blockchain_connection", {})
            print(f"   Blockchain: {'Connected' if connection_status.get('connected') else 'Disconnected'}")
        
        print("=" * 60)
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        print(f"🔄 Starting live monitoring (interval: {self.monitoring_interval}s)")
        self.start_time = datetime.now()
        
        try:
            while self.is_running:
                # Perform risk assessment
                await self.perform_risk_assessment()
                
                # Print status every 10 cycles
                if self.total_assessments % 10 == 0:
                    await self.print_status()
                
                # Wait for next cycle
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            print("🛑 Monitoring loop cancelled")
        except Exception as e:
            print(f"❌ Monitoring loop failed: {e}")
            raise
    
    async def start(self):
        """Start the live monitoring service"""
        try:
            print("🚀 Starting AI Risk Oracle Live Monitoring")
            print("=" * 50)
            
            # Initialize services
            await self.initialize_services()
            
            # Start monitoring
            self.is_running = True
            
            print(f"📡 Monitoring {len(self.assets_to_monitor)} assets every {self.monitoring_interval} seconds")
            print("Press Ctrl+C to stop monitoring")
            print()
            
            # Run monitoring loop
            await self.monitoring_loop()
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring failed: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitoring service"""
        print("🛑 Stopping AI Risk Oracle monitoring...")
        
        self.is_running = False
        
        # Stop blockchain service
        if self.blockchain_service:
            await self.blockchain_service.stop_monitoring()
        
        # Clear inference service cache
        if self.inference_service:
            self.inference_service.clear_cache()
        
        # Print final statistics
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
            print(f"📊 Final Statistics:")
            print(f"   Total Runtime: {uptime/3600:.1f} hours")
            print(f"   Total Assessments: {self.total_assessments}")
            print(f"   Blockchain Updates: {self.blockchain_updates}")
            print(f"   Alerts Triggered: {self.alerts_triggered}")
        
        print("✅ Monitoring service stopped")

def setup_signal_handlers(monitor):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down...")
        monitor.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main function"""
    try:
        monitor = LiveRiskMonitor()
        setup_signal_handlers(monitor)
        
        await monitor.start()
        return 0
        
    except Exception as e:
        print(f"❌ Live monitoring failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)