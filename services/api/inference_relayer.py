"""
FastAPI Inference Relayer Service
Production-ready REST API for AI Risk Oracle with health checks and real-time risk scoring
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config import get_config
from logging_config import get_data_logger, get_performance_logger
from services.inference.risk_inference_service import RiskInferenceService, RiskAssessment
from services.blockchain.blockchain_integration import BlockchainIntegrationService

# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]
    model_info: Dict[str, Any]

class RiskScoreResponse(BaseModel):
    asset_symbol: str
    timestamp: str
    combined_risk_score: float
    risk_level: str
    ml_risk_score: float
    sentiment_risk_score: float
    confidence: float
    market_regime: str
    correlation_signals: List[str]
    inference_latency_ms: float
    data_freshness_seconds: int
    reasoning: List[str]

class MultiAssetRiskResponse(BaseModel):
    timestamp: str
    assessments: Dict[str, RiskScoreResponse]
    summary: Dict[str, Any]
    total_latency_ms: float

class RiskAlertResponse(BaseModel):
    alert_level: str
    triggered_assets: List[str]
    max_risk_score: float
    timestamp: str
    recommended_action: str

class StatusResponse(BaseModel):
    service: str
    status: str
    current_time: str
    model_version: str
    model_loaded_at: Optional[str]
    cache_size: int
    performance_metrics: Dict[str, float]

# Global service instances
inference_service: Optional[RiskInferenceService] = None
blockchain_service: Optional[BlockchainIntegrationService] = None
service_start_time: datetime = datetime.now()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global inference_service, blockchain_service
    
    # Startup
    print("🚀 Starting AI Risk Oracle Inference Relayer...")
    
    # Initialize the inference service
    inference_service = RiskInferenceService()
    
    # Load the latest model
    if inference_service.load_latest_model():
        print(f"✅ Model loaded: {inference_service.model_version}")
    else:
        print("❌ Failed to load model - service will be degraded")
    
    # Initialize blockchain integration service
    blockchain_service = BlockchainIntegrationService()
    blockchain_initialized = await blockchain_service.initialize()
    
    if blockchain_initialized:
        print("✅ Blockchain integration initialized")
        await blockchain_service.start_monitoring()
    else:
        print("❌ Failed to initialize blockchain integration - transactions disabled")
    
    print("🔄 Inference Relayer is ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down AI Risk Oracle Inference Relayer...")
    if inference_service:
        inference_service.clear_cache()
    if blockchain_service:
        await blockchain_service.stop_monitoring()
    print("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="AI Risk Oracle Inference Relayer",
    description="Production-ready REST API for real-time crypto risk assessment using ML and sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the inference service
def get_inference_service() -> RiskInferenceService:
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")
    if not inference_service.current_model:
        raise HTTPException(status_code=503, detail="No model loaded")
    return inference_service

# Dependency to get the blockchain service
def get_blockchain_service() -> BlockchainIntegrationService:
    if blockchain_service is None:
        raise HTTPException(status_code=503, detail="Blockchain service not initialized")
    return blockchain_service

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring"""
    try:
        uptime = (datetime.now() - service_start_time).total_seconds()
        
        # Get service status
        if inference_service:
            status_info = inference_service.get_service_status()
            components = status_info["components"]
            model_info = {
                "version": status_info["model_version"],
                "loaded_at": status_info["model_loaded_at"],
                "status": status_info["status"]
            }
        else:
            components = {"inference_service": "not_initialized"}
            model_info = {"status": "not_loaded"}
        
        # Determine overall health
        healthy_components = sum(1 for status in components.values() if status == "available")
        total_components = len(components)
        
        if healthy_components >= total_components * 0.8:  # 80% components healthy
            overall_status = "healthy"
        elif healthy_components >= total_components * 0.5:  # 50% components healthy
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            uptime_seconds=uptime,
            components=components,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status(service: RiskInferenceService = Depends(get_inference_service)):
    """Detailed service status endpoint"""
    try:
        status_info = service.get_service_status()
        
        # Calculate performance metrics
        uptime = (datetime.now() - service_start_time).total_seconds()
        
        return StatusResponse(
            service="AI Risk Oracle Inference Relayer",
            status=status_info["status"],
            current_time=datetime.now().isoformat(),
            model_version=status_info["model_version"],
            model_loaded_at=status_info["model_loaded_at"],
            cache_size=status_info["cache_size"],
            performance_metrics={
                "uptime_seconds": uptime,
                "cache_hit_ratio": 0.0,  # TODO: Implement cache metrics
                "avg_response_time_ms": 0.0  # TODO: Implement response time tracking
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/risk/{asset}", response_model=RiskScoreResponse)
async def get_risk_score(
    asset: str,
    service: RiskInferenceService = Depends(get_inference_service)
):
    """Get risk assessment for a single asset"""
    try:
        asset = asset.upper()
        
        # Validate asset
        valid_assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD", "QIE"]
        if asset not in valid_assets:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid asset. Supported assets: {', '.join(valid_assets)}"
            )
        
        # Get risk assessment
        assessment = await service.get_risk_assessment(asset)
        
        if not assessment:
            raise HTTPException(
                status_code=503, 
                detail=f"Unable to assess risk for {asset} - data unavailable"
            )
        
        # Generate reasoning
        reasoning = _generate_risk_reasoning(assessment)
        
        return RiskScoreResponse(
            asset_symbol=assessment.asset_symbol,
            timestamp=assessment.timestamp.isoformat(),
            combined_risk_score=assessment.combined_risk_score,
            risk_level=assessment.risk_level,
            ml_risk_score=assessment.ml_risk_score,
            sentiment_risk_score=assessment.sentiment_risk_score,
            confidence=assessment.confidence,
            market_regime=assessment.market_regime,
            correlation_signals=assessment.correlation_signals,
            inference_latency_ms=assessment.inference_latency_ms,
            data_freshness_seconds=assessment.data_freshness_seconds,
            reasoning=reasoning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@app.get("/risk", response_model=MultiAssetRiskResponse)
async def get_multi_asset_risk(
    assets: Optional[str] = None,
    service: RiskInferenceService = Depends(get_inference_service)
):
    """Get risk assessments for multiple assets"""
    try:
        start_time = time.time()
        
        # Parse assets parameter
        if assets:
            asset_list = [asset.strip().upper() for asset in assets.split(",")]
        else:
            asset_list = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD"]
        
        # Validate assets
        valid_assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD", "QIE"]
        invalid_assets = [asset for asset in asset_list if asset not in valid_assets]
        if invalid_assets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid assets: {', '.join(invalid_assets)}. Supported: {', '.join(valid_assets)}"
            )
        
        # Get assessments
        assessments = await service.get_multi_asset_assessment(asset_list)
        
        if not assessments:
            raise HTTPException(
                status_code=503,
                detail="Unable to assess risk for any assets - data unavailable"
            )
        
        # Convert to response format
        response_assessments = {}
        risk_scores = []
        
        for asset, assessment in assessments.items():
            reasoning = _generate_risk_reasoning(assessment)
            
            response_assessments[asset] = RiskScoreResponse(
                asset_symbol=assessment.asset_symbol,
                timestamp=assessment.timestamp.isoformat(),
                combined_risk_score=assessment.combined_risk_score,
                risk_level=assessment.risk_level,
                ml_risk_score=assessment.ml_risk_score,
                sentiment_risk_score=assessment.sentiment_risk_score,
                confidence=assessment.confidence,
                market_regime=assessment.market_regime,
                correlation_signals=assessment.correlation_signals,
                inference_latency_ms=assessment.inference_latency_ms,
                data_freshness_seconds=assessment.data_freshness_seconds,
                reasoning=reasoning
            )
            
            risk_scores.append(assessment.combined_risk_score)
        
        # Calculate summary statistics
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        max_risk = max(risk_scores) if risk_scores else 0
        min_risk = min(risk_scores) if risk_scores else 0
        
        # Count risk levels
        risk_levels = [assessment.risk_level for assessment in assessments.values()]
        risk_level_counts = {
            "CRITICAL": risk_levels.count("CRITICAL"),
            "HIGH": risk_levels.count("HIGH"),
            "MEDIUM": risk_levels.count("MEDIUM"),
            "LOW": risk_levels.count("LOW")
        }
        
        total_latency = (time.time() - start_time) * 1000
        
        return MultiAssetRiskResponse(
            timestamp=datetime.now().isoformat(),
            assessments=response_assessments,
            summary={
                "total_assets": len(assessments),
                "average_risk_score": round(avg_risk, 3),
                "highest_risk_score": round(max_risk, 3),
                "lowest_risk_score": round(min_risk, 3),
                "risk_level_distribution": risk_level_counts,
                "market_regime": _determine_overall_market_regime(assessments)
            },
            total_latency_ms=round(total_latency, 1)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-asset risk assessment failed: {str(e)}")

@app.get("/alerts", response_model=RiskAlertResponse)
async def get_risk_alerts(
    threshold: float = 0.7,
    service: RiskInferenceService = Depends(get_inference_service)
):
    """Get current risk alerts for assets above threshold"""
    try:
        # Get assessments for all major assets
        assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD"]
        assessments = await service.get_multi_asset_assessment(assets)
        
        # Filter assets above threshold
        triggered_assets = []
        max_risk = 0.0
        
        for asset, assessment in assessments.items():
            if assessment.combined_risk_score >= threshold:
                triggered_assets.append(asset)
                max_risk = max(max_risk, assessment.combined_risk_score)
        
        # Determine alert level
        if max_risk >= 0.9:
            alert_level = "CRITICAL"
            recommended_action = "IMMEDIATE HEDGE REQUIRED - Consider emergency portfolio protection"
        elif max_risk >= 0.7:
            alert_level = "HIGH"
            recommended_action = "HEDGE RECOMMENDED - Increase defensive positions"
        elif max_risk >= 0.5:
            alert_level = "MEDIUM"
            recommended_action = "MONITOR CLOSELY - Prepare for potential volatility"
        else:
            alert_level = "LOW"
            recommended_action = "NORMAL OPERATIONS - Continue regular monitoring"
        
        return RiskAlertResponse(
            alert_level=alert_level,
            triggered_assets=triggered_assets,
            max_risk_score=round(max_risk, 3),
            timestamp=datetime.now().isoformat(),
            recommended_action=recommended_action
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert check failed: {str(e)}")

@app.post("/cache/clear")
async def clear_cache(service: RiskInferenceService = Depends(get_inference_service)):
    """Clear the service cache"""
    try:
        service.clear_cache()
        return {"message": "Cache cleared successfully", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

@app.get("/blockchain/status")
async def get_blockchain_status(blockchain: BlockchainIntegrationService = Depends(get_blockchain_service)):
    """Get blockchain integration status"""
    try:
        status = await blockchain.get_blockchain_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain status check failed: {str(e)}")

@app.post("/blockchain/update")
async def manual_blockchain_update(
    risk_score: float,
    reason: str = "Manual API trigger",
    blockchain: BlockchainIntegrationService = Depends(get_blockchain_service)
):
    """Manually trigger a blockchain update"""
    try:
        if not (0.0 <= risk_score <= 1.0):
            raise HTTPException(status_code=400, detail="Risk score must be between 0.0 and 1.0")
        
        update = await blockchain.manual_update(risk_score, reason)
        
        if not update:
            raise HTTPException(status_code=500, detail="Failed to create blockchain update")
        
        return {
            "message": "Blockchain update triggered",
            "risk_score": risk_score,
            "reason": reason,
            "success": update.transaction_result.success if update.transaction_result else False,
            "tx_hash": update.transaction_result.tx_hash if update.transaction_result else None,
            "timestamp": update.triggered_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual blockchain update failed: {str(e)}")

@app.get("/blockchain/history")
async def get_blockchain_history(
    limit: int = 20,
    blockchain: BlockchainIntegrationService = Depends(get_blockchain_service)
):
    """Get blockchain update history"""
    try:
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
        
        history = blockchain.get_update_history(limit)
        
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blockchain history: {str(e)}")

@app.post("/risk/{asset}/trigger")
async def trigger_risk_assessment_with_blockchain(
    asset: str,
    force_blockchain_update: bool = False,
    service: RiskInferenceService = Depends(get_inference_service),
    blockchain: BlockchainIntegrationService = Depends(get_blockchain_service)
):
    """Get risk assessment and optionally trigger blockchain update"""
    try:
        asset = asset.upper()
        
        # Validate asset
        valid_assets = ["BTC", "ETH", "XRP", "SOL", "BNB", "GOLD", "QIE"]
        if asset not in valid_assets:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid asset. Supported assets: {', '.join(valid_assets)}"
            )
        
        # Get risk assessment
        assessment = await service.get_risk_assessment(asset)
        
        if not assessment:
            raise HTTPException(
                status_code=503, 
                detail=f"Unable to assess risk for {asset} - data unavailable"
            )
        
        # Process blockchain update
        blockchain_update = await blockchain.process_risk_assessment(
            assessment, force_update=force_blockchain_update
        )
        
        # Generate reasoning
        reasoning = _generate_risk_reasoning(assessment)
        
        # Prepare response
        response = {
            "risk_assessment": {
                "asset_symbol": assessment.asset_symbol,
                "timestamp": assessment.timestamp.isoformat(),
                "combined_risk_score": assessment.combined_risk_score,
                "risk_level": assessment.risk_level,
                "ml_risk_score": assessment.ml_risk_score,
                "sentiment_risk_score": assessment.sentiment_risk_score,
                "confidence": assessment.confidence,
                "market_regime": assessment.market_regime,
                "correlation_signals": assessment.correlation_signals,
                "inference_latency_ms": assessment.inference_latency_ms,
                "data_freshness_seconds": assessment.data_freshness_seconds,
                "reasoning": reasoning
            },
            "blockchain_update": None
        }
        
        # Add blockchain update info if it occurred
        if blockchain_update:
            response["blockchain_update"] = {
                "triggered": True,
                "trigger_reason": blockchain_update.trigger_reason.value,
                "success": blockchain_update.transaction_result.success if blockchain_update.transaction_result else False,
                "tx_hash": blockchain_update.transaction_result.tx_hash if blockchain_update.transaction_result else None,
                "error": blockchain_update.transaction_result.error_message if blockchain_update.transaction_result and not blockchain_update.transaction_result.success else None,
                "latency_ms": blockchain_update.transaction_result.latency_ms if blockchain_update.transaction_result else 0
            }
        else:
            response["blockchain_update"] = {
                "triggered": False,
                "reason": "Risk threshold not met or update interval not reached"
            }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment with blockchain trigger failed: {str(e)}")

def _generate_risk_reasoning(assessment: RiskAssessment) -> List[str]:
    """Generate human-readable reasoning for risk assessment"""
    reasoning = []
    
    # ML model reasoning
    if assessment.ml_risk_score >= 0.8:
        reasoning.append(f"ML model detects HIGH RISK ({assessment.ml_risk_score:.3f}) based on market patterns")
    elif assessment.ml_risk_score >= 0.6:
        reasoning.append(f"ML model indicates ELEVATED RISK ({assessment.ml_risk_score:.3f})")
    elif assessment.ml_risk_score <= 0.3:
        reasoning.append(f"ML model shows LOW RISK ({assessment.ml_risk_score:.3f})")
    
    # Sentiment reasoning
    if assessment.sentiment_risk_score >= 0.7:
        reasoning.append(f"Negative sentiment detected ({assessment.sentiment_risk_score:.3f})")
    elif assessment.sentiment_risk_score <= 0.3:
        reasoning.append(f"Positive sentiment detected ({assessment.sentiment_risk_score:.3f})")
    
    # Market regime
    if assessment.market_regime == "high_volatility":
        reasoning.append("High volatility market regime detected")
    elif assessment.market_regime == "crisis":
        reasoning.append("CRISIS market regime - extreme conditions")
    
    # Correlation signals
    if assessment.correlation_signals:
        reasoning.append(f"Correlation patterns: {', '.join(assessment.correlation_signals)}")
    
    # Confidence
    if assessment.confidence < 0.5:
        reasoning.append(f"LOW CONFIDENCE ({assessment.confidence:.3f}) - limited data quality")
    elif assessment.confidence > 0.8:
        reasoning.append(f"HIGH CONFIDENCE ({assessment.confidence:.3f}) - strong signal quality")
    
    # Data freshness
    if assessment.data_freshness_seconds > 300:  # 5 minutes
        reasoning.append(f"WARNING: Data is {assessment.data_freshness_seconds}s old")
    
    return reasoning

def _determine_overall_market_regime(assessments: Dict[str, RiskAssessment]) -> str:
    """Determine overall market regime from multiple assessments"""
    regimes = [assessment.market_regime for assessment in assessments.values()]
    
    if "crisis" in regimes:
        return "crisis"
    elif regimes.count("high_volatility") >= len(regimes) * 0.5:
        return "high_volatility"
    else:
        return "normal"

# Background task for periodic cache warming
async def warm_cache():
    """Background task to keep cache warm with fresh data"""
    if inference_service:
        try:
            # Get assessments for major assets to warm cache
            await inference_service.get_multi_asset_assessment(["BTC", "ETH", "XRP"])
        except Exception:
            pass  # Ignore errors in background task

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "services.api.inference_relayer:app",
        host="0.0.0.0",
        port=8001,  # Use different port to avoid conflicts
        reload=False,
        log_level="info"
    )