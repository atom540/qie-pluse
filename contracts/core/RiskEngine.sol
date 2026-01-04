// SPDX-License-Identifier: MIT

pragma solidity ^0.8.20;

import "../oracle/AIRiskOracle.sol";
import "./ORFToken.sol";

contract RiskEngine {

    enum RiskState { SAFE, WARNING, HIGH, CRITICAL }

    AIRiskOracle public oracle;
    ORFToken public orf;
    RiskState public currentState;

    event StateChanged(RiskState state);

    constructor(address oracleAddr, address orfAddr) {
        oracle = AIRiskOracle(oracleAddr);
        orf = ORFToken(orfAddr);
    }

    function evaluateRisk() external {
        uint8 score = oracle.riskScore();

        RiskState newState;

        if (score <= 30) newState = RiskState.SAFE;
        else if (score <= 60) newState = RiskState.WARNING;
        else if (score <= 80) newState = RiskState.HIGH;
        else newState = RiskState.CRITICAL;

        if (newState != currentState) {
            currentState = newState;
            emit StateChanged(newState);

            if (newState == RiskState.HIGH) {
                orf.mint(msg.sender, 100 * 1e18);
            }

            if (newState == RiskState.CRITICAL) {
                orf.mint(msg.sender, 200 * 1e18);
            }
        }
    }
}
