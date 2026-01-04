// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract AIRiskOracle {
    address public updater;
    uint8 public riskScore;
    uint256 public lastUpdated;

    event RiskUpdated(uint8 score, uint256 timestamp);

    constructor(address _updater) {
        updater = _updater;
    }

    modifier onlyUpdater() {
        require(msg.sender == updater, "Not authorized");
        _;
    }

    function updateRisk(uint8 _score) external onlyUpdater {
        require(_score <= 100, "Invalid score");
        riskScore = _score;
        lastUpdated = block.timestamp;

        emit RiskUpdated(_score, block.timestamp);
    }
}
