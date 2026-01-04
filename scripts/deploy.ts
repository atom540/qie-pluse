import { network } from "hardhat";


const { ethers } = await network.connect({
  network: "hardhatOp",
  chainType: "op",
});

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying with:", deployer.address);

  const Oracle = await ethers.getContractFactory("AIRiskOracle");
  const oracle = await Oracle.deploy(deployer.address);
  await oracle.waitForDeployment();

  const ORF = await ethers.getContractFactory("ORFToken");
  const orf = await ORF.deploy(deployer.address);
  await orf.waitForDeployment();

  const Engine = await ethers.getContractFactory("RiskEngine");
  const engine = await Engine.deploy(
    await oracle.getAddress(),
    await orf.getAddress()
  );
  await engine.waitForDeployment();

  console.log("Oracle:", await oracle.getAddress());
  console.log("ORF:", await orf.getAddress());
  console.log("Engine:", await engine.getAddress());
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
