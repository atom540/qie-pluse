import hardhatToolboxMochaEthersPlugin from "@nomicfoundation/hardhat-toolbox-mocha-ethers";
import { configVariable, defineConfig } from "hardhat/config";

export default defineConfig({
  plugins: [hardhatToolboxMochaEthersPlugin,],
  solidity: {
    profiles: {
      default: {
        version: "0.8.28",
      },
      production: {
        version: "0.8.28",
        settings: {
          optimizer: {
            enabled: true,
            runs: 200,
          },
        },
      },
    },
  },
  networks: {
    hardhatMainnet: {
      type: "edr-simulated",
      chainType: "l1",
    },
    hardhatOp: {
      type: "edr-simulated",
      chainType: "op",
    },
    qie: {
      type: "http",
      chainType: "l1",
      url: "https://rpc-main2.qiblockchain.online/",
      accounts: ["0c24c9fc9910c278efac56ccdab009af7e1ac0833e11c49cb0a6a9f4aa6ed619"],
      chainId: Number(5656),
    },
  },
});
