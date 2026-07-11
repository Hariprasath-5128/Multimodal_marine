# Two-Stage Marine Pipeline — Evaluation Report

**Weights:** Top-1=1.0 | Top-2=0.35 | Top-3=0.15

## Domain Summary

| Domain | N | Top-1 | Top-2 | Top-3 | Miss | DomErr | Coverage | WgtAcc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dolphin | 138 | 121 | 7 | 1 | 9 | 1 | 93.5% | **89.6%** | [PASS] |
| whale | 81 | 70 | 4 | 1 | 6 | 3 | 92.6% | **88.3%** | [PASS] |
| seal | 73 | 60 | 5 | 2 | 6 | 1 | 91.8% | **85.0%** | [PASS] |
| sealion | 29 | 22 | 3 | 1 | 3 | 2 | 89.7% | **80.0%** | [PASS] |
| porpoise | 17 | 13 | 1 | 0 | 3 | 3 | 82.4% | **78.5%** | [PASS] |
| manatee | 17 | 13 | 4 | 0 | 0 | 0 | 100.0% | **84.7%** | [PASS] |
| **TOTAL** | 355 | | | | | | | **86.8%** | [PASS] |

## Per-Species Detail


### Dolphin

| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| amazon river dolphin | 7 | 7 | 0 | 0 | 0 | 100.0% |
| atlantic spotted dolphin | 7 | 7 | 0 | 0 | 0 | 100.0% |
| bolivian river dolphin | 5 | 4 | 0 | 0 | 0 | 80.0% |
| bottlenose dolphin | 3 | 2 | 0 | 0 | 0 | 66.7% |
| burrunan dolphin | 5 | 4 | 0 | 1 | 0 | 83.0% |
| clymene dolphin | 6 | 4 | 1 | 0 | 1 | 72.5% |
| common dolphin | 10 | 10 | 0 | 0 | 0 | 100.0% |
| dusky dolphin | 7 | 7 | 0 | 0 | 0 | 100.0% |
| frasers dolphin | 3 | 2 | 1 | 0 | 0 | 78.3% |
| guiana dolphin | 5 | 4 | 0 | 0 | 0 | 80.0% |
| hectors dolphin | 9 | 9 | 0 | 0 | 0 | 100.0% |
| indo-pacific bottlenose dolphin | 7 | 5 | 1 | 0 | 0 | 76.4% |
| indo-pacific humpbacked dolphin | 4 | 4 | 0 | 0 | 0 | 100.0% |
| irrawaddy dolphin | 4 | 4 | 0 | 0 | 0 | 100.0% |
| la plata dolphin | 4 | 4 | 0 | 0 | 0 | 100.0% |
| pantropical spotted dolphin | 3 | 2 | 0 | 0 | 0 | 66.7% |
| risso's dolphin | 5 | 4 | 1 | 0 | 0 | 87.0% |
| rough-toothed dolphin | 4 | 3 | 0 | 0 | 0 | 75.0% |
| south asian river dolphin | 5 | 5 | 0 | 0 | 0 | 100.0% |
| spinner dolphin | 6 | 5 | 1 | 0 | 0 | 89.2% |
| striped dolphin | 10 | 10 | 0 | 0 | 0 | 100.0% |
| vaquita | 6 | 6 | 0 | 0 | 0 | 100.0% |
| white-beaked dolphin | 6 | 4 | 1 | 0 | 0 | 72.5% |
| white-sided dolphin | 7 | 5 | 1 | 0 | 0 | 76.4% |

### Whale

| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| beluga whale | 1 | 0 | 0 | 0 | 1 | 0.0% |
| bowhead whale | 4 | 4 | 0 | 0 | 0 | 100.0% |
| dwarf sperm whale | 6 | 4 | 0 | 0 | 0 | 66.7% |
| false killer whale | 5 | 4 | 0 | 0 | 1 | 80.0% |
| fin whale | 6 | 6 | 0 | 0 | 0 | 100.0% |
| gray whale | 8 | 7 | 1 | 0 | 0 | 91.9% |
| humpback whale | 4 | 4 | 0 | 0 | 0 | 100.0% |
| killer whale | 1 | 1 | 0 | 0 | 0 | 100.0% |
| long-finned pilot whale | 2 | 2 | 0 | 0 | 0 | 100.0% |
| melon-headed whale | 3 | 3 | 0 | 0 | 0 | 100.0% |
| minke whale | 3 | 3 | 0 | 0 | 0 | 100.0% |
| narwhal | 5 | 4 | 1 | 0 | 0 | 87.0% |
| north atlantic right whale | 3 | 2 | 0 | 1 | 0 | 71.7% |
| orca | 10 | 9 | 0 | 0 | 1 | 90.0% |
| right whale | 6 | 5 | 1 | 0 | 0 | 89.2% |
| short-finned pilot whale | 4 | 3 | 1 | 0 | 0 | 83.8% |
| southern right whale | 4 | 3 | 0 | 0 | 0 | 75.0% |
| sperm whale | 6 | 6 | 0 | 0 | 0 | 100.0% |

### Seal

| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| baikal seal | 6 | 5 | 1 | 0 | 0 | 89.2% |
| bearded seal | 4 | 4 | 0 | 0 | 0 | 100.0% |
| caspian seal | 6 | 5 | 0 | 0 | 0 | 83.3% |
| grey seal | 5 | 4 | 0 | 0 | 0 | 80.0% |
| harp seal | 5 | 4 | 1 | 0 | 0 | 87.0% |
| hooded seal | 5 | 4 | 0 | 0 | 0 | 80.0% |
| leopard seal | 5 | 5 | 0 | 0 | 0 | 100.0% |
| mediterranean monk seal | 6 | 5 | 0 | 1 | 0 | 85.8% |
| northern elephant seal | 5 | 5 | 0 | 0 | 0 | 100.0% |
| ringed seal | 5 | 5 | 0 | 0 | 0 | 100.0% |
| ross seal | 4 | 1 | 0 | 0 | 1 | 25.0% |
| southern elephant seal | 6 | 4 | 2 | 0 | 0 | 78.3% |
| spotted seal | 5 | 4 | 1 | 0 | 0 | 87.0% |
| weddell seal | 6 | 5 | 0 | 1 | 0 | 85.8% |

### Sealion

| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| australian sea lion | 6 | 4 | 0 | 0 | 2 | 66.7% |
| california sea lion | 6 | 5 | 0 | 1 | 0 | 85.8% |
| new zealand sea lion | 6 | 5 | 1 | 0 | 0 | 89.2% |
| south american sea lion | 6 | 3 | 2 | 0 | 0 | 61.7% |
| steller sea lion | 5 | 5 | 0 | 0 | 0 | 100.0% |

### Porpoise

| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| dalls porpoise | 6 | 5 | 0 | 0 | 1 | 83.3% |
| harbour porpoise | 6 | 5 | 0 | 0 | 1 | 83.3% |
| indo-pacific finless porpoise | 5 | 3 | 1 | 0 | 1 | 67.0% |

### Manatee

| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| amazonian manatee | 6 | 5 | 1 | 0 | 0 | 89.2% |
| west african manatee | 5 | 4 | 1 | 0 | 0 | 87.0% |
| west indian manatee | 6 | 4 | 2 | 0 | 0 | 78.3% |