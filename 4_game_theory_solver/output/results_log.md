====================================================================
  DUAL-TRACK STACKELBERG AUDIT POLICY ANALYSIS
  Single-attacker-type framing (m=1)
====================================================================
  Defender reward R_D = 100.0
  Audit cost per layer c = 2.5
  Layers tested: [8, 9, 10, 11, 12, 13, 14, 15, 16]
  Jitter levels: [0.0, 0.01, 0.02, 0.05, 0.1]
  Lambda levels (L2 reg.): [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]

====================================================================
  REGIME: Track A (alpha=3.0)
====================================================================
  Attacker benefit B = 33.6
  Q vector (by layer [8, 9, 10, 11, 12, 13, 14, 15, 16]):
    [0.270,   nan,   nan, 0.530,   nan,   nan,   nan,   nan,   nan]
  Defined layers: 2/9
  Defined-entries stats: min=0.270  max=0.530  std=0.130
  Regime classification: UNDEFINED

  Jitter sensitivity:
   epsilon |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    50.5000 |   0.0000 |   1.0000 |            11 | optimal
      0.01 |    50.3057 |   0.0000 |   1.0000 |            11 | optimal
      0.02 |    50.6523 |   0.0000 |   1.0000 |            11 | optimal
      0.05 |    49.6412 |   0.0000 |   1.0000 |            11 | optimal
      0.10 |    47.6696 |   0.0000 |   1.0000 |            11 | optimal

  Raw policy (pure LP, epsilon=0, lambda=0) by layer:
    Layer  8: p = 0.0000  
    Layer  9: p = 0.0000  
    Layer 10: p = 0.0000  
    Layer 11: p = 1.0000  #######################################
    Layer 12: p = 0.0000  
    Layer 13: p = 0.0000  
    Layer 14: p = 0.0000  
    Layer 15: p = 0.0000  
    Layer 16: p = 0.0000  

  Regularisation sweep (L2 penalty -lambda*||p||^2):
    lambda |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    50.5000 |   0.0000 |   1.0000 |            11 | optimal
      1.00 |    49.5000 |   0.0000 |   1.0000 |            11 | optimal
      5.00 |    45.5000 |   0.0000 |   1.0000 |            11 | optimal
     10.00 |    40.5000 |   0.0000 |   1.0000 |            11 | optimal
     25.00 |    28.3800 |   0.5511 |   0.7600 |            11 | optimal
     50.00 |    14.9678 |   1.2796 |   0.5522 |            11 | optimal

  Regularised policies by layer (each row = one lambda):
    lambda | L08 L09 L10 L11 L12 L13 L14 L15 L16
  ----------------------------------------------
      0.00 | 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00
      1.00 | 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00
      5.00 | 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00
     10.00 | 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00
     25.00 | 0.24 0.00 0.00 0.76 0.00 0.00 0.00 0.00 0.00
     50.00 | 0.29 0.02 0.02 0.55 0.02 0.02 0.02 0.02 0.02

====================================================================
  REGIME: Track A sensitive (alpha=1.5)
====================================================================
  Attacker benefit B = 33.6
  Q vector (by layer [8, 9, 10, 11, 12, 13, 14, 15, 16]):
    [0.470, 0.530, 0.655, 0.340, 0.240, 0.530, 0.530, 0.530, 0.530]
  Defined layers: 9/9
  Defined-entries stats: min=0.240  max=0.655  std=0.116
  Regime classification: WELL_POSED

  Jitter sensitivity:
   epsilon |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    63.0000 |   0.0000 |   1.0000 |            10 | optimal
      0.01 |    62.4036 |   0.0000 |   1.0000 |            10 | optimal
      0.02 |    63.0205 |   0.0000 |   1.0000 |            10 | optimal
      0.05 |    60.5035 |   0.0000 |   1.0000 |            10 | optimal
      0.10 |    68.6271 |   0.0000 |   1.0000 |            10 | optimal

  Raw policy (pure LP, epsilon=0, lambda=0) by layer:
    Layer  8: p = 0.0000  
    Layer  9: p = 0.0000  
    Layer 10: p = 1.0000  #######################################
    Layer 11: p = 0.0000  
    Layer 12: p = 0.0000  
    Layer 13: p = 0.0000  
    Layer 14: p = 0.0000  
    Layer 15: p = 0.0000  
    Layer 16: p = 0.0000  

  Regularisation sweep (L2 penalty -lambda*||p||^2):
    lambda |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    63.0000 |   0.0000 |   1.0000 |            10 | optimal
      1.00 |    62.0000 |   0.0000 |   1.0000 |            10 | optimal
      5.00 |    58.0000 |   0.0000 |   1.0000 |            10 | optimal
     10.00 |    54.1719 |   1.1240 |   0.6875 |            10 | optimal
     25.00 |    49.7193 |   1.6870 |   0.3743 |            10 | optimal
     50.00 |    45.2168 |   1.8862 |   0.2586 |            10 | optimal

  Regularised policies by layer (each row = one lambda):
    lambda | L08 L09 L10 L11 L12 L13 L14 L15 L16
  ----------------------------------------------
      0.00 | 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00
      1.00 | 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00
      5.00 | 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00
     10.00 | 0.00 0.06 0.69 0.00 0.00 0.06 0.06 0.06 0.06
     25.00 | 0.00 0.12 0.37 0.00 0.00 0.12 0.12 0.12 0.12
     50.00 | 0.07 0.13 0.26 0.00 0.00 0.13 0.13 0.13 0.13

====================================================================
  REGIME: Track B (alpha=3.0)
====================================================================
  Attacker benefit B = 42.3
  Q vector (by layer [8, 9, 10, 11, 12, 13, 14, 15, 16]):
    [1.000, 0.400, 0.155, 0.510, 1.000, 1.000, 1.000, 1.000, 1.000]
  Defined layers: 9/9
  Defined-entries stats: min=0.155  max=1.000  std=0.316
  Regime classification: WELL_POSED

  Jitter sensitivity:
   epsilon |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    97.5000 |   1.7918 |   0.1667 |             8 | optimal
      0.01 |    97.5000 |   0.6931 |   0.5000 |            13 | optimal
      0.02 |    97.5000 |   1.3863 |   0.2500 |             8 | optimal
      0.05 |    97.5000 |   0.6931 |   0.5000 |            13 | optimal
      0.10 |    97.5000 |   1.0986 |   0.3333 |             8 | optimal

  Raw policy (pure LP, epsilon=0, lambda=0) by layer:
    Layer  8: p = 0.1667  ######
    Layer  9: p = 0.0000  
    Layer 10: p = 0.0000  
    Layer 11: p = 0.0000  
    Layer 12: p = 0.1667  ######
    Layer 13: p = 0.1667  ######
    Layer 14: p = 0.1667  ######
    Layer 15: p = 0.1667  ######
    Layer 16: p = 0.1667  ######

  Regularisation sweep (L2 penalty -lambda*||p||^2):
    lambda |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    97.5000 |   1.7918 |   0.1667 |             8 | optimal
      1.00 |    97.3333 |   1.7918 |   0.1667 |            12 | optimal
      5.00 |    96.6667 |   1.7918 |   0.1667 |            12 | optimal
     10.00 |    95.8333 |   1.7918 |   0.1667 |             8 | optimal
     25.00 |    93.3333 |   1.7918 |   0.1667 |             8 | optimal
     50.00 |    89.1667 |   1.7918 |   0.1667 |            12 | optimal

  Regularised policies by layer (each row = one lambda):
    lambda | L08 L09 L10 L11 L12 L13 L14 L15 L16
  ----------------------------------------------
      0.00 | 0.17 0.00 0.00 0.00 0.17 0.17 0.17 0.17 0.17
      1.00 | 0.17 0.00 0.00 0.00 0.17 0.17 0.17 0.17 0.17
      5.00 | 0.17 0.00 0.00 0.00 0.17 0.17 0.17 0.17 0.17
     10.00 | 0.17 0.00 0.00 0.00 0.17 0.17 0.17 0.17 0.17
     25.00 | 0.17 0.00 0.00 0.00 0.17 0.17 0.17 0.17 0.17
     50.00 | 0.17 0.00 0.00 0.00 0.17 0.17 0.17 0.17 0.17

====================================================================
  REGIME: Clean baseline
====================================================================
  Attacker benefit B = 0.0
  Q vector (by layer [8, 9, 10, 11, 12, 13, 14, 15, 16]):
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  Defined layers: 9/9
  Defined-entries stats: min=0.000  max=0.000  std=0.000
  Regime classification: DEGENERATE

  Jitter sensitivity:
   epsilon |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    -2.5000 |   2.1972 |   0.1111 |             8 | optimal
      0.01 |    -2.5000 |   2.1972 |   0.1111 |             8 | optimal
      0.02 |    -2.5000 |   2.1972 |   0.1111 |             8 | optimal
      0.05 |    -2.5000 |   2.1972 |   0.1111 |             8 | optimal
      0.10 |    -2.5000 |   2.1972 |   0.1111 |             8 | optimal

  Raw policy (pure LP, epsilon=0, lambda=0) by layer:
    Layer  8: p = 0.1111  ####
    Layer  9: p = 0.1111  ####
    Layer 10: p = 0.1111  ####
    Layer 11: p = 0.1111  ####
    Layer 12: p = 0.1111  ####
    Layer 13: p = 0.1111  ####
    Layer 14: p = 0.1111  ####
    Layer 15: p = 0.1111  ####
    Layer 16: p = 0.1111  ####

  Regularisation sweep (L2 penalty -lambda*||p||^2):
    lambda |    utility |  entropy |   max p* |  argmax layer | status
  ------------------------------------------------------------------------------
      0.00 |    -2.5000 |   2.1972 |   0.1111 |             8 | optimal
      1.00 |    -2.6111 |   2.1972 |   0.1111 |             8 | optimal
      5.00 |    -3.0556 |   2.1972 |   0.1111 |             8 | optimal
     10.00 |    -3.6111 |   2.1972 |   0.1111 |             8 | optimal
     25.00 |    -5.2778 |   2.1972 |   0.1111 |             8 | optimal
     50.00 |    -8.0556 |   2.1972 |   0.1111 |             8 | optimal

  Regularised policies by layer (each row = one lambda):
    lambda | L08 L09 L10 L11 L12 L13 L14 L15 L16
  ----------------------------------------------
      0.00 | 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11
      1.00 | 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11
      5.00 | 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11
     10.00 | 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11
     25.00 | 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11
     50.00 | 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11 0.11

====================================================================
  CROSS-REGIME POLICY COMPARISON (pure LP, epsilon=0, lambda=0)
====================================================================

  Track A (alpha=3.0)  vs  Track B (alpha=3.0)
    L1 distance       : 2.0000
    L2 distance       : 1.0801
    Peak layer match  : False  (layer 11 vs 8)

  Track A (alpha=3.0)  vs  Track A sensitive (alpha=1.5)
    L1 distance       : 2.0000
    L2 distance       : 1.4142
    Peak layer match  : False  (layer 11 vs 10)

  Track A sensitive (alpha=1.5)  vs  Track B (alpha=3.0)
    L1 distance       : 2.0000
    L2 distance       : 1.0801
    Peak layer match  : False  (layer 10 vs 8)

  Track A (alpha=3.0)  vs  Clean baseline
    L1 distance       : 1.7778
    L2 distance       : 0.9428
    Peak layer match  : False  (layer 11 vs 8)

  Track B (alpha=3.0)  vs  Clean baseline
    L1 distance       : 0.6667
    L2 distance       : 0.2357
    Peak layer match  : True  (layer 8 vs 8)

  Jitter sweep exported: output\stackelberg_jitter_sweep.csv
  Lambda sweep exported: output\stackelberg_lambda_sweep.csv

====================================================================
  SUMMARY
====================================================================
  Policy L2 shift (pure LP)  ||p*_A - p*_B||  =  1.0801
  -> Allocations SHIFT: de novo vs amplified backdoors demand
     measurably different audit allocations.
====================================================================