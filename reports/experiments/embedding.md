| scorer | level | n_features | AP | AP_sd | R50 | P25 | vendor_AP | fp_top50 |
|---|---|---|---|---|---|---|---|---|
| isolation_forest | D explicit | 35 | 0.274 | 0.032 | 0.465 | 0.413 | 0.401 | 34.667 |
| logistic_oof | D explicit | 35 | 0.505 | 0.022 | 0.667 | 0.587 | 0.669 | 28.000 |
| isolation_forest | E embedding only | 10 | 0.081 | 0.022 | 0.212 | 0.160 | 0.215 | 43.000 |
| logistic_oof | E embedding only | 10 | 0.303 | 0.032 | 0.586 | 0.373 | 0.438 | 30.667 |
| isolation_forest | D + E | 45 | 0.180 | 0.016 | 0.394 | 0.333 | 0.320 | 37.000 |
| logistic_oof | D + E | 45 | 0.518 | 0.018 | 0.707 | 0.560 | 0.682 | 26.667 |
| isolation_forest | E only, encoder fitted to 2024-06-30 | 10 | 0.064 | 0.005 | 0.253 | 0.147 | 0.210 | 41.667 |
| logistic_oof | E only, encoder fitted to 2024-06-30 | 10 | 0.296 | 0.065 | 0.535 | 0.373 | 0.507 | 32.333 |
| isolation_forest | D + E, encoder fitted to 2024-06-30 | 45 | 0.206 | 0.034 | 0.404 | 0.280 | 0.345 | 36.667 |
| logistic_oof | D + E, encoder fitted to 2024-06-30 | 45 | 0.530 | 0.040 | 0.758 | 0.600 | 0.696 | 25.000 |