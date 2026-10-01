| scorer | level | n_features | AP | AP_sd | R50 | P25 | vendor_AP | fp_top50 |
|---|---|---|---|---|---|---|---|---|
| isolation_forest | D explicit | 35 | 0.314 | 0.016 | 0.444 | 0.413 | 0.473 | 35.333 |
| logistic_oof | D explicit | 35 | 0.535 | 0.076 | 0.667 | 0.640 | 0.696 | 28.000 |
| isolation_forest | E embedding only | 10 | 0.088 | 0.018 | 0.202 | 0.120 | 0.208 | 43.333 |
| logistic_oof | E embedding only | 10 | 0.322 | 0.186 | 0.535 | 0.387 | 0.440 | 32.333 |
| isolation_forest | D + E | 45 | 0.213 | 0.028 | 0.424 | 0.320 | 0.378 | 36.000 |
| logistic_oof | D + E | 45 | 0.542 | 0.093 | 0.687 | 0.573 | 0.712 | 27.333 |
| isolation_forest | E only, encoder fitted to 2024-06-30 | 10 | 0.070 | 0.018 | 0.212 | 0.107 | 0.222 | 43.000 |
| logistic_oof | E only, encoder fitted to 2024-06-30 | 10 | 0.256 | 0.027 | 0.505 | 0.360 | 0.425 | 33.333 |
| isolation_forest | D + E, encoder fitted to 2024-06-30 | 45 | 0.218 | 0.023 | 0.414 | 0.333 | 0.396 | 36.333 |
| logistic_oof | D + E, encoder fitted to 2024-06-30 | 45 | 0.539 | 0.053 | 0.717 | 0.600 | 0.701 | 26.333 |