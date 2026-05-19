import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合非交互式环境
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# 日志内容处理（假设已复制日志内容到log_text变量）
log_text = '''
2025-11-19 10:47:53,779 - INFO - Epoch 0 | Loss: 8.9588 | F1: 0.4608 | Precision: 0.9274 | Recall: 0.5000 | FPR: 0.0000 | AUC: 0.7436
2025-11-19 10:47:53,779 - INFO - Best model state updated at epoch 0.
2025-11-19 10:47:56,587 - INFO - Epoch 20 | Loss: 5.3377 | F1: 0.6687 | Precision: 0.6507 | Recall: 0.7398 | FPR: 0.2013 | AUC: 0.8257
2025-11-19 10:47:56,588 - INFO - Best model state updated at epoch 20.
2025-11-19 10:47:59,447 - INFO - Epoch 40 | Loss: 5.0753 | F1: 0.6672 | Precision: 0.6522 | Recall: 0.7577 | FPR: 0.2260 | AUC: 0.8390
2025-11-19 10:47:59,448 - INFO - Best model state updated at epoch 40.
2025-11-19 10:48:02,234 - INFO - Epoch 60 | Loss: 4.5564 | F1: 0.6851 | Precision: 0.6648 | Recall: 0.7664 | FPR: 0.2004 | AUC: 0.8501
2025-11-19 10:48:02,235 - INFO - Best model state updated at epoch 60.
2025-11-19 10:48:05,038 - INFO - Epoch 80 | Loss: 4.2426 | F1: 0.7119 | Precision: 0.6878 | Recall: 0.7649 | FPR: 0.1495 | AUC: 0.8570
2025-11-19 10:48:05,038 - INFO - Best model state updated at epoch 80.
2025-11-19 10:48:07,865 - INFO - Epoch 100 | Loss: 3.9269 | F1: 0.6952 | Precision: 0.6727 | Recall: 0.7724 | FPR: 0.1883 | AUC: 0.8572
2025-11-19 10:48:07,865 - INFO - Best model state updated at epoch 100.
2025-11-19 10:48:10,713 - INFO - Epoch 120 | Loss: 3.4968 | F1: 0.7010 | Precision: 0.6773 | Recall: 0.7724 | FPR: 0.1776 | AUC: 0.8580
2025-11-19 10:48:10,713 - INFO - Best model state updated at epoch 120.
2025-11-19 10:48:13,529 - INFO - Epoch 140 | Loss: 3.5327 | F1: 0.7119 | Precision: 0.6868 | Recall: 0.7746 | FPR: 0.1605 | AUC: 0.8581
2025-11-19 10:48:16,385 - INFO - Epoch 160 | Loss: 3.0880 | F1: 0.7119 | Precision: 0.6871 | Recall: 0.7703 | FPR: 0.1558 | AUC: 0.8579
2025-11-19 10:48:16,386 - INFO - Best model state updated at epoch 160.
2025-11-19 10:48:19,258 - INFO - Epoch 180 | Loss: 3.0464 | F1: 0.7187 | Precision: 0.6949 | Recall: 0.7653 | FPR: 0.1384 | AUC: 0.8557
2025-11-19 10:48:19,258 - INFO - Best model state updated at epoch 180.
2025-11-19 10:48:22,083 - INFO - Epoch 200 | Loss: 2.7148 | F1: 0.7169 | Precision: 0.6928 | Recall: 0.7663 | FPR: 0.1425 | AUC: 0.8582
2025-11-19 10:48:22,083 - INFO - Best model state updated at epoch 200.
2025-11-19 10:48:25,025 - INFO - Epoch 220 | Loss: 2.7851 | F1: 0.7183 | Precision: 0.6941 | Recall: 0.7673 | FPR: 0.1413 | AUC: 0.8569
2025-11-19 10:48:27,922 - INFO - Epoch 240 | Loss: 2.5382 | F1: 0.7203 | Precision: 0.6990 | Recall: 0.7567 | FPR: 0.1257 | AUC: 0.8497
2025-11-19 10:48:27,923 - INFO - Best model state updated at epoch 240.
2025-11-19 10:48:30,814 - INFO - Epoch 260 | Loss: 2.2843 | F1: 0.7159 | Precision: 0.6926 | Recall: 0.7612 | FPR: 0.1384 | AUC: 0.8514
2025-11-19 10:48:30,815 - INFO - Best model state updated at epoch 260.
2025-11-19 10:48:33,700 - INFO - Epoch 280 | Loss: 2.1019 | F1: 0.7168 | Precision: 0.6928 | Recall: 0.7653 | FPR: 0.1416 | AUC: 0.8555
2025-11-19 10:48:33,700 - INFO - Best model state updated at epoch 280.
2025-11-19 10:48:36,579 - INFO - Epoch 300 | Loss: 2.2648 | F1: 0.7283 | Precision: 0.7120 | Recall: 0.7508 | FPR: 0.1060 | AUC: 0.8565
2025-11-19 10:48:36,579 - INFO - Best model state updated at epoch 300.
2025-11-19 10:48:39,477 - INFO - Epoch 320 | Loss: 2.0707 | F1: 0.7253 | Precision: 0.7056 | Recall: 0.7562 | FPR: 0.1169 | AUC: 0.8559
2025-11-19 10:48:39,477 - INFO - Best model state updated at epoch 320.
2025-11-19 10:48:42,410 - INFO - Epoch 340 | Loss: 2.0669 | F1: 0.7219 | Precision: 0.7096 | Recall: 0.7375 | FPR: 0.1004 | AUC: 0.8450
2025-11-19 10:48:42,411 - INFO - Best model state updated at epoch 340.
2025-11-19 10:48:45,433 - INFO - Epoch 360 | Loss: 1.8642 | F1: 0.7222 | Precision: 0.7021 | Recall: 0.7544 | FPR: 0.1200 | AUC: 0.8550
2025-11-19 10:48:45,433 - INFO - Best model state updated at epoch 360.
2025-11-19 10:48:48,342 - INFO - Epoch 380 | Loss: 1.7328 | F1: 0.7261 | Precision: 0.7065 | Recall: 0.7566 | FPR: 0.1161 | AUC: 0.8594
2025-11-19 10:48:48,343 - INFO - Best model state updated at epoch 380.
2025-11-19 10:48:51,252 - INFO - Epoch 400 | Loss: 1.6248 | F1: 0.7159 | Precision: 0.6935 | Recall: 0.7576 | FPR: 0.1342 | AUC: 0.8528
2025-11-19 10:48:51,253 - INFO - Best model state updated at epoch 400.
2025-11-19 10:48:54,114 - INFO - Epoch 420 | Loss: 1.6225 | F1: 0.7269 | Precision: 0.7111 | Recall: 0.7485 | FPR: 0.1056 | AUC: 0.8523
2025-11-19 10:48:54,114 - INFO - Best model state updated at epoch 420.
2025-11-19 10:48:57,085 - INFO - Epoch 440 | Loss: 1.7183 | F1: 0.7245 | Precision: 0.7080 | Recall: 0.7480 | FPR: 0.1087 | AUC: 0.8512
2025-11-19 10:48:59,950 - INFO - Epoch 460 | Loss: 1.5022 | F1: 0.7273 | Precision: 0.7095 | Recall: 0.7530 | FPR: 0.1102 | AUC: 0.8524
2025-11-19 10:48:59,950 - INFO - Best model state updated at epoch 460.
2025-11-19 10:49:02,880 - INFO - Epoch 480 | Loss: 1.5249 | F1: 0.7279 | Precision: 0.7207 | Recall: 0.7361 | FPR: 0.0892 | AUC: 0.8476
2025-11-19 10:49:05,828 - INFO - Epoch 500 | Loss: 1.3578 | F1: 0.7264 | Precision: 0.7116 | Recall: 0.7462 | FPR: 0.1036 | AUC: 0.8508
2025-11-19 10:49:05,828 - INFO - Best model state updated at epoch 500.
2025-11-19 10:49:08,764 - INFO - Epoch 520 | Loss: 1.3734 | F1: 0.7295 | Precision: 0.7123 | Recall: 0.7540 | FPR: 0.1077 | AUC: 0.8521
2025-11-19 10:49:08,764 - INFO - Best model state updated at epoch 520.
2025-11-19 10:49:11,697 - INFO - Epoch 540 | Loss: 1.2656 | F1: 0.7258 | Precision: 0.7083 | Recall: 0.7511 | FPR: 0.1104 | AUC: 0.8529
2025-11-19 10:49:11,697 - INFO - Best model state updated at epoch 540.
2025-11-19 10:49:14,686 - INFO - Epoch 560 | Loss: 1.2769 | F1: 0.7303 | Precision: 0.7159 | Recall: 0.7493 | FPR: 0.1010 | AUC: 0.8547
2025-11-19 10:49:14,687 - INFO - Best model state updated at epoch 560.
2025-11-19 10:49:17,565 - INFO - Epoch 580 | Loss: 1.3175 | F1: 0.7305 | Precision: 0.7123 | Recall: 0.7569 | FPR: 0.1095 | AUC: 0.8560
2025-11-19 10:49:17,566 - INFO - Best model state updated at epoch 580.
2025-11-19 10:49:20,397 - INFO - Epoch 600 | Loss: 1.1945 | F1: 0.7342 | Precision: 0.7258 | Recall: 0.7439 | FPR: 0.0888 | AUC: 0.8548
2025-11-19 10:49:20,398 - INFO - Best model state updated at epoch 600.
2025-11-19 10:49:23,363 - INFO - Epoch 620 | Loss: 1.0946 | F1: 0.7252 | Precision: 0.7127 | Recall: 0.7411 | FPR: 0.0994 | AUC: 0.8483
2025-11-19 10:49:23,363 - INFO - Best model state updated at epoch 620.
2025-11-19 10:49:26,379 - INFO - Epoch 640 | Loss: 1.2528 | F1: 0.7303 | Precision: 0.7226 | Recall: 0.7391 | FPR: 0.0890 | AUC: 0.8489
2025-11-19 10:49:29,404 - INFO - Epoch 660 | Loss: 1.2201 | F1: 0.7272 | Precision: 0.7151 | Recall: 0.7422 | FPR: 0.0977 | AUC: 0.8491
2025-11-19 10:49:32,371 - INFO - Epoch 680 | Loss: 1.0483 | F1: 0.7273 | Precision: 0.7168 | Recall: 0.7400 | FPR: 0.0948 | AUC: 0.8480
2025-11-19 10:49:32,371 - INFO - Best model state updated at epoch 680.
2025-11-19 10:49:35,360 - INFO - Epoch 700 | Loss: 1.0113 | F1: 0.7298 | Precision: 0.7217 | Recall: 0.7391 | FPR: 0.0898 | AUC: 0.8516
2025-11-19 10:49:35,360 - INFO - Best model state updated at epoch 700.
2025-11-19 10:49:38,178 - INFO - Epoch 720 | Loss: 0.9685 | F1: 0.7314 | Precision: 0.7226 | Recall: 0.7416 | FPR: 0.0904 | AUC: 0.8533
2025-11-19 10:49:38,178 - INFO - Best model state updated at epoch 720.
2025-11-19 10:49:41,208 - INFO - Epoch 740 | Loss: 0.9200 | F1: 0.7295 | Precision: 0.7196 | Recall: 0.7414 | FPR: 0.0929 | AUC: 0.8524
2025-11-19 10:49:41,208 - INFO - Best model state updated at epoch 740.
2025-11-19 10:49:44,309 - INFO - Epoch 760 | Loss: 0.9620 | F1: 0.7294 | Precision: 0.7177 | Recall: 0.7440 | FPR: 0.0963 | AUC: 0.8530
2025-11-19 10:49:47,402 - INFO - Epoch 780 | Loss: 0.9023 | F1: 0.7338 | Precision: 0.7330 | Recall: 0.7347 | FPR: 0.0784 | AUC: 0.8523
2025-11-19 10:49:47,403 - INFO - Best model state updated at epoch 780.
2025-11-19 10:49:50,360 - INFO - Epoch 800 | Loss: 1.0197 | F1: 0.7326 | Precision: 0.7213 | Recall: 0.7465 | FPR: 0.0942 | AUC: 0.8519
2025-11-19 10:49:53,429 - INFO - Epoch 820 | Loss: 0.8932 | F1: 0.7287 | Precision: 0.7215 | Recall: 0.7368 | FPR: 0.0888 | AUC: 0.8504
2025-11-19 10:49:53,429 - INFO - Best model state updated at epoch 820.
2025-11-19 10:49:56,478 - INFO - Epoch 840 | Loss: 0.9099 | F1: 0.7314 | Precision: 0.7236 | Recall: 0.7403 | FPR: 0.0888 | AUC: 0.8505
2025-11-19 10:49:59,429 - INFO - Epoch 860 | Loss: 0.9984 | F1: 0.7260 | Precision: 0.7132 | Recall: 0.7424 | FPR: 0.0997 | AUC: 0.8498
2025-11-19 10:50:02,448 - INFO - Epoch 880 | Loss: 0.8781 | F1: 0.7291 | Precision: 0.7227 | Recall: 0.7362 | FPR: 0.0875 | AUC: 0.8477
2025-11-19 10:50:02,448 - INFO - Best model state updated at epoch 880.
2025-11-19 10:50:05,335 - INFO - Epoch 900 | Loss: 0.8336 | F1: 0.7321 | Precision: 0.7216 | Recall: 0.7447 | FPR: 0.0930 | AUC: 0.8543
2025-11-19 10:50:05,335 - INFO - Best model state updated at epoch 900.
2025-11-19 10:50:08,315 - INFO - Epoch 920 | Loss: 0.7550 | F1: 0.7299 | Precision: 0.7251 | Recall: 0.7350 | FPR: 0.0848 | AUC: 0.8527
2025-11-19 10:50:08,315 - INFO - Best model state updated at epoch 920.
2025-11-19 10:50:11,224 - INFO - Epoch 940 | Loss: 0.6839 | F1: 0.7289 | Precision: 0.7238 | Recall: 0.7343 | FPR: 0.0855 | AUC: 0.8504
2025-11-19 10:50:11,224 - INFO - Best model state updated at epoch 940.
2025-11-19 10:50:14,132 - INFO - Epoch 960 | Loss: 0.8158 | F1: 0.7293 | Precision: 0.7209 | Recall: 0.7390 | FPR: 0.0905 | AUC: 0.8502
2025-11-19 10:50:17,073 - INFO - Epoch 980 | Loss: 0.7385 | F1: 0.7323 | Precision: 0.7271 | Recall: 0.7379 | FPR: 0.0846 | AUC: 0.8497
2025-11-19 10:50:20,045 - INFO - Epoch 1000 | Loss: 0.8003 | F1: 0.7300 | Precision: 0.7256 | Recall: 0.7347 | FPR: 0.0842 | AUC: 0.8493
2025-11-19 10:50:22,903 - INFO - Epoch 1020 | Loss: 0.7511 | F1: 0.7331 | Precision: 0.7224 | Recall: 0.7461 | FPR: 0.0929 | AUC: 0.8563
2025-11-19 10:50:25,981 - INFO - Epoch 1040 | Loss: 0.7175 | F1: 0.7313 | Precision: 0.7262 | Recall: 0.7369 | FPR: 0.0849 | AUC: 0.8513
2025-11-19 10:50:28,909 - INFO - Epoch 1060 | Loss: 0.7759 | F1: 0.7284 | Precision: 0.7210 | Recall: 0.7367 | FPR: 0.0892 | AUC: 0.8493
2025-11-19 10:50:31,859 - INFO - Epoch 1080 | Loss: 0.6942 | F1: 0.7310 | Precision: 0.7222 | Recall: 0.7412 | FPR: 0.0905 | AUC: 0.8543
2025-11-19 10:50:34,743 - INFO - Epoch 1100 | Loss: 0.7774 | F1: 0.7280 | Precision: 0.7184 | Recall: 0.7394 | FPR: 0.0930 | AUC: 0.8483
2025-11-19 10:50:37,712 - INFO - Epoch 1120 | Loss: 0.7051 | F1: 0.7353 | Precision: 0.7287 | Recall: 0.7426 | FPR: 0.0856 | AUC: 0.8554
2025-11-19 10:50:37,713 - INFO - Best model state updated at epoch 1120.
2025-11-19 10:50:40,680 - INFO - Epoch 1140 | Loss: 0.6193 | F1: 0.7336 | Precision: 0.7303 | Recall: 0.7370 | FPR: 0.0815 | AUC: 0.8530
2025-11-19 10:50:40,681 - INFO - Best model state updated at epoch 1140.
2025-11-19 10:50:43,691 - INFO - Epoch 1160 | Loss: 0.6680 | F1: 0.7332 | Precision: 0.7290 | Recall: 0.7377 | FPR: 0.0830 | AUC: 0.8535
2025-11-19 10:50:46,697 - INFO - Epoch 1180 | Loss: 0.5844 | F1: 0.7360 | Precision: 0.7291 | Recall: 0.7437 | FPR: 0.0858 | AUC: 0.8570
2025-11-19 10:50:46,697 - INFO - Best model state updated at epoch 1180.
2025-11-19 10:50:49,738 - INFO - Epoch 1200 | Loss: 0.5218 | F1: 0.7308 | Precision: 0.7272 | Recall: 0.7345 | FPR: 0.0828 | AUC: 0.8504
2025-11-19 10:50:49,739 - INFO - Best model state updated at epoch 1200.
2025-11-19 10:50:52,651 - INFO - Epoch 1220 | Loss: 0.6618 | F1: 0.7302 | Precision: 0.7267 | Recall: 0.7339 | FPR: 0.0829 | AUC: 0.8488
2025-11-19 10:50:55,663 - INFO - Epoch 1240 | Loss: 0.5573 | F1: 0.7298 | Precision: 0.7325 | Recall: 0.7273 | FPR: 0.0754 | AUC: 0.8481
2025-11-19 10:50:58,636 - INFO - Epoch 1260 | Loss: 0.5445 | F1: 0.7303 | Precision: 0.7247 | Recall: 0.7365 | FPR: 0.0859 | AUC: 0.8520
2025-11-19 10:51:01,641 - INFO - Epoch 1280 | Loss: 0.6924 | F1: 0.7296 | Precision: 0.7261 | Recall: 0.7333 | FPR: 0.0831 | AUC: 0.8485
2025-11-19 10:51:04,578 - INFO - Epoch 1300 | Loss: 0.5602 | F1: 0.7302 | Precision: 0.7237 | Recall: 0.7375 | FPR: 0.0872 | AUC: 0.8508
2025-11-19 10:51:07,560 - INFO - Epoch 1320 | Loss: 0.5699 | F1: 0.7305 | Precision: 0.7222 | Recall: 0.7399 | FPR: 0.0898 | AUC: 0.8511
2025-11-19 10:51:10,544 - INFO - Epoch 1340 | Loss: 0.5171 | F1: 0.7358 | Precision: 0.7269 | Recall: 0.7462 | FPR: 0.0890 | AUC: 0.8544
2025-11-19 10:51:10,544 - INFO - Best model state updated at epoch 1340.
2025-11-19 10:51:13,559 - INFO - Epoch 1360 | Loss: 0.6739 | F1: 0.7323 | Precision: 0.7324 | Recall: 0.7321 | FPR: 0.0777 | AUC: 0.8527
2025-11-19 10:51:16,551 - INFO - Epoch 1380 | Loss: 0.6242 | F1: 0.7307 | Precision: 0.7290 | Recall: 0.7326 | FPR: 0.0805 | AUC: 0.8509
2025-11-19 10:51:19,503 - INFO - Epoch 1400 | Loss: 0.5463 | F1: 0.7272 | Precision: 0.7260 | Recall: 0.7284 | FPR: 0.0808 | AUC: 0.8474
2025-11-19 10:51:22,434 - INFO - Epoch 1420 | Loss: 0.6897 | F1: 0.7285 | Precision: 0.7236 | Recall: 0.7337 | FPR: 0.0853 | AUC: 0.8485
2025-11-19 10:51:25,449 - INFO - Epoch 1440 | Loss: 0.5821 | F1: 0.7308 | Precision: 0.7260 | Recall: 0.7359 | FPR: 0.0845 | AUC: 0.8535
2025-11-19 10:51:28,417 - INFO - Epoch 1460 | Loss: 0.5637 | F1: 0.7330 | Precision: 0.7288 | Recall: 0.7376 | FPR: 0.0830 | AUC: 0.8540
2025-11-19 10:51:31,374 - INFO - Epoch 1480 | Loss: 0.5441 | F1: 0.7287 | Precision: 0.7266 | Recall: 0.7309 | FPR: 0.0816 | AUC: 0.8506
2025-11-19 10:51:34,339 - INFO - Epoch 1500 | Loss: 0.6119 | F1: 0.7302 | Precision: 0.7273 | Recall: 0.7333 | FPR: 0.0821 | AUC: 0.8508
2025-11-19 10:51:37,346 - INFO - Epoch 1520 | Loss: 0.4569 | F1: 0.7331 | Precision: 0.7301 | Recall: 0.7363 | FPR: 0.0813 | AUC: 0.8511
2025-11-19 10:51:37,347 - INFO - Best model state updated at epoch 1520.
2025-11-19 10:51:40,359 - INFO - Epoch 1540 | Loss: 0.4920 | F1: 0.7307 | Precision: 0.7259 | Recall: 0.7359 | FPR: 0.0846 | AUC: 0.8511
2025-11-19 10:51:43,451 - INFO - Epoch 1560 | Loss: 0.4426 | F1: 0.7286 | Precision: 0.7256 | Recall: 0.7316 | FPR: 0.0827 | AUC: 0.8490
2025-11-19 10:51:43,452 - INFO - Best model state updated at epoch 1560.
2025-11-19 10:51:46,506 - INFO - Epoch 1580 | Loss: 0.4425 | F1: 0.7327 | Precision: 0.7307 | Recall: 0.7347 | FPR: 0.0801 | AUC: 0.8509
2025-11-19 10:51:46,507 - INFO - Best model state updated at epoch 1580.
2025-11-19 10:51:49,560 - INFO - Epoch 1600 | Loss: 0.6013 | F1: 0.7256 | Precision: 0.7271 | Recall: 0.7242 | FPR: 0.0779 | AUC: 0.8471
2025-11-19 10:51:52,631 - INFO - Epoch 1620 | Loss: 0.4945 | F1: 0.7279 | Precision: 0.7241 | Recall: 0.7320 | FPR: 0.0841 | AUC: 0.8512
2025-11-19 10:51:55,631 - INFO - Epoch 1640 | Loss: 0.4808 | F1: 0.7364 | Precision: 0.7352 | Recall: 0.7375 | FPR: 0.0779 | AUC: 0.8552
2025-11-19 10:51:55,632 - INFO - Best model state updated at epoch 1640.
2025-11-19 10:51:58,597 - INFO - Epoch 1660 | Loss: 0.5349 | F1: 0.7342 | Precision: 0.7267 | Recall: 0.7428 | FPR: 0.0875 | AUC: 0.8526
2025-11-19 10:52:01,611 - INFO - Epoch 1680 | Loss: 0.5073 | F1: 0.7301 | Precision: 0.7226 | Recall: 0.7386 | FPR: 0.0887 | AUC: 0.8529
2025-11-19 10:52:04,662 - INFO - Epoch 1700 | Loss: 0.4519 | F1: 0.7321 | Precision: 0.7335 | Recall: 0.7308 | FPR: 0.0762 | AUC: 0.8520
2025-11-19 10:52:07,663 - INFO - Epoch 1720 | Loss: 0.5105 | F1: 0.7337 | Precision: 0.7286 | Recall: 0.7392 | FPR: 0.0840 | AUC: 0.8501
2025-11-19 10:52:10,676 - INFO - Epoch 1740 | Loss: 0.4476 | F1: 0.7308 | Precision: 0.7264 | Recall: 0.7355 | FPR: 0.0839 | AUC: 0.8494
2025-11-19 10:52:13,741 - INFO - Epoch 1760 | Loss: 0.5118 | F1: 0.7353 | Precision: 0.7320 | Recall: 0.7388 | FPR: 0.0811 | AUC: 0.8531
2025-11-19 10:52:16,697 - INFO - Epoch 1780 | Loss: 0.4931 | F1: 0.7317 | Precision: 0.7277 | Recall: 0.7360 | FPR: 0.0832 | AUC: 0.8508
2025-11-19 10:52:19,715 - INFO - Epoch 1800 | Loss: 0.4027 | F1: 0.7320 | Precision: 0.7277 | Recall: 0.7365 | FPR: 0.0834 | AUC: 0.8519
2025-11-19 10:52:19,716 - INFO - Best model state updated at epoch 1800.
2025-11-19 10:52:22,703 - INFO - Epoch 1820 | Loss: 0.4830 | F1: 0.7373 | Precision: 0.7357 | Recall: 0.7390 | FPR: 0.0782 | AUC: 0.8543
2025-11-19 10:52:22,703 - INFO - Best model state updated at epoch 1820.
2025-11-19 10:52:25,764 - INFO - Epoch 1840 | Loss: 0.3812 | F1: 0.7355 | Precision: 0.7293 | Recall: 0.7423 | FPR: 0.0849 | AUC: 0.8546
2025-11-19 10:52:25,765 - INFO - Best model state updated at epoch 1840.
2025-11-19 10:52:28,749 - INFO - Epoch 1860 | Loss: 0.4353 | F1: 0.7360 | Precision: 0.7351 | Recall: 0.7369 | FPR: 0.0778 | AUC: 0.8559
2025-11-19 10:52:31,698 - INFO - Epoch 1880 | Loss: 0.4464 | F1: 0.7335 | Precision: 0.7286 | Recall: 0.7388 | FPR: 0.0838 | AUC: 0.8531
2025-11-19 10:52:34,686 - INFO - Epoch 1900 | Loss: 0.4245 | F1: 0.7346 | Precision: 0.7342 | Recall: 0.7349 | FPR: 0.0776 | AUC: 0.8535
2025-11-19 10:52:37,746 - INFO - Epoch 1920 | Loss: 0.4527 | F1: 0.7306 | Precision: 0.7303 | Recall: 0.7309 | FPR: 0.0787 | AUC: 0.8486
2025-11-19 10:52:40,730 - INFO - Epoch 1940 | Loss: 0.4500 | F1: 0.7358 | Precision: 0.7372 | Recall: 0.7344 | FPR: 0.0751 | AUC: 0.8537
2025-11-19 10:52:43,663 - INFO - Epoch 1960 | Loss: 0.4993 | F1: 0.7294 | Precision: 0.7347 | Recall: 0.7245 | FPR: 0.0725 | AUC: 0.8490
2025-11-19 10:52:46,695 - INFO - Epoch 1980 | Loss: 0.4172 | F1: 0.7287 | Precision: 0.7304 | Recall: 0.7270 | FPR: 0.0768 | AUC: 0.8458
2025-11-19 10:52:49,655 - INFO - Epoch 2000 | Loss: 0.4296 | F1: 0.7342 | Precision: 0.7308 | Recall: 0.7378 | FPR: 0.0815 | AUC: 0.8552
2025-11-19 10:52:52,500 - INFO - Epoch 2020 | Loss: 0.4004 | F1: 0.7316 | Precision: 0.7315 | Recall: 0.7317 | FPR: 0.0781 | AUC: 0.8505
2025-11-19 10:52:55,395 - INFO - Epoch 2040 | Loss: 0.3894 | F1: 0.7339 | Precision: 0.7332 | Recall: 0.7345 | FPR: 0.0781 | AUC: 0.8502
'''

# 解析日志数据
data = []
lines = log_text.strip().split('\n')
for line in lines:
    if 'Epoch' in line and 'Loss' in line:
        # 提取Epoch、loss和各项指标
        parts = line.split('|')
        epoch = int(parts[0].split('Epoch ')[1].strip())
        loss = float(parts[1].split(': ')[1].strip())
        f1 = float(parts[2].split(': ')[1].strip())
        precision = float(parts[3].split(': ')[1].strip())
        recall = float(parts[4].split(': ')[1].strip())
        fpr = float(parts[5].split(': ')[1].strip())
        auc = float(parts[6].split(': ')[1].strip())

        data.append({
            'Epoch': epoch,
            'Loss': loss,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'FPR': fpr,
            'AUC': auc
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 绘制训练过程图表
plt.figure(figsize=(12, 8))

# 子图1：Loss变化
plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['Loss'], marker='o', color='blue')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 子图2：F1变化
plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['F1'], marker='o', color='green')
plt.title('F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.grid(True)

# 子图3：Precision变化
plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['Precision'], marker='o', color='red')
plt.title('Precision per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.grid(True)

# 子图4：AUC变化
plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['AUC'], marker='o', color='purple')
plt.title('AUC per Epoch')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True)

plt.tight_layout()
plt.savefig('ResGCN_training.png')  # 保存为图片
plt.close()