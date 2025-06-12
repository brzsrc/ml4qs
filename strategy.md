['Accelerometer' 'Linear Accelerometer' 'Gyroscope' 'Magnetometer']
  skill_type    user  sample_number  time  X_Accelerometer  Y_Accelerometer  Z_Accelerometer  X_Linear Accelerometer  Y_Linear Accelerometer  Z_Linear Accelerometer  X_Gyroscope  Y_Gyroscope  Z_Gyroscope  X_Magnetometer  Y_Magnetometer  Z_Magnetometer
0      clear  keeley              0  0.00         6.648724         0.605491         7.740404                     NaN                     NaN                     NaN          NaN          NaN          NaN             NaN             NaN             NaN
1      clear  keeley              0  0.01         6.541696         0.579295         7.732919                0.444033                0.243212                0.116630     0.133790     0.341643     0.017793       42.958206      -11.092033       26.174484
2      clear  keeley              0  0.02         6.260880         0.532742         7.726632                0.207950                0.155021                0.058808     0.120539     0.351478     0.037069       42.744019      -11.037231       26.213936
3      clear  keeley              0  0.03         5.981861         0.634680         7.838001               -0.048183                0.228001                0.128125     0.126878     0.356774     0.061455       42.706146      -11.064934       26.381683
4      clear  keeley              0  0.04         5.687124         0.681383         8.030051               -0.309722                0.296517                0.291339     0.142171     0.297570     0.090196       42.835358      -10.900688       26.257538

## init dataset:
(133468, 16)
skill_type                   0
user                         0
sample_number                0
time                         0
X_Accelerometer            954
Y_Accelerometer            954
Z_Accelerometer            954
X_Linear Accelerometer    1078
Y_Linear Accelerometer    1078
Z_Linear Accelerometer    1078
X_Gyroscope               1077
Y_Gyroscope               1077
Z_Gyroscope               1077
X_Magnetometer            1077
Y_Magnetometer            1077
Z_Magnetometer            1077

## after trimmed:
(78328, 16)

  null_values:
  skill_type                  0
  user                        0
  sample_number               0
  time                        0
  X_Accelerometer           172
  Y_Accelerometer           172
  Z_Accelerometer           172
  X_Linear Accelerometer    212
  Y_Linear Accelerometer    212
  Z_Linear Accelerometer    212
  X_Gyroscope               212
  Y_Gyroscope               212
  Z_Gyroscope               212
  X_Magnetometer            212
  Y_Magnetometer            212
  Z_Magnetometer            212

## impute values by KNN:
n_neighbor  = 10

## after sliding window:
(1527, 40)

snd_try_50_25: trim -> impute -> sliding window - 0.66
applied_kalman_50_25: trim -> impute -> kalman -> sliding window - 0.65
snd_try_50_25_2: impute -> trim -> sliding window - 0.66
applied_kalman_50_25_2: impute -> kalman -> trim -> sliding window - 0.67

snd_try_100_50: trim -> impute -> sliding window - 0.69
applied_kalman_100_50: trim -> impute -> kalman -> sliding window - 0.72
snd_try_100_50_2: impute -> trim -> sliding window - 0.71
applied_kalman_100_50_2: impute -> kalman -> trim -> sliding window - 0.73