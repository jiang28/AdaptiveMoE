# MoE-Climate Dataset for Spatiotemporal Modeling

This repository provides access to the **MoE-Climate Dataset**, a high-resolution, gridded climate dataset designed for multimodal modeling of extreme weather events. The dataset captures a wide array of atmospheric variables over a critical 9-day period in **South Florida** during the landfall of a hurricane (September 23, 2022 to October 2, 2022), enabling research in precipitation forecasting, climate simulation, and multimodal fusion with Mixture-of-Experts (MoE) frameworks.

---

## 📅 Temporal Coverage
- **Time range:** 2022/09/23 00:00 to 2022/10/02 00:00  
- **Duration:** 9 days  
- **Resolution:** Hourly  
- **Total timestamps:** 216 hours  

---

## 📍 Spatial Coverage
- **Region:** South Florida  
- **Grid Size:** 100 × 100  
- **Resolution per cell:** 3 km × 3 km (approx.)

---

## 🌦️ Key Climate Variables

| Feature                     | File Path                                    |
|----------------------------|----------------------------------------------|
| Precipitation rate         | `Feature grouping/Moisture/83.csv`          |
| Total precipitation        | `Feature grouping/Moisture/84.csv`          |
| Cloud cover                | `Feature grouping/Moisture/83.csv`          |
| Wind speed                 | `Feature grouping/Momentum/9.csv`           |
| Temperature                | `Feature grouping/Temperature/64.csv`       |
| Humidity                   | `Feature grouping/Moisture/143.csv`         |

---

## 🧪 DataFrames (Selected Features)

| DataFrame | Description                                        |
|-----------|----------------------------------------------------|
| df1       | Precipitation rate                                 |
| df2       | Cloud cover                                        |
| df3       | Plant canopy surface water                         |
| df4       | Simulated Brightness Temperature                   |
| df5       | Wind speed (gust)                                  |
| df6       | 2 metre temperature                                |
| df8       | Pressure: Cloud Base                               |
| df9       | U component of wind                                |
| df10      | V component of wind                                |
| df11      | Dew point temperature                              |
| df12      | Moisture availability                              |
| df13      | Total precipitation                                |
| df15      | Low cloud cover                                    |
| df16      | Medium cloud cover                                 |
| df17      | Pressure: Cloud Top                                |
| df18      | Relative humidity                                  |
| df19      | Simulated BT for GOES 12                           |
| df20      | Simulated BT for GOES 11 Channel 3                 |
| df21      | Simulated BT for GOES 11 Channel 4                 |

---

## 🔬 Feature Categories

| Category                    | DataFrames     | Variable Index Example | Description                                                                 |
|----------------------------|----------------|--------------------------|-----------------------------------------------------------------------------|
| **1. Momentum**            | df5, df9, df10 | 9, 10, 11                | Wind components and gusts                                                  |
| **2. Temperature**         | df6, df11      | 21, 64, 71               | Surface and dew point temperatures                                          |
| **3. Mass**                | df8, df17      | 122, 123                 | Pressure at cloud layers                                                    |
| **4. Moisture**            | df1, df3, df12, df13, df18 | 66, 67, 83, 84       | Water vapor, total precipitation, humidity                                  |
| **5. Cloud**               | df2, df15, df16| 115–118                  | Low and medium cloud covers                                                 |
| **6. Short-wave Radiation**| None           | —                        | Solar radiation (files to be added)                                         |
| **7. Long-wave Radiation** | df4, df19–21   | 170–173                  | Outgoing longwave radiation, simulated satellite readings                   |
| **8. Forecast Radar Imagery** | None        | —                        | 2D radar images for landscape/weather visualization                         |
| **9. Thermodynamic Stability Indices** | None | —                    | Indices predicting atmospheric instability and thunderstorm likelihood       |
| **10. Hydrology**          | df7, df14      | 88, 89                   | Water runoff, soil moisture content                                         |

---

## 📂 Folder Structure
