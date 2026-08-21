# esp32-s3-llm-0.2b

> **Ourselves, Adwaith Hari and Chandrakiran S. We ran a 0.2 Billion parameter Large Language Model (Transformer) on an ESP32-S3 microcontroller with 8mb psram**

This repository implements a lightweight **0.2B parameter Transformer** capable of executing local, on-device inference on an **ESP32-S3** microcontroller without any cloud APIs or internet connection.

## Hardware Setup and Requirements
* **ESP32-S3 Board:** ESP32-S3 with integrated MicroSD Card slot - **https://robu.in/product/esp32-s3-devkit-esp32-s3-wroom-1-n16r8**
* **MicroSD Card:** Formatted to **FAT32** containing the quantized model bin files (do not store it inside another folder in the sd card, if you're doing so you must specify the folder along with the file bin names)

Also set the Tools > PSRAM > OPI PSRAM
