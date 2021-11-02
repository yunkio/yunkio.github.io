---
date: 2021-11-02
title: "[Paper Review] Multi-Sensor Fault Detection, Identification, Isolation and Health Forecasting for Autonomous Vehicles"
categories: 
  - Paper Review
tags: 
  - 머신러닝
  - Fault Detection
  - Forecasting
  - Health Index
  - 논문 리뷰
toc: true  
toc_sticky: true 
---

# Paper contents

Multi-Sensor Fault Detection, Identification, Isolation and Health Forecasting for Autonomous Vehicles

Saeid Safavi, Mohammad Amin Safavi, Hossein Hamid, Saber Fallah

Sensors, 2021.

https://www.mdpi.com/1424-8220/21/7/2547

## 0. Abstract

자율주행 연구에는 주행의 신뢰성과 정확성이 매우 중요하다. 센서의 결함으로 인해 고장이 발생하는 경우가 많으며 치명적인 결과로 이어질 수 있다. 그러므로 가능한 조기에 문제를 예측하는 것이 중요하다. 이를 위해 다중 센서 시스템에 다중 fault를 예측, 식별, 감지하기 위한 아키텍쳐를 제안한다. 두 개의 구분되는 딥 뉴럴 네트워크를 사용하며 좋은 성능을 얻었다. 또 모델의 결과를 활용해 **건강 지수** *health index*를 도입하고, 이를 예측하는 네트워크를 만들었다.

## 1. Introduction

센서 모니터링 시스템의 목적은 Fault가 있는 센서를 감지, 격리 및 식별하며 센서의 성능과 신뢰성을 예측하는 것이다. Fault는 크게 sensor fault, actuator fault, part or process fault로 분류된다. **Sensor fault**는 입력 모듈의 오류, **actuator fault**는 출력 모듈의 오류를 의미한다. 이러한 위험을 통제하기 위한 방법으로는 시스템에 결함이 있는지 감지해내는 **fault detection**, 어떤 센서가 문제인지 찾아내는 **fault isolation**, 센서가 고장난 원인을 규명하는 **fault identification**, 그리고 센서의 현재 상태와 미래 상태를 보여주는 **sensor health forecasting strategy**가 있다.

## 2. Dataset Description

## 3. System Description

### 3.1. Sensor Fault Detection

### 3.2. Sensor Fault Identification and Isolation

#### 3.2.1. Feature Extraction

#### 3.2.2. Fault Isolation and Identification Based on Multi-Class DNN

### 3.3. Sensor Health Forecasting

#### 3.3.1. Health Index Definition

#### 3.3.2. Sensor Health Forecasting Strategy

## 4. Experimental Results and Discussion

### 4.1. Sensor Fault Detection

### 4.2. Sensor Fault Identification and Isolation

### 4.3. Sensor Health Forecasting

#### 4.3.1. Performance Measure

#### 4.3.2. Prediction Results

## 5. Conclusions


WIP