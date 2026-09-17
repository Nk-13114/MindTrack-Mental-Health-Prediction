# MindTrack – Dual-Modal Mental Health Risk Prediction System

## Overview

MindTrack is a machine learning-based dual-modal system designed to assess mental health risk by combining subjective questionnaire data with objective physiological data.

The system uses two different data sources:

- **PHQ-9 questionnaire data** – represents subjective psychological information.
- **Wearable physiological data** – represents objective physiological information such as heart rate, sleep duration, body temperature, and blood oxygen.

The two data streams are processed independently using machine learning models and their predictions are combined to provide an overall assessment.

## System Architecture

```text
                 MindTrack
                     |
          -----------------------
          |                     |
      PHQ-9 Data          Wearable Data
          |                     |
   Preprocessing          Preprocessing
          |                     |
        SMOTE                  SMOTE
          |                     |
       TabNet                 TabNet
          |                     |
   PHQ-9 Prediction      Wearable Prediction
          |                     |
          --------- Fusion -----
                     |
              Final Assessment
