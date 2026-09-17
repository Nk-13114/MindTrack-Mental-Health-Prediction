# Dataset Information

MindTrack uses two datasets representing different aspects of mental health assessment.

## PHQ-9 Dataset

The PHQ-9 dataset contains responses to the Patient Health Questionnaire-9.

The questionnaire consists of nine items related to depressive symptoms. Responses are converted into numerical values from 0 to 3 and used to classify depression severity.

### Response Mapping

| Response | Value |
|---|---:|
| Not at all | 0 |
| Several days | 1 |
| More than half the days | 2 |
| Nearly every day | 3 |

The `PHQ_Severity` column is used as the target variable.

## Wearable Dataset

The wearable dataset contains physiological measurements such as:

- Heart rate
- Sleep duration
- Body temperature
- Blood oxygen
- Environmental conditions

The dataset does not contain direct mental-health labels. Therefore, the project creates a `wearable_risk` proxy label using sleep duration and heart-rate patterns.

## Data Privacy

The original CSV datasets are not included in this public repository.

They were used for academic and research purposes during development of the MindTrack project.

To reproduce the implementation, provide the required datasets locally and update the file paths in `MindTrack.py` if necessary.
