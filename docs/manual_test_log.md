# Manual Test Log: DDoS Attack Detection Dashboard

Tester: ______________________    Date: ______________
Python 3.12.3, Windows. Start the app with `streamlit run app.py` (virtual environment active).
Automated tests: `python -m pytest tests -v` gave 64 passed. Date: ______________

Expected values come from the evaluation scripts (`artifacts\final_evaluation.json`,
`model_metadata.json`, `sample_profile.json`, `raw_inspection.json`).
Result: write PASS or FAIL. Evidence: the screenshot file name. For any FAIL,
describe what happened under "Defects" at the end.

## 1. Start-up and Page A, Dashboard Overview

| ID | Steps | Expected | Result | Evidence |
|---|---|---|---|---|
| A1 | Run `streamlit run app.py` | Browser opens at localhost:8501. Sidebar shows "Model artifacts loaded" and a training time of 2026-10-27T12:02:27 (UTC). | | |
| A2 | Overview, section 1 | Raw records 10,197,998; benign 42,918; attack 10,155,080; benign share 0.42%; sample 20,000 (4,000 benign, 16,000 attack); input features 74 / 7 | | |
| A3 | Section 2, full model | Accuracy 98.2%, precision 99.6%, recall 98.2%, F1 98.9%, ROC-AUC 0.982, false-positive rate 1.45%. Matrix 1971 / 29 / 146 / 7854. NetBIOS row says "NO, never seen". | | |
| A4 | Section 2, compact model | Accuracy 96.2%, precision 99.5%, recall 95.6%, F1 97.5%, ROC-AUC 0.993, false-positive rate 1.80%. Matrix 1964 / 36 / 349 / 7651. NetBIOS detected 80.8%. | | |
| A5 | Section 3, both schemas | 6 models (full) and 5 models (compact). "yes" in the saved-model column on exactly one row per schema: Decision Tree (depth 4) for full, Logistic Regression for compact. | | |
| A6 | Section 4 | A note that no predictions are made on this page | | |
| A7 | **No-model state.** Stop nothing. Run `Rename-Item artifacts\model_metadata.json model_metadata.json.bak`, refresh the page, then rename it back and refresh again. | While renamed: red "No trained model found" with setup commands, and no crash. After renaming back: normal page. | | |

## 2. Page B, Dataset Explorer

| ID | Steps | Expected | Result | Evidence |
|---|---|---|---|---|
| B1 | Working sample, section 1 | 20,000 rows, 92 columns, 83 / 9 numeric / text, 155 rows identical to another row | | |
| B2 | Section 3 | Label column `label_original`: Syn 8,000; BENIGN 4,000; DrDoS_LDAP 4,000; LDAP 3,000; NetBIOS 1,000. Binary split 4,000 / 16,000, 0 unusable. | | |
| B3 | Section 4 | 2 columns, 1,024 rows (5.12%). Flow Bytes/s: 531 missing, 493 infinite. Flow Packets/s: 1,024 infinite. | | |
| B4 | Section 5, pick Flow Duration | Histogram with attack mass near 0 on the log axis and benign mass spread out | | |
| B5 | Tick "random rows", move the slider | Table changes without error | | |
| B6 | Section 6, download all three CSVs, tick the cleaned-copy box and download it | Files saved. The cleaned copy has blank cells where infinity was. | | |
| B7 | Tab "Raw CIC-DDoS2019 files" | 4 files, 10,197,998 rows, 42,918 benign. Only 03-11/Syn.csv has duplicates (407,315). | | |
| B8 | Demo file `demo_test_day_with_labels.csv` | 400 rows, 79 columns, 100 of each label | | |
| B9 | Demo files `demo_empty.csv`, `demo_header_only.csv` | A clear error message, no crash | | |
| B10 | Upload any CSV from `data\processed\demo` | Profiles like the demo file. The original file is unchanged. | | |
| B11 | After B6 to B10, run `python src\dataset_explorer.py` | Ends with "Data folder unchanged" and ALL EXPLORER CHECKS PASSED | | |

## 3. Page C, Manual Detection

| ID | Steps | Expected | Result | Evidence |
|---|---|---|---|---|
| C1 | Open the page and read it | A notice that the values are demonstration inputs, and that the output is a model prediction and not proof of an attack | | |
| C2 | Defaults (TCP, 1500 ms, 850 packets, 124000 bytes, ACK 1) | Sent to the model: 1,500,000 µs, 566.67 packets/s, 82,666.67 bytes/s | | |
| C3 | Press DETECT | A result banner, an attack probability, a summary sentence, a "values you entered" table and an "exact row sent to the model" table | | |
| C4 | Choose UDP, press DETECT | A result appears | | |
| C5 | Set ACK to 0, press DETECT | A result appears | | |
| C6 | Duration 0, press DETECT | Packets/s and bytes/s show "undefined". A warning says 603 of the 634 zero-duration training flows were attacks. | | |
| C7 | Packets 1, bytes 70000, press DETECT | Error: bytes cannot exceed 65,535 per packet. No prediction. | | |
| C8 | Protocol "Other (not supported)", press DETECT | A warning that the protocol is not supported. No prediction. | | |
| C9 | Duration 80000000, packets 850, bytes 124000, press DETECT | A prediction, plus a warning listing Flow Duration and Flow Packets/s as outside the training range | | |
| C10 | **NEW:** Duration 86400000 (the largest the form accepts) | Accepted, with an out-of-range warning | | |
| C11 | **NEW:** Try entering packets 0 and a negative duration | Each is blocked by the form or rejected with a message. No prediction in either case. | | |
| C12 | **NEW:** Set bytes to 0 with packets 2, press DETECT | A result appears (0 bytes is allowed) | | |
| C13 | Expand "How this prediction was made, and how far to trust it" | Mentions the test-day recall and false-positive rate, the NetBIOS detection, and that it is not proof of an attack | | |

## 4. Page D, CSV Prediction

Demo files are in `data\processed\demo`. Create them with `python src\make_demo_csv.py`.

| ID | Steps | Expected | Result | Evidence |
|---|---|---|---|---|
| D1 | Upload `demo_test_day_with_labels.csv` | File check: 400 rows, 79 columns, 74 / 74 required, 5 ignored. Summary: 101 BENIGN, 299 DDoS ATTACK, 74.8%. | | |
| D2 | Same file, section 4, do not tick the box | A label table (Syn, BENIGN, NetBIOS, LDAP at 100 each). No metrics shown yet. | | |
| D3 | Tick "These labels are correct" | Evaluation on 400 rows (100 benign, 300 attack). Matrix 98 / 2 / 3 / 297. Accuracy 98.8%, precision 99.3%, recall 99.0%, false-positive rate 2.00%. | | |
| D4 | Section 3, switch the radio to "Predicted ATTACK only", then tick the sort box | The table shows only attacks, sorted by probability | | |
| D5 | Click "Download all predictions as CSV" | A file `ddos_predictions.csv` is saved. Check it with the command in the notes below. | | |
| D6 | **NEW:** Upload `demo_test_day_no_labels.csv` | 78 columns, 4 ignored, 101 BENIGN and 299 ATTACK. Section 4 says no label column was found. | | |
| D7 | **NEW:** Upload `demo_missing_columns.csv` | Red message listing Protocol and Flow Duration. No predictions. | | |
| D8 | **NEW:** Upload `demo_text_value.csv` | "Non-numeric values found in column(s): ['Total Fwd Packets']" | | |
| D9 | **NEW:** Upload `demo_unseen_protocol.csv` | A warning about 5 rows. Predictions still shown. | | |
| D10 | **NEW:** Upload `demo_empty.csv`, then `demo_header_only.csv` | "The file is empty", then "... no data rows". No crash. | | |
| D11 | **NEW:** Try uploading a non-CSV file, such as a `.txt` or `.png` | The uploader refuses it | | |
| D12 | **NEW:** Upload a CSV that is not from this project, such as a small spreadsheet export | Red message listing the 74 missing columns. No crash. | | |

Download check for D5 (should print `(400, 83)`):

    python -c "import pandas as pd, pathlib; d=pd.read_csv(pathlib.Path.home()/'Downloads'/'ddos_predictions.csv'); print(d.shape)"

## 5. Page E, Model Performance

| ID | Steps | Expected | Result | Evidence |
|---|---|---|---|---|
| E1 | Full model, section 1 | Seven metrics, all six count-based rows say "matches". Values 98.25%, 99.63%, 98.17%, 98.90%, 1.45%, 98.36%, AUC 0.982. | | |
| E2 | Section 2 | Matrix 1971 / 29 / 146 / 7854 and the four plain-language lines with the same counts | | |
| E3 | Section 3 | Three evaluation methods side by side. Tree d4: 99.0 / 99.3 / 98.4. Forest: 99.9 / 99.9 / 93.0. | | |
| E4 | Section 4 | Tree d4 detects 99.9% of NetBIOS; the other full-schema models 0.0 to 0.1% | | |
| E5 | **NEW:** Section 5 and the "tree's decision rules" expander | A bar chart led by Flow IAT Std and URG Flag Count, and a 4-level rule list ending in BENIGN / DDoS ATTACK | | |
| E6 | **NEW:** Section 6, open Sensitivity, Figures and Limitations | A sensitivity table (trees at 100%), three images, and a limitations list mentioning NetBIOS and the 2,582 repeated test rows | | |
| E7 | Compact model, sections 1, 3, 5 | Matrix 1964 / 36 / 349 / 7651. No random-split numbers (n/a). A signed coefficient chart. No decision-rules expander. | | |

## 6. Restart and consistency

| ID | Steps | Expected | Result | Evidence |
|---|---|---|---|---|
| R1 | Press Ctrl+C to stop the app. Close the terminal and VS Code. Reopen the project, activate `.venv`, run `streamlit run app.py`. | The app starts without training anything. Overview numbers equal A2 to A4. | | |
| R2 | Run `python src\predict.py` | ALL RELOAD CHECKS PASSED | | |
| R3 | Run `python -m pytest tests -v` | 64 passed | | |
| R4 | Compare Page A with `artifacts\model_metadata.json` and Page E with `artifacts\final_evaluation.json` | Counts are identical | | |

## Defects found

| ID | What happened | Fixed? |
|---|---|---|
| | | |

## Summary

Rows tested: ____ Passed: ____ Failed: ____