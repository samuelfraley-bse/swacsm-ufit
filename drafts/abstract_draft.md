# SWACSM 2026 abstract — working draft (to be revised by the author)

Status: numbers from steps 04-06 (30 repeats each). Still pending: paired model comparisons (CNN vs RF p-value), step 06b (label-free calibration), step 08 (edge). Check against the 2026 SWACSM template for headers and word limit. No brand names.

---

**How Much Labeled Data Does a Wrist Accelerometer Need to Recognize Resistance Exercises for a New Team?**

**PURPOSE:** Wrist-worn accelerometers can automatically log resistance training, but a program adopting one must first record labeled examples, and high-turnover programs repeatedly onboard new athletes. We quantified how many athletes and sets must be recorded before exercise recognition stabilizes, and whether athletes who were never recorded are recognized as well as those who were.

**METHODS:** Secondary analysis of a public dataset of 35 adults performing 15 exercises (2,355 sets) with a single wrist-worn 3-axis accelerometer. Twenty-three participants recorded on at least 2 days were included. In each of 30 random draws, 6 participants were held out as "new athletes" and teams of 2 to 17 were formed from the rest. Models were trained on 1 or 2 sets per exercise from each team member's first day and tested on later days only. We compared a random forest on hand-crafted features, a 1-D convolutional neural network (CNN) and a long short-term memory network (LSTM). We also tested whether recording a new athlete on only 2 exercises improved recognition of their other 13. The primary outcome was set-level macro-F1. Participants were the unit of analysis (paired Wilcoxon tests; 95% bootstrap CIs across participants).

**RESULTS:** For never-recorded athletes, random-forest macro-F1 rose from 0.64 (95% CI 0.60-0.67) with 2 recorded teammates to 0.78 (0.74-0.82) with 8 and 0.84 (0.80-0.87) with 17 (+0.14 from 2 to 8, +0.06 from 8 to 17; both p < 0.001). A second set per exercise added little (+0.02, p < 0.001). Recorded athletes scored higher than never-recorded athletes (0.94 vs 0.86 with 17 recorded; p < 0.001). The CNN surpassed the random forest only with larger teams (0.90 vs 0.86 with 17 recorded [p = ?]); the LSTM trailed both (0.85). Recording a new athlete on 2 exercises did not improve recognition of their other exercises (change −0.008 to 0.000 across models), whereas recording all 15 improved it by 0.06-0.09. Most remaining errors involved bench, military and push press, lifts with similar forearm orientation relative to gravity.

**CONCLUSION:** In this dataset, recognition of new athletes improved most over the first ~8 recorded athletes, and one recorded set per exercise captured most of the benefit. The advantage of recording an athlete was exercise-specific: a brief 2-exercise onboarding did not substitute for recording each exercise. [Label-free calibration / on-device result, if positive.] These estimates can guide how programs budget data collection when adopting wearable exercise tracking.
