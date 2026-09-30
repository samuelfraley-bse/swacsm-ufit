# Findings (as of 2026-09-29, after step 08)

**Question:** How should a resource-limited strength program allocate a small wearable data-collection budget?

**Design:** public uLift dataset (15 exercises, one wrist accelerometer). Pool of 23 adults recorded on ≥2 days. A simulated program records N athletes (2-17) for k sets per exercise (1 or 2) on day 1. Testing uses later days only, on the recorded athletes and on 6 athletes who were never recorded. 30 random repeats. Models: random forest (RF), 1-D CNN, LSTM. Metric: set-level macro-F1. Athletes are the unit of analysis (paired Wilcoxon tests; 95% bootstrap CIs across athletes; n = 23).

## RQ1: Breadth. How many athletes should be recorded?
- New-athlete recognition rises steeply at first and then flattens. RF, 1 set: 0.64 (2 athletes) → 0.78 (8) → 0.84 (17).
- Gain per added athlete falls from about 0.04-0.055 (2→4 athletes) to 0.003-0.015 (12→17).
- **About 12 athletes** (CI ~10.5-13.5 across models) reach 90% of the improvement observed between 2 and 17 athletes; 95% takes about 14-16. With 8 athletes a program has about 70-75% of it.
- The curve had not fully plateaued by 17, the most this dataset allows.

## RQ2: Depth. Is a second set per exercise worth it?
- A second set helps: RF +0.02-0.03, CNN +0.02-0.04, LSTM +0.04-0.09 (all p < 0.001).
- **At an equal budget, more athletes beats more sets, every time.** 2N athletes × 1 set beat N athletes × 2 sets in all 12 comparisons (3 models × 4 budgets; +0.025 to +0.070; all p ≤ 0.003; 16-23 of 23 athletes better). For example, 8 athletes × 1 set vs 4 × 2: RF 0.78 vs 0.74; CNN 0.79 vs 0.73.

## RQ3: Does recording the target athlete help them?
- Yes, at every team size. The recorded-vs-new gap shrinks as more athletes are recorded but never closes: CNN 0.29 (2 athletes) → 0.11 (8) → 0.06 (17); RF 0.27 → 0.13 → 0.09 (all p < 1e-5).
- More diverse population data reduces, but does not replace, athlete-specific recordings.

## RQ4: Does a short 2-exercise battery transfer to other exercises?
- No. Recording a new athlete on squat + deadlift, push-up + push press, or random pairs changed recognition of their other 13 exercises by −0.008 to 0.000 across models. Recording all 15 improved it by 0.06-0.13.
- Transfer is not positive even to *similar* lifts. For the RF, recording one lift slightly *hurt* recognition of its confusable partners (e.g. −0.037 at 8 athletes, p = 2e-5); the CNN and LSTM point the same way but are not significant. There is no cheap, generic "athlete signature"; individual information is movement-specific.

## RQ5: Does label-free calibration help?
- The new athlete wears the watch for one session with no labels, and the CNN recalibrates its internal statistics (AdaBN).
- Gain: +0.042 with 2 athletes recorded (p = 5e-7; 22 of 23 improved), +0.027 with 8 (p = 6e-5), not significant with 17 (+0.012, p = 0.07).
- About one extra recorded teammate's worth when data are scarce. It is free, but modest and no substitute for recording.

## Few athletes recorded many times vs many recorded once (step 12)
- Same budget, all 15 exercises, only the number of different people changes. At every budget and in both RF and CNN: many athletes × 1 set > in between > few athletes × many sets.
- **1 athlete × 4 sets vs 4 athletes × 1 set: +0.15 (RF) / +0.16 (CNN), every one of 23 athletes better**, even though the 4 sets span two recording days.
- One number (regression, step 15): **doubling the athletes recorded buys 2.5-4× as much accuracy as doubling each athlete's sets** (RF +0.068 vs +0.016; CNN +0.098 vs +0.039 per doubling).

## Individual athletes and confusions (step 15)
- With 8 athletes recorded, 12 of 23 (RF) and 15 of 23 (CNN) new athletes are still below 0.80; one athlete stays far below the rest.
- The pressing lifts are the most confused: the military press is called correctly 41% of the time (34% called bench press, 20% push press; CNN, 8 athletes).

## Real-time (step 13, poster only)
- Once a set is underway (set start given), the tracker names it within about one rep: accuracy after 2 s is within 0.01-0.03 of the end-of-set accuracy.

*(A "posture is shared, movement is personal" explanation was explored in steps 07 and 15 and removed: it did not hold up feature by feature.)*

## Which model?
- RF is best with very little data (CNN − RF = −0.05 at 2 athletes, p = 0.003); they tie at 8 (p = 0.22); the CNN is best at 17 (+0.04, p = 0.001). The LSTM is worse than the CNN at every size (p < 1e-5).

## Coach takeaways (this dataset)
1. Spend the budget on **more athletes, one set each**, rather than repeated sets from a few.
2. Expect steep gains up to ~8 athletes and diminishing returns after ~12.
3. Record each new athlete on every exercise you want tracked; a quick 2-exercise battery is not a shortcut.
4. Label-free calibration is a free, modest bonus when a program has little data.
5. Pressing variations are the likeliest to be confused.

## Limitations
- 23 adult volunteers (not collegiate athletes; JUCO is the motivating use case only); lab-style circuits of already-segmented sets (real use also needs automatic set detection).
- Team sizes capped at 17; the curve had not plateaued.
- "Depth" is a second set on the same day, not a second day.
- One outlier participant (possibly a different strap placement); the LSTM used a fixed recipe with no tuning.
