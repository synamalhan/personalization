# On-Device Personalization: iOS Build Plan


## Concept and what it shows

Build an iOS app that learns new image classes from a handful of photos, entirely on the phone. A frozen, optimized Core ML backbone turns each photo into a vector; a small classifier head is fit on the device in seconds; and the app stores only those vectors, not the photos. You teach it your own objects, and nothing leaves the device.

The ML study around it is the point. You measure how many photos a new class needs, which head and which backbone work best, whether adding classes makes the old ones worse, whether the app can say "unknown", and what each step costs on the phone.

When it is done you should be able to say, with numbers behind each line:

- I compared two backbones and two classifier heads for few-shot learning and report accuracy against the number of photos, with confidence intervals over many random trials.
- I measured forgetting: how much accuracy on old classes drops as new ones are added, and how storing embeddings fixes it.
- I built open-set rejection and measured how well the app tells unknown objects from taught ones.
- I checked whether quantizing the backbone changes few-shot accuracy and speed.
- I shipped it in an iOS app where a new class is learned in a measured number of seconds, with no data leaving the phone.
- I tested it on my own photos taken across different days and lighting, not only on a public dataset.

This shows modeling, evaluation, and product work in one project, and it is built on the on-device and privacy themes an Apple ML role cares about.

## Architecture

&#91;embedded content: on-device personalization · offline prep and phone pipeline\]

Offline, a laptop script prepares the backbones and runs the experiments on public datasets. On the phone, a photo goes through the frozen backbone to an embedding. Teaching stores that vector, and the head is refit from the stored vectors.

## ML design

The backbone is frozen, so "learning" a class means fitting a small head on stored embeddings. That is cheap enough to do on a phone and easy to evaluate.

- **Backbones:** compare a small convolutional network (MobileNetV3-class) with a small self-supervised vision transformer (a DINOv2-small-class model). Check each model's license and weight terms before use. The question is how much few-shot accuracy each buys against its on-device cost.
- **Heads:** (1) nearest class mean with cosine distance, which needs no training; (2) a logistic-regression head fit on the device with a few optimizer steps. Start with these two; a fine-tuned last layer is a stretch comparison.
- **Storing exemplars:** keep only embeddings (at fp16), never the photos. Refit the head from all stored embeddings each time a class is added, so old classes are replayed and not forgotten. Compare against a no-replay version to measure forgetting.
- **Unknown detection:** reject an image when its distance to every class is above a threshold set from the taught classes' own spread. Measure how well this separates unseen objects from taught ones.
- **Calibration:** report confidence calibration (ECE) for each head, because a head that is sure and wrong is worse than one that says unknown.
- **Quantization:** convert each backbone at fp16 and int8 and measure whether few-shot accuracy and latency change. This is where the optimization work from the earlier benchmark idea fits, as one experiment instead of the whole project.
- **Public data:** use fine-grained public datasets (for example Oxford Flowers 102 and Oxford-IIIT Pets; confirm each license) to simulate users who teach a few new classes. Draw many random class sets and seeds so the results are not one lucky draw.
- **Core ML updatable models:** Core ML has some support for on-device training. Check in a short spike whether it fits a head on your target OS version; the plan does not depend on it, because the head is small enough to fit in Swift with Accelerate.

## Evaluation plan

Eight experiments answer the questions in the concept. Run the first six offline on a laptop and the last two on the phone.

| # | Question | Design |
| --- | --- | --- |
| 1 | How many photos does a new class need? | 1, 3, 5, 10, and 20 shots per class; many random class sets and seeds |
| 2 | Which head is better? | Nearest class mean vs logistic head, same splits |
| 3 | Which backbone is better, and at what cost? | Convolutional vs transformer: accuracy against measured latency |
| 4 | Does quantization hurt few-shot learning? | Embed with fp16 and int8 backbones; compare accuracy and latency |
| 5 | Does adding classes break old ones? | Add classes one at a time, with and without replay; track accuracy on earlier classes |
| 6 | Can it say unknown? | AUROC separating unseen classes from taught ones, plus calibration (ECE) |
| 7 | What does it cost on the phone? | Time to embed a photo and to add a class, and memory, on a real device |
| 8 | Does it work on real photos? | Five personal classes, 20 photos each taken on different days and in different light; train on some days, test on the others |

**Metrics:** top-1 accuracy and macro-F1, with 95% confidence intervals from the repeated trials; AUROC for unknown detection; ECE for calibration; median and 95th-percentile latency; seconds to add a class.

**Reporting rules:** state the number of trials, show intervals and not only means, keep the worst cases in the results, and say plainly that the public datasets simulate users and the real-photo test is small. Write the on-device measurement protocol (warm-up, repeats, thermal state, a real device) before measuring, as in any benchmark.

## App design

The app is a small, clean tool with four screens. Keep the UI simple and spend the effort on the learning code and its measurements.

- **Teach:** pick a class name, capture or choose 3 to 20 photos, and watch the head update. Show the time it took.
- **Recognize:** a live camera view that shows the predicted class with its confidence, or "unknown" when the image is too far from everything taught.
- **Library:** every taught class with its example count, and a button to delete one. Deleting a class removes its stored embeddings and refits the head, which doubles as a clear privacy demo.
- **Settings:** the unknown threshold, the backbone choice, and a switch for whether photos are deleted right after embedding (default on).

**Code structure.** A Swift package, `PersonalizeCore`, holds the head, the exemplar store, and the open-set logic with no UI, so it can be unit tested and compared with the Python results on the same embeddings.

**Storage.** Embeddings and class names live in a local store. The photos are not kept unless the user turns that on.

**Privacy.** No network calls, no analytics, and no cloud models. The README should state this and the code should make it checkable, for example by not requesting any network entitlement beyond what the build needs.

## Timeline

&#91;embedded content: build plan · 11 items, 1 gate\]

The plan runs from Nov 19 to Dec 17. The gate on Dec 8 is the point where the phone's embeddings and predictions are checked against the offline results; real-use testing does not start until it passes. If time runs short, polish and the demo are cut first.

## Testing, risks, and what to cut

**Testing.**

- Unit-test `PersonalizeCore` in Swift: the head must give the same predictions as the Python reference on a fixed set of saved embeddings, within a small tolerance.
- Check the converted Core ML backbone against the original on a few hundred images (embedding cosine similarity close to 1 for fp16, lower but measured for int8).
- Test deleting a class: its embeddings are gone and predictions for the others are unchanged except for the refit.
- Test the empty and one-class states so the app does not crash or claim confidence.

**Risks.**

| Risk | What to do |
| --- | --- |
| Few-shot accuracy on real photos is much lower than on the public datasets | Report it. That gap is a finding, and the held-out-by-day split is what makes it honest |
| int8 conversion shifts embeddings enough to hurt accuracy | Keep fp16 as the default and report int8 as measured |
| Offline and on-device results disagree | Treat it as a bug first (preprocessing, color order, resize); the gate milestone exists to catch this early |
| Timing is noisy on a phone | Use a real device, warm up, repeat, and report median and 95th percentile with the thermal state |
| Photos or weights carry license limits | Check licenses before publishing anything, and keep personal photos out of the repo |

**Cut order if time runs short.** Drop polish and the demo video first, then the transformer backbone comparison, then the forgetting experiment's extra variants. Never cut the offline-versus-on-device check or the real-photo test; those are what make the project credible.

## Repo layout, README checklist, and open questions

**Suggested layout.**

- `ios/` holds the Xcode project and the `PersonalizeCore` Swift package.
- `research/` holds the Python experiments, with one script per experiment and a seeded config.
- `models/` holds conversion scripts (not the weights, if their licenses forbid redistribution).
- `results/` holds raw CSVs, the figures made from them, and the measurement protocol.
- `docs/` holds the architecture diagram and screenshots.

**README checklist.**

- [ ] One-paragraph summary and a screenshot or short clip of teaching a class
- [ ] Architecture diagram
- [ ] The claims, each with its number, interval, and the experiment that produced it
- [ ] On-device measurement protocol and the device used
- [ ] Limitations: simulated users, small real-photo test, one device
- [ ] How to reproduce the offline results with one command
- [ ] Privacy statement: what is stored, what is never sent
- [ ] Licenses for datasets and model weights

**Open questions.**

1. Do you have a Mac with a recent Xcode and an iPhone to run on? The on-device results need a real device, not only the simulator.
2. Which dataset licenses and backbone weight terms allow what you plan to publish? I have not verified these.
3. Does this project replace Sleep Pattern Analysis as #9 in the top ten, or sit beside it? It is a stronger match for an ML role, but the Sleep repo is already pushed.
4. The earlier on-device benchmark plan is now folded in as experiment 4. Retire that doc, or keep it as a reference?
5. Check the Core ML and coremltools details with a short spike before committing to them; I wrote them from memory and have not run them.
