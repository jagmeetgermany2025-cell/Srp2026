# Selective betting under the winner's curse

**Short version:** we test a counterintuitive prediction — that being *more*
selective about which football bets you place can make your returns *worse*,
not better. On 162,000 matches, it does.

---

## The idea

Imagine an oil field of unknown value. Ten companies each estimate what it's
worth and bid. Whoever wins the auction is, almost by definition, the company
that *most overestimated* the field. Winning is bad news about your own
estimate. Economists call this the **winner's curse**.

Now the betting version.

You build a model that predicts football results. For most matches it roughly
agrees with the bookmaker's odds. Occasionally it disagrees and says *"the
bookmaker has this at 5.0, but I think it should be 4.0 — that's good value."*
So you bet.

Here's the problem. Your model has errors. When does it say a bet is good
value? Exactly when its error happens to point in the direction of "the odds
are too long." **You are not selecting matches where you have an edge. You are
selecting matches where your model is most optimistic** — and optimism and
genuine edge look identical from the inside.

This leads to a prediction that sounds wrong:

> The stricter your filter, the worse your returns should get.

Normally, being pickier should help. But if the filter mostly selects your own
noise, then a stricter filter selects *more extreme* noise, and returns fall.

That's what this project tests.

---

## How you'd bet on this (the setup)

For each match the bookmaker publishes odds like `2.50`. That means: bet £1, get
£2.50 back if you're right (£1.50 profit), lose your £1 if you're wrong.

Odds of 2.50 imply a probability of 1 ÷ 2.50 = **40%**.

Add up the implied probabilities of all three outcomes (home win, draw, away
win) and you get about **107%**, not 100%. That extra 7% is the bookmaker's
built-in profit margin, called the *overround* or *vig*. Removing it to recover
the "fair" probabilities is called **de-vigging**.

So: our model produces a probability, the odds produce a probability, and we bet
when ours is enough higher than theirs. "Enough" is the **threshold**, written
τ (tau). τ = 5% means *only bet when the model expects at least a 5% profit per
£1 staked*.

The whole study is: **what happens to returns as you turn τ up?**

---

## ⚠️ Read this before looking at the numbers

**Every return below is negative. That is expected and it is not the finding.**

The bookmaker's margin is about 6.8%. You start every bet 6.8% behind. A model
with no real edge should lose roughly that much, and ours does.

**The finding is about the *slope*, not the *level*.** We're not claiming a
profitable strategy exists. We're asking whether returns go *down* as the filter
gets stricter — because that's the fingerprint of the winner's curse, and if
it's there, it means EV filters have a systematic flaw that nobody corrects for.

---

## What we found

Betting flat £1 stakes on 48,622 test matches:

| Filter strength (τ) | Bets placed | Return |
| --- | ---: | ---: |
| 0% — bet on anything with positive value | 4,632 | **−10.8%** |
| 1% | 2,651 | −12.9% |
| 2% | 1,962 | −18.3% |
| 3% | 1,706 | −21.2% |
| 5% | 1,440 | −25.0% |
| 7.5% | 1,197 | −28.2% |
| 10% — only the very best-looking bets | 984 | **−29.5%** |

Being *pickier* made things nearly **three times worse**.

Is this just luck? No. We tested it by resampling the data 2,000 times: the gap
between "bet on everything" and "bet only on strong signals" is statistically
solid at four different filter levels. The pattern also appears in both halves
of the time period separately, so it's not a fluke of one era.

The correlation between filter strength and returns is **−0.98** — almost a
perfect downward line.

**On one season of data this pattern was visible but too weak to be sure. On 22
seasons it's clear.**

---

## Three things that complicate the story

We think being upfront about these matters more than a clean headline.

### 1. There's a second explanation we can't yet rule out

As the filter gets stricter, it stops picking favourites and starts picking
**longshots** — the hit rate falls from 33% to 6%. Long-odds bets are exactly
where bookmaker odds are known to be most distorted (a well-documented effect
called the *favourite-longshot bias*: outsiders win less often than their odds
suggest).

So the declining returns could be caused by:
- the winner's curse (our hypothesis), **or**
- our de-vigging being inaccurate at long odds

Both predict the same downward curve. Separating them is the next job, and it
has to be done before we can claim the winner's curse is the cause.

### 2. Our model barely beats the odds

Measured on how well probabilities predict outcomes, the bookmaker scores
1.0018 and our model scores 1.0014 (lower is better). That difference is
almost nothing.

This cuts both ways. It's *consistent* with the winner's curse story — if
there's no real edge, then everything the filter selects is noise, which is
exactly the scenario. But it also means there's very little genuine signal for
any correction to protect.

Beating a betting market is hard. This is a reminder of how hard.

### 3. The correction we planned may not have much to correct

The plan was to "shrink" the model's disagreements with the market toward zero,
by a factor **w**. w = 0 means *ignore your model entirely*; w = 1 means *trust
it fully*.

We measured **w = 0.86**, with a range of 0.60 to 1.12. Since that range
includes 1, a single fixed shrinkage factor is barely justified.

So if a correction is going to help, it can't be one number applied everywhere.
It has to vary — shrinking more where the model is likely to be noisier (long
odds, obscure leagues, teams with little history) and less where it's reliable.
That's the more interesting version of the idea anyway.

---

## The dataset

**162,053 matches. 22 European divisions. 2005 to 2026.**

We started with one season (7,452 matches) and it wasn't enough — the
statistical noise swamped everything. Detecting a 5% effect needs roughly 2,200
bets, and one season's test set only had ~2,200 *matches* total.

Building the bigger dataset turned out to be most of the work, because the
source files are messier than they look. Things we found and fixed:

| What was wrong | Why it mattered |
| --- | --- |
| 11 team names had invisible trailing spaces in older files | `"Ajax "` and `"Ajax"` are different teams to a computer. Their rating history would silently split in half. |
| `Ath Madrid` became `Atl. Madrid` in 2026 | Same problem, on a major club. |
| One corrupted cell (`#REF!`) in a file | Makes pandas read that file's *entire* odds column as text. Comparing text to numbers gives wrong answers without any error message. |
| 5 files had more columns than their header declared | The file won't load. The obvious fix silently throws away ~2,000 real matches. |
| 10 matches had odds of zero or negative | Zero gives `1/0 = infinity`. A *negative* price is worse — it produces a nonsense probability that looks perfectly normal. |

We also had to be careful about a subtler trap. Our checks suggested matching
`Northwich` to `Norwich`, and `Reggina` to `Reggiana`. Both are wrong —
Northwich Victoria isn't Norwich City, and Reggina and Reggiana are different
clubs from different cities that both played in Serie B at the same time.
Blindly accepting fuzzy matches would have **invented data**. Both rejections
are recorded in the code with reasons.

**The lesson:** none of these announced themselves. They don't crash anything.
They just quietly make your results wrong. Every check in `phase_a.py` exists
because it caught something real.

---

## Running it

```bash
pip install -r requirements.txt

python run.py phase_a.py --full     # build the dataset (~10 min, downloads ~480 files)
python run.py phase_b.py            # run the analysis (~1 min)
```

`run.py` finds a Python that has the dependencies, or sets one up for you.
Each script prints a full report and saves it to a text file.

Optional, for team strength ratings:

```bash
git clone https://github.com/xgabora/Club-Football-Match-Data-2000-2025
```

---

## What's in here

| File | What it does |
| --- | --- |
| `phase_a.py` | Downloads and cleans the data, then runs every quality check. Produces `matches_multiseason.parquet`. |
| `phase_b.py` | The main analysis: builds the model, sweeps the filter, produces the result above. |
| `fetch_data.py` | All the file-downloading and messy-CSV-handling logic. |
| `pipeline.py` | The plan for the rest of the study. Every function has its source paper, equation and pseudocode written out; most are still stubs. |
| `EDA.py` | The original single-season exploration that started all this. |
| `README.md` | You are here. |

Notebook versions of both phases exist for interactive work, but **the `.py`
files are the real ones** — edit those, or the two copies will drift apart.

---

## Honest limitations

- **The test period starts in February 2020**, i.e. right at COVID. Empty
  stadiums measurably changed home advantage, and our model was trained
  entirely on pre-COVID football. This probably makes the early test results
  look worse than they should.
- **We used one train/test split.** Proper practice is to slide the split
  forward repeatedly. That's on the list.
- **The confidence intervals aren't strictly valid after filtering.** The same
  selection effect we're studying also distorts the intervals. We say so rather
  than hiding it.
- **We haven't proven the winner's curse causes this** — see complication #1.

---

## What happens next

1. **Fix the de-vigging question** — compare three methods of removing the
   bookmaker margin and see which one gets long-odds probabilities right. Until
   this is settled, the headline result has two possible explanations.
2. **Build a proper evaluation harness** — multiple metrics, sliding time
   splits, and tests that would catch data leakage.
3. **Simulate it** — the hypothesis is about filtering noisy estimates, which
   isn't really about football at all. On synthetic data we control the noise
   exactly, so we can measure the bias instead of inferring it.

---

## Data sources

- **[football-data.co.uk](https://www.football-data.co.uk/)** — free match
  results and bookmaker odds for 22 European divisions.
- **[xgabora/Club-Football-Match-Data-2000-2025](https://github.com/xgabora/Club-Football-Match-Data-2000-2025)**
  — the same data pre-merged with [ClubElo](https://www.clubelo.com/) team
  ratings.

This is academic research into market efficiency and statistical bias. It is
not betting advice, and every return we measured was negative.
