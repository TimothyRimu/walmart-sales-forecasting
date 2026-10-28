# Walmart Weekly Sales: What Actually Predicts Store Sales?

Weekly sales for 45 Walmart stores over 143 weeks (6,435 rows), with temperature, fuel price, CPI, unemployment and a holiday flag. This began as a class exercise in my Certified Data Scientist training (IABAC / AFRICDSA) that predicted *unemployment rate* from the other columns. I reframed it around the question a retailer would actually ask, **"what drives weekly sales, and how well can we forecast them?"**, and tested it three different ways because the validation method changes the answer a lot. The write-up and code were produced with AI assistance and I reviewed the results.

## Results (error = MAPE, mean absolute % error; lower is better)

| Model | A: random 5-fold | B: time split (train wk 0-99, test wk 100-142) | C: unseen stores |
|---|---|---|---|
| Baseline: each store's average | 8.4% | 8.3% | 67% |
| Ridge: store + weather/economy | 8.9% | 11.0% | 67% |
| Ridge + week-of-year | 8.3% | 8.1% | 66% |
| Boosting, no store id | 47% | 64% | 88% |
| **Boosting + store + week-of-year** | **4.1%** | **5.9%** | 50% |

## What I found
1. **Which store it is explains most of the variation.** Guessing each store's own average already gives R² of 0.92 to 0.96. Because of that, R² is a poor yardstick here; MAPE tells the story better.
2. **Time of year is the real signal.** Adding week-of-year cuts error on the time split from 8.3% to 5.9% with gradient boosting. Weeks 51, 47, 50 and 49 (Christmas and Thanksgiving) sell 61%, 38%, 26% and 16% above a store's average.
3. **Weather, fuel price, CPI and unemployment add essentially nothing.** Ridge with them is no better than the store average (8.9% vs 8.4%) and worse on the time split (11.0%). Their value is not to be trusted from this data.
4. **The holiday flag is a weak summary.** Flagged weeks average only 6.8% above normal, because the flag misses the biggest weeks (the Christmas run-up is not flagged).
5. **A new store cannot be forecast from these columns.** On stores the model has never seen, every model scores near R² 0 (some negative). Without store size or location the economic columns cannot tell a big store from a small one.
6. **The validation method matters.** The random split (A) shows 4.1% for the best model; the realistic time split (B) shows 5.9%. Random splits put neighbouring weeks of the same store in both train and test, which flatters the score.

## Limitations (please read)
- **The CSV has no date column.** I rebuilt dates by assuming rows are in time order per store, starting the week of 2010-02-05 (the convention for this dataset). All 45 stores have their holiday flags in the same row positions, which supports the ordering, but the exact start date is an assumption and week-of-year depends on it.
- One retailer, about 2.7 years, no store attributes (size, region), no promotions data.
- Boosting hyperparameters were not tuned; the model comparison is a baseline study, not an optimised system.

## Run it

A notebook version, [`walmart_sales_forecasting.ipynb`](walmart_sales_forecasting.ipynb), walks through the same analysis step by step.
```
pip install pandas numpy scikit-learn
python analysis.py              # reads data/Walmart.csv
```
Outputs go to `results.json` and `effects.json`. The CSV is the Walmart store sales dataset as supplied in my course materials (also used in my earlier datasets repo); I have not verified its original licence.
