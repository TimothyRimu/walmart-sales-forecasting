"""Walmart weekly store sales: what predicts it, and how much do validation choices change the answer?
Data: 45 stores x 143 weeks. The CSV has no Date column; rows are in time order per store (every store has the
same holiday-week positions), so week 0 is taken as the week of 2010-02-05 (standard for this dataset) -- an assumption."""
import pandas as pd, numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
D = sys.argv[1] if len(sys.argv) > 1 else 'data/Walmart.csv'
w = pd.read_csv(D)
w['week'] = w.groupby('Store').cumcount()
w['date'] = pd.Timestamp('2010-02-05') + pd.to_timedelta(w.week*7, unit='D')   # ASSUMED start date
w['woy'] = w.date.dt.isocalendar().week.astype(int)
w['month'] = w.date.dt.month
num = ['Temperature','Fuel_Price','CPI','Unemployment','Holiday_Flag']
y = w.Weekly_Sales.values
def mape(t,p): return float(np.mean(abs(t-p)/t)*100)
def sc(t,p): return dict(MAE=float(mean_absolute_error(t,p)), MAPE=mape(t,p), R2=float(r2_score(t,p)))
def models(train_idx):
    tr = w.iloc[train_idx]
    return {
     'baseline: store mean': None,
     'ridge: store + weather/economy': make_pipeline(ColumnTransformer([('s',OneHotEncoder(handle_unknown='ignore'),['Store']),('n',StandardScaler(),num)]), Ridge(1.0)),
     'ridge + week-of-year': make_pipeline(ColumnTransformer([('s',OneHotEncoder(handle_unknown='ignore'),['Store','woy']),('n',StandardScaler(),num)]), Ridge(1.0)),
     'boosting (no store id)': HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=0),
     'boosting + store + week-of-year': HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=0,categorical_features=[0]),
    }
cols = {'baseline: store mean':None,
        'ridge: store + weather/economy':['Store']+num,
        'ridge + week-of-year':['Store','woy']+num,
        'boosting (no store id)':num+['woy'],
        'boosting + store + week-of-year':['Store']+num+['woy']}
def run(splits):
    out = {}
    preds = {k: np.full(len(w), np.nan) for k in cols}
    for tr, te in splits:
        M = models(tr)
        for k, m in M.items():
            if m is None:
                mean = w.iloc[tr].groupby('Store').Weekly_Sales.mean()
                preds[k][te] = w.iloc[te].Store.map(mean).fillna(y[tr].mean()).values   # unseen store -> overall mean
            else:
                m.fit(w.iloc[tr][cols[k]], y[tr]); preds[k][te] = m.predict(w.iloc[te][cols[k]])
    for k in cols:
        ok = ~np.isnan(preds[k]); out[k] = sc(y[ok], preds[k][ok])
    return out
res = {}
res['A random 5-fold (rows shuffled; neighbouring weeks leak)'] = run(KFold(5, shuffle=True, random_state=0).split(w))
tr = np.where(w.week < 100)[0]; te = np.where(w.week >= 100)[0]
res['B time split (train weeks 0-99, test weeks 100-142, all stores)'] = run([(tr, te)])
res['C unseen stores (GroupKFold by store, 5 folds)'] = run(GroupKFold(5).split(w, groups=w.Store))
json.dump(res, open('results.json','w'), indent=1)
for k, v in res.items():
    print(k)
    for m, s in v.items(): print(f'   {m:36s} MAPE {s["MAPE"]:5.1f}%  MAE {s["MAE"]:>9,.0f}  R2 {s["R2"]:.3f}')
# holiday effect
h = w.assign(r=w.Weekly_Sales/w.groupby('Store').Weekly_Sales.transform('mean')).groupby('Holiday_Flag').r.mean()
t = w.assign(r=w.Weekly_Sales/w.groupby('Store').Weekly_Sales.transform('mean')).groupby('woy').r.mean().sort_values(ascending=False).head(4)
print('holiday-week sales vs store average:', h.round(3).to_dict()); print('top weeks of year:', t.round(2).to_dict())
json.dump({'holiday_ratio':h.round(4).to_dict(),'top_weeks':t.round(3).to_dict()}, open('effects.json','w'))
