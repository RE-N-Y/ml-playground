# -*- coding: utf-8 -*-
from sklearn.metrics import r2_score
y_true = [1.0,12.0,1.6]
y_pred = [1.9,12.0,1.3]
print(r2_score(y_true, y_pred))