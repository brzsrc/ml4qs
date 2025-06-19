import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report


class Task4:
    def __init__(self, csv: str, test_user: str):
        df = pd.read_csv(csv).drop('Unnamed: 0', axis=1)
        mapping = {'clear': 0, 'drive': 1, 'drop': 2, 'lob': 3, 'smash': 4}
        df['skill_type'] = df['skill_type'].map(mapping)

        # X = df.drop(columns="skill_type user sample_number time".split()).copy()
        # t = df['skill_type']

        # X_train, X_test, y_train, y_test = train_test_split(X, t, test_size = 0.25, random_state = 32)

        self.X_train = df[(df["user"] != test_user)].drop(columns="skill_type user sample_number time".split()).copy()
        print(self.X_train.shape)
        self.X_test = df[(df["user"] == test_user)].drop(columns="skill_type user sample_number time".split()).copy()
        self.y_train = df[(df["user"] != test_user)]['skill_type']
        self.y_test = df[(df["user"] == test_user)]['skill_type']

        self.train =  df[(df["user"] != test_user)]
        self.test = df[(df["user"] == test_user)]

    # param_grid = {
    #     'n_estimators': [200, 300, 500],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'max_depth' : [4,5,6,7,8],
    #     'criterion' :['gini', 'entropy']
    # }
    @classmethod
    def run_cv(cls, title: str, x: pd.DataFrame, y: pd.Series, qid: pd.Series,
               n_estimators: int, max_depth: int, criterion: str) -> float:
        folds = GroupKFold(n_splits=3).split(X=x, y=y, groups=qid)
        models = {}
        print("=" * 20, title, "=" * 20)
        scores_v = []
        for i, (idx_t, idx_v) in enumerate(folds):
            x_t, y_t = [s.iloc[idx_t].reset_index(drop=True) for s in (x, y)]
            x_v, y_v = [s.iloc[idx_v].reset_index(drop=True) for s in (x, y)]
            models[i] = RandomForestClassifier(n_estimators = n_estimators, random_state = 32,
                                               max_depth = max_depth, criterion = criterion)
            models[i].fit(x_t, y_t)

            y_v_pred = models[i].predict(x_v)
            y_t_pred = models[i].predict(x_t)

            print(classification_report(y_v, y_v_pred))
            score_v = accuracy_score(y_v, y_v_pred)
            score_t = accuracy_score(y_t, y_t_pred)
            print(f"fold {i} | train: {score_t:.2f}, val: {score_v:.2f}")
            scores_v.append(score_v)

        val = pd.Series(scores_v)
        print(f"val: mu={val.mean():.6f} min={val.min():.6f} max={val.max():.6f} std={val.std():.5f}")
        return val.max()

    def random_selection(self):
        y_pred = np.random.random_integers(0, 0, size=len(self.y_test))
        print(classification_report(self.y_test, y_pred))

    def random_forest(self):

        # best_paras = grid_search_cv() # criterion='gini', max_depth=8, n_estimators=200

        hyper_params = {

            "param1": dict(
                n_estimators=200,
                max_depth = 6,
                criterion = 'gini', # 'entropy'
            ),
            "param2": dict(
                n_estimators=300,
                max_depth=6,
                criterion='gini',  # 'entropy'
            ),
            "param3": dict(
                n_estimators=300,
                max_depth=8,
                criterion='gini',  # 'entropy'
            ),
            "param4": dict(
                n_estimators=300,
                max_depth=8,
                criterion='gini',  # 'entropy'
            ),
            "param5": dict(
                n_estimators=400,
                max_depth=8,
                criterion='gini',  # 'entropy'
            ),
            "param6": dict(
                n_estimators=500,
                max_depth=8,
                criterion='gini',  # 'entropy'
            ),
            "param7": dict(
                n_estimators=500,
                max_depth=8,
                criterion='entropy',  # 'entropy'
            ),

            "param8": dict(
                n_estimators=600,
                max_depth=8,
                criterion='entropy',  # 'entropy'
            ),
            "param9": dict(
                n_estimators=500,
                max_depth=10,
                criterion='entropy',  # 'entropy'
            ),
        }


        # for title, params in hyper_params.items():
        #     self.run_cv(title, self.X_train, self.y_train, self.train["sample_number"], **params)

        # rfc_base = RandomForestClassifier(random_state=32, n_estimators = 500, max_features = "sqrt", max_depth = 8, criterion = "entropy")
        rfc_base = RandomForestClassifier(random_state=32)
        rfc_base.fit(self.X_train, self.y_train)
        y_pred = rfc_base.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(confusion_matrix(self.y_test, y_pred))

if __name__ == '__main__':
    # CSV_FILE = 'datasets/add_lowpass_filter.csv'
    CSV_FILE = 'datasets/applied_fft_100_50_2_4ppl.csv'
    # CSV_FILE = 'datasets/applied_fft_100_50_2.csv'

    task4 = Task4(CSV_FILE, "keeley")
    task4.random_forest()
    # task4.random_selection()

    # df.to_csv("datasets/applied_fft_100_50_change_sample_num.csv")