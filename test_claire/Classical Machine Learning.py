
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier  # Added XGBoost import
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Task4:
    """Extended version of the original Task4 class.

    Adds multiple classical and ensemble ML algorithms (SVM, KNN, Decision
    Tree, Naive Bayes, Neural Networks, Gradient Boosting, AdaBoost, Voting),
    and offers a one‑stop `evaluate_models` method that trains, evaluates and
    compares their performance on a leave‑one‑user‑out split. The same method
    now **also draws a micro‑average ROC curve for every model** so you can
    visually compare AUCs in multi‑class settings.
    """

    def __init__(self, csv: str, test_user: str):
        # --------------------------- Data Loading --------------------------- #
        df = pd.read_csv(csv).drop(columns=["Unnamed: 0"], errors="ignore")

        # Encode skill labels to integers (makes life easier for all models)
        self.mapping = {"clear": 0, "drive": 1, "drop": 2, "lob": 3, "smash": 4}
        df["skill_type"] = df["skill_type"].map(self.mapping)

        # Leave‑one‑user‑out split, identical to original approach
        self.X_train = df[df["user"] != test_user].drop(
            columns="skill_type user sample_number time".split()
        )
        self.X_test = df[df["user"] == test_user].drop(
            columns="skill_type user sample_number time".split()
        )
        self.y_train = (
            df[df["user"] != test_user]["skill_type"].reset_index(drop=True)
        )
        self.y_test = (
            df[df["user"] == test_user]["skill_type"].reset_index(drop=True)
        )

        # Keep these around for optional grouped CV by sample_number
        self.train = df[df["user"] != test_user].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def _group_cv_score(self, model, folds: int = 10) -> float:
        """Return mean accuracy on grouped CV (groups = sample_number)."""
        gkf = GroupKFold(n_splits=folds)
        scores = []
        for tr_idx, val_idx in gkf.split(
            self.X_train, self.y_train, groups=self.train["sample_number"]
        ):
            model.fit(self.X_train.iloc[tr_idx], self.y_train.iloc[tr_idx])
            y_pred = model.predict(self.X_train.iloc[val_idx])
            scores.append(accuracy_score(self.y_train.iloc[val_idx], y_pred))
        return float(np.mean(scores))

    # ------------------------------------------------------------------ #
    # def evaluate_models(self, *, use_group_cv: bool = False, plot_confusion_matrix: bool = True):
    #
    #     # --------------------------- Model Zoo --------------------------- #
    #     base_models = {
    #
    #         "DecisionTree": DecisionTreeClassifier(
    #             criterion='entropy',
    #             max_depth=25,
    #             min_samples_split=8,
    #             random_state=32
    #         ),
    #         "KNN": KNeighborsClassifier(
    #             n_neighbors=6,
    #             metric='manhattan',
    #             weights='distance'
    #         ),
    #         "NaiveBayes": GaussianNB(),
    #         "MLP": MLPClassifier(
    #             hidden_layer_sizes=(128,),
    #             alpha=0.00462,
    #             learning_rate='constant',
    #             max_iter=1000,
    #             random_state=32
    #         ),
    #         "SVM": SVC(kernel="rbf", probability=True, random_state=32),
    #         "RandomForest": RandomForestClassifier(
    #             bootstrap=False,
    #             max_depth=30,
    #             min_samples_leaf=1,
    #             min_samples_split=5,
    #             n_estimators=406,
    #             random_state=32
    #         ),
    #         # "GradientBoost": GradientBoostingClassifier(
    #         #     learning_rate=0.216,
    #         #     max_depth= 5,
    #         #     n_estimators= 176,
    #         #     random_state=32),
    #         "XGB": XGBClassifier(
    #             objective="multi:softprob",
    #             eval_metric="mlogloss",
    #             use_label_encoder=False,
    #             random_state=32,
    #             n_jobs=-1,
    #             colsample_bytree=0.5611662398028049,
    #             learning_rate=0.02854693771182679,
    #             max_depth=8,
    #             n_estimators=531,
    #             subsample=0.5899138254852011,
    #         ),
    #     }
    #     #
    #     # base_models["VotingSoft"] = VotingClassifier(
    #     #     estimators=[
    #     #         ("rf", base_models["RandomForest"]),
    #     #         ("gb", base_models["GradientBoost"]),
    #     #         ("xgb", base_models["XGB"]),
    #     #     ],
    #     #     voting="soft",
    #     # )
    #
    #
    #     # --------------------------- Training --------------------------- #
    #     results = {}
    #     model_scores = {}  # 用于保存 predict_proba 用于 ROC
    #
    #     for name, model in base_models.items():
    #         # Optional GroupKFold CV (on training data only)
    #         if use_group_cv:
    #             cv_acc = self._group_cv_score(model)
    #             results[f"{name}_CV"] = cv_acc
    #             print(f"{name}  |  3‑fold GroupCV accuracy: {cv_acc:.3f}")
    #
    #         # Fit on all training data & evaluate on held‑out user
    #         model.fit(self.X_train, self.y_train)
    #         y_pred = model.predict(self.X_test)
    #         acc = accuracy_score(self.y_test, y_pred)
    #         results[name] = acc
    #
    #         # 预测概率 / decision_function，用于之后画 ROC
    #         try:
    #             y_score = model.predict_proba(self.X_test)
    #         except AttributeError:  # e.g. if model lacks predict_proba
    #             y_score = model.decision_function(self.X_test)
    #             # 若返回向量，扩展成两列以与 binarize 形状相符
    #             if y_score.ndim == 1:
    #                 y_score = np.vstack([1 - y_score, y_score]).T
    #         model_scores[name] = y_score
    #
    #         print("\n" + "=" * 72)
    #         uniq_user = ", ".join(self.train["user"].unique())
    #         print(
    #             f"{name}  |  Test accuracy on user(s) '{uniq_user}': {acc:.3f}\n"
    #         )
    #         # zero_division=0 avoids warnings when a class is never predicted
    #         print(classification_report(self.y_test, y_pred, zero_division=0))
    #
    #
    #     # --------------------------- Bar Plot --------------------------- #
    #     colors = ["#5096DE" if name.endswith("_CV") else "darkorange" for name in results]
    #     plt.figure(figsize=(10, 5))
    #     plt.bar(results.keys(), results.values(), color=colors)
    #
    #     # 添加图例
    #     cv_patch = plt.Rectangle((0, 0), 1, 1, fc="#5096DE", edgecolor="none")
    #     non_cv_patch = plt.Rectangle((0, 0), 1, 1, fc="darkorange", edgecolor="none")
    #     plt.legend([cv_patch, non_cv_patch], ["Validation ACC", "Test ACC"], loc="best")
    #
    #     plt.xticks(rotation=45, ha="right")
    #     plt.ylabel("Accuracy")
    #     plt.title("Model Comparison")
    #     plt.tight_layout()
    #     plt.savefig("Model_Comparison.png")
    #
    #     # --------------------------- 混淆矩阵 ---------------------------
    #     if plot_confusion_matrix:
    #         from sklearn.metrics import confusion_matrix
    #         import seaborn as sns
    #         plt.figure(figsize=(15, 10))
    #         plt.suptitle("Confusion Matrix Comparison", y=1.02)
    #
    #         for i, (name, model) in enumerate(base_models.items(), 1):
    #             y_pred = model.predict(self.X_test)
    #             cm = confusion_matrix(self.y_test, y_pred)
    #
    #             plt.subplot(3, 4, i)
    #             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                         xticklabels=list(self.mapping.keys()),
    #                         yticklabels=list(self.mapping.keys()))
    #             plt.title(f"{name}\nAccuracy: {results[name]:.3f}")
    #             plt.xlabel('Predicted')
    #             plt.ylabel('Actual')
    #
    #         plt.tight_layout()
    #         plt.savefig('Confusion_Matrices.png', bbox_inches='tight')
    #         plt.close()
    #
    #     # --------------------------- ROC Curves --------------------------- #
    #     n_classes = len(np.unique(self.y_train))
    #     y_test_bin = label_binarize(self.y_test, classes=np.arange(n_classes))
    #
    #     plt.figure(figsize=(8, 6))
    #     for name, y_score in model_scores.items():
    #
    #         if y_score.ndim == 1:
    #             continue
    #         fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    #         roc_auc = auc(fpr, tpr)
    #         plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
    #
    #     plt.plot([0, 1], [0, 1], "k--", lw=2)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("ROC Curve Comparison")
    #     plt.legend(loc="lower right")
    #     plt.tight_layout()
    #     plt.savefig("ROC_Comparison.png")
    #
    #     return results

    def get_model_zoo(self):
        """定义所有基础模型"""
        return {
            "DecisionTree": DecisionTreeClassifier(
                criterion='entropy', max_depth=25, min_samples_split=8, random_state=32
            ),
            "KNN": KNeighborsClassifier(n_neighbors=6, metric='manhattan', weights='distance'),
            "NaiveBayes": GaussianNB(),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128,), alpha=0.00462,
                learning_rate='constant', max_iter=1000, random_state=32
            ),
            "SVM": SVC(kernel="rbf", probability=True, random_state=32),
            "RandomForest": RandomForestClassifier(
                bootstrap=False, max_depth=30, min_samples_leaf=1,
                min_samples_split=5, n_estimators=406, random_state=32
            ),
            "XGB": XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                use_label_encoder=False, random_state=32, n_jobs=-1,
                colsample_bytree=0.5611662398028049,
                learning_rate=0.02854693771182679,
                max_depth=8, n_estimators=531,
                subsample=0.5899138254852011,
            )
        }

    def fit_and_evaluate_model(self,name, model, X_train, y_train, X_test, y_test):
        """训练并评估单个模型"""
        from sklearn.metrics import accuracy_score, classification_report
        import numpy as np

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        try:
            y_score = model.predict_proba(X_test)
        except AttributeError:
            y_score = model.decision_function(X_test)
            if y_score.ndim == 1:
                y_score = np.vstack([1 - y_score, y_score]).T

        print("\n" + "=" * 72)
        print(f"{name} | Test accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        return acc, y_score

    def plot_model_comparison(self,results):
        """绘制模型准确率对比柱状图"""
        import matplotlib.pyplot as plt

        colors = ["#5096DE" if name.endswith("_CV") else "darkorange" for name in results]
        plt.figure(figsize=(10, 5))
        plt.bar(results.keys(), results.values(), color=colors)

        cv_patch = plt.Rectangle((0, 0), 1, 1, fc="#5096DE", edgecolor="none")
        non_cv_patch = plt.Rectangle((0, 0), 1, 1, fc="darkorange", edgecolor="none")
        plt.legend([cv_patch, non_cv_patch], ["Validation ACC", "Test ACC"], loc="best")

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Model Comparison")
        plt.tight_layout()
        plt.savefig("Model_Comparison.png")
        plt.close()

    def plot_confusion_matrices(self,models, X_test, y_test, mapping, results):
        """绘制所有模型的混淆矩阵"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        plt.figure(figsize=(15, 10))
        plt.suptitle("Confusion Matrix Comparison", y=1.02)
        for i, (name, model) in enumerate(models.items(), 1):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            plt.subplot(3, 4, i)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=list(mapping.keys()),
                        yticklabels=list(mapping.keys()))
            plt.title(f"{name}\nAccuracy: {results[name]:.3f}")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig("Confusion_Matrices.png", bbox_inches="tight")
        plt.close()

    def plot_roc_curves(self,model_scores, y_test, y_train):
        """绘制所有模型的 ROC 曲线"""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        n_classes = len(np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

        plt.figure(figsize=(8, 6))
        for name, y_score in model_scores.items():
            if y_score.ndim == 1:
                continue
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("ROC_Comparison.png")
        plt.close()

    def evaluate_models(self, use_group_cv=False, plot_confusion_matrix=True):
        models = self.get_model_zoo()
        results = {}
        model_scores = {}

        for name, model in models.items():
            if use_group_cv:
                acc = self._group_cv_score(model)
                results[f"{name}_CV"] = acc
            acc, y_score = self.fit_and_evaluate_model(name, model, self.X_train, self.y_train, self.X_test, self.y_test)
            results[name] = acc
            model_scores[name] = y_score

        self.plot_model_comparison(results)

        if plot_confusion_matrix:
            self.plot_confusion_matrices(models, self.X_test, self.y_test, self.mapping, results)

        self.plot_roc_curves(model_scores, self.y_test, self.y_train)
        return results


if __name__ == "__main__":
    # Example invocation — just change CSV and user as needed
    CSV_FILE = "datasets/applied_fft_100_50_2_4ppl.csv"

    task4 = Task4(CSV_FILE, test_user="keeley")
    # Set use_group_cv=True to also see grouped cross‑validation scores
    task4.evaluate_models(use_group_cv=True, plot_confusion_matrix=True)

