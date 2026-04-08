# ============================================================
# 🔥 0. IMPORTS
# ============================================================
import numpy as np
import torch
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from scipy.optimize import linear_sum_assignment

from topmost import (
    Preprocess,
    RawDataset,
    BasicTrainer,
    FASTopicTrainer,
    eva,
    TSCTM,
    ECRTM,
    NSTM,
    ETM
)

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from TopicClusterDocument.GloCOM import GloCOM


# ============================================================
# 📦 DATASET WRAPPER
# ============================================================
class DatasetWrapper:
    def __init__(self, X, y=None, test_size=None, random_state=42):
        self.X = X
        self.y = y

        if y is not None and test_size:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
        else:
            self.X_train = X
            self.X_test = None
            self.y_train = y
            self.y_test = None

    def has_labels(self):
        return self.y is not None


# ============================================================
# ⚙️ PREPROCESSOR (SHARED VOCAB)
# ============================================================
class Preprocessor:
    def __init__(self, vocab_size=10000, stopwords="English", device="cpu"):
        self.vocab_size = vocab_size
        self.preprocess = Preprocess(vocab_size=vocab_size, stopwords=stopwords)
        self.device = device

    def fit_transform(self, texts):
        dataset = RawDataset(
            docs=texts,
            preprocess=self.preprocess,
            batch_size=200,
            device=self.device,
            as_tensor=True,
            contextual_embed=False,
            verbose=True
        )
        return dataset


# ============================================================
# 📊 METRICS
# ============================================================
def unsupervised_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return sum(w[i, j] for i, j in zip(row_ind, col_ind)) / len(y_true)


def purity_score(y_true, y_pred):
    y_true = np.array(y_true)   # 🔥 FIX
    y_pred = np.array(y_pred)

    clusters = np.unique(y_pred)
    total = 0

    for c in clusters:
        idx = np.where(y_pred == c)[0]   # 🔥 FIX (extract array)
        true_labels = y_true[idx]

        if len(true_labels) == 0:
            continue

        total += np.bincount(true_labels).max()

    return total / len(y_true)

def evaluate_clustering(y_true, y_pred):
    return {
        "ACC": unsupervised_accuracy(y_true, y_pred),
        "Purity": purity_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "V_measure": v_measure_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
    }


def evaluate_topics(top_words, texts, vocab):
    return {
        "TC": eva._coherence(texts, vocab, top_words),
        "TD": eva._diversity(top_words),
    }


# ============================================================
# 🧠 BASE MODEL
# ============================================================
class TopicModelWrapper:
    def fit(self, dataset):
        raise NotImplementedError

    def predict(self, texts=None):
        raise NotImplementedError

    def get_topics(self):
        raise NotImplementedError


# ============================================================
# 🟣 TOPMOST MODELS
# ============================================================
class TopMostWrapper(TopicModelWrapper):
    def __init__(self, model_class, vocab_size, num_topics=20, device="cpu"):
        self.model = model_class(
            vocab_size=vocab_size,
            num_topics=num_topics
        ).to(device)

    def fit(self, dataset):
        self.trainer = BasicTrainer(self.model, dataset)
        self.top_words, self.theta = self.trainer.train()

    def predict(self, texts=None):
        return self.theta.argmax(axis=1)

    def get_topics(self):
        return self.top_words


# ============================================================
# 🔵 FASTOPIC (FIXED)
# ============================================================
class FASTopicWrapper(TopicModelWrapper):
    def __init__(self, num_topics=20, vocab_size=10000):
        self.num_topics = num_topics
        self.vocab_size = vocab_size

    def fit(self, dataset):
        self.trainer = FASTopicTrainer(
            dataset=dataset,
            num_topics=self.num_topics,
            verbose=True
        )
        self.top_words, self.theta = self.trainer.train()

    def predict(self, texts):
        theta = self.trainer.test(texts)
        return theta.argmax(axis=1)

    def get_topics(self):
        return self.top_words


# ============================================================
# 🔴 GLOCOM (FINAL FIXED)
# ============================================================
class GloCOMWrapper(TopicModelWrapper):
    def __init__(self, vocab_size, num_topics=20, device="cpu", epochs=20):
        self.device = device
        self.num_topics = num_topics
        self.epochs = epochs
        self.vocab_size = vocab_size

        self.model = GloCOM(
            vocab_size=vocab_size,
            num_topics=num_topics,
            en_units=200,
            dropout=0.2
        ).to(device)

    def fit(self, dataset):

        vocab = dataset.vocab
        texts = dataset.train_texts

        vectorizer = CountVectorizer(
            vocabulary=vocab,
            stop_words=None
        )

        X = vectorizer.transform(texts).toarray()
        X_glocom = np.concatenate([X, X], axis=1)

        class BoWDataset(Dataset):
            def __init__(self, X):
                self.X = torch.tensor(X, dtype=torch.float32)

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                return self.X[idx]

        loader = DataLoader(BoWDataset(X_glocom), batch_size=64, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0

            for batch in loader:
                batch = batch.to(self.device)

                optimizer.zero_grad()
                loss = self.model(batch)["loss"]

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"GloCOM Epoch {epoch+1}: Loss={total_loss:.2f}")

        self.X_glocom = X_glocom
        self.vocab = vocab

    def predict(self, texts=None):
        self.model.eval()

        with torch.no_grad():
            tensor = torch.tensor(self.X_glocom, dtype=torch.float32).to(self.device)

            if hasattr(self.model, "get_theta"):
                theta = self.model.get_theta(tensor)
            elif hasattr(self.model, "inference"):
                theta = self.model.inference(tensor)
            else:
                theta = self.model.encoder(tensor)

            return torch.argmax(theta, dim=1).cpu().numpy()

    def get_topics(self):
        beta = self.model.get_beta().detach().cpu().numpy()

        topics = []
        for k in range(beta.shape[0]):
            top_idx = np.argsort(beta[k])[-10:][::-1]
            words = [self.vocab[i] for i in top_idx]
            topics.append(" ".join(words))

        return topics


# ============================================================
# 🟡 LDA
# ============================================================
class LDAWrapper(TopicModelWrapper):
    def __init__(self, num_topics=20):
        self.model = LatentDirichletAllocation(n_components=num_topics)

    def fit(self, X):
        self.model.fit(X)
        self.theta = self.model.transform(X)

    def predict(self, X):
        return self.theta.argmax(axis=1)

    def get_topics(self, vocab):
        topics = []
        for topic in self.model.components_:
            top_idx = topic.argsort()[-10:][::-1]
            words = [vocab[i] for i in top_idx]
            topics.append(" ".join(words))
        return topics


# ============================================================
# 🚀 BENCHMARK RUNNER
# ============================================================
class BenchmarkRunner:
    def __init__(self, models, device="cpu", vocab_size=10000):
        self.models = models
        self.device = device
        self.vocab_size = vocab_size

    def run(self, dataset: DatasetWrapper):

        results = []

        preprocessor = Preprocessor(
            vocab_size=self.vocab_size,
            device=self.device
        )

        train_data = preprocessor.fit_transform(dataset.X_train)
        vocab = train_data.vocab

        # LDA input
        _, bow = preprocessor.preprocess.parse(dataset.X_train, vocab=vocab)
        X_bow = bow.toarray()

        for name, model in self.models.items():

            print(f"\n===== {name} =====")

            if isinstance(model, LDAWrapper):
                model.fit(X_bow)
                preds = model.predict(X_bow)
                topics = model.get_topics(vocab)
            else:
                model.fit(train_data)
                preds = model.predict(dataset.X_train)
                topics = model.get_topics()

            result = {"model": name}

            if dataset.has_labels():
                result.update(evaluate_clustering(dataset.y_train, preds))

            result.update(evaluate_topics(topics, train_data.train_texts, vocab))
            results.append(result)

        return pd.DataFrame(results).sort_values(by="NMI", ascending=False)