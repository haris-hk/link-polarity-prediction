import argparse
import gzip
import itertools
import json
import math
import random
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

try:
	import matplotlib.pyplot as plt
	HAS_MPL = True
except ImportError:
	plt = None
	HAS_MPL = False

try:
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
	from sklearn.model_selection import train_test_split
except ImportError as exc:
	raise ImportError(
		"This script requires scikit-learn. Install it with: pip install scikit-learn"
	) from exc

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
except ImportError:
	torch = None

try:
	from torch_geometric.nn import GCNConv

	HAS_PYG = True
except Exception:
	HAS_PYG = False


RANDOM_SEED = 42


@dataclass
class SplitData:
	x_train: np.ndarray
	x_test: np.ndarray
	y_train: np.ndarray
	y_test: np.ndarray


@dataclass
class IndexSplitData:
	train_idx: np.ndarray
	val_idx: np.ndarray
	test_idx: np.ndarray


def set_seed(seed: int = RANDOM_SEED) -> None:
	random.seed(seed)
	np.random.seed(seed)
	if torch is not None:
		torch.manual_seed(seed)


def load_election_data(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Election file not found: {path}")

	df = pd.read_csv(path)

	# Normalize header variants (case/space/typos) into canonical names.
	normalized_map = {}
	for col in df.columns:
		key = re.sub(r"\s+", "", str(col).strip().lower())
		normalized_map[col] = key

	inverse = {v: k for k, v in normalized_map.items()}
	canonical = {
		"region": inverse.get("region"),
		"constituency": inverse.get("constituency") or inverse.get("contituancy"),
		"member": inverse.get("member"),
		"party": inverse.get("party"),
	}
	required = {"region", "constituency", "member", "party"}
	missing = {k for k in required if canonical.get(k) is None}
	if missing:
		raise ValueError(f"Missing election columns: {sorted(missing)}")

	cleaned = df.rename(columns={v: k for k, v in canonical.items() if v is not None}).copy()
	for col in ["region", "constituency", "member", "party"]:
		cleaned[col] = cleaned[col].astype(str).str.strip()
	cleaned = cleaned.dropna(subset=["member", "party"])
	return cleaned


def load_tweets_data(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Tweets file not found: {path}")

	df = pd.read_csv(path)
	required = {"User", "Tweet"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"Missing tweet columns: {sorted(missing)}")

	cleaned = df.copy()
	cleaned["User"] = cleaned["User"].astype(str).str.strip()
	cleaned["Tweet"] = cleaned["Tweet"].astype(str)
	cleaned = cleaned.dropna(subset=["User", "Tweet"])
	return cleaned


def load_party_rivalry_map(path: Path) -> Dict[str, List[str]]:
	if not path.exists():
		raise FileNotFoundError(
			f"Rivalry config not found: {path}. Create this file or pass --rivalry-config."
		)

	with path.open("r", encoding="utf-8") as fp:
		data = json.load(fp)

	if not isinstance(data, dict):
		raise ValueError("Rivalry config must be a JSON object mapping party -> list of rival parties.")

	normalized: Dict[str, List[str]] = {}
	for party, rivals in data.items():
		party_n = normalize_party_name(party)
		if not isinstance(rivals, list):
			raise ValueError(f"Rivalry entry for party '{party_n}' must be a list.")
		normalized[party_n] = [normalize_party_name(x) for x in rivals]

	return normalized


def normalize_party_name(name: str) -> str:
	return re.sub(r"\s+", " ", str(name).strip())


def add_signed_edge(
	graph: nx.Graph,
	u: str,
	v: str,
	sign: int,
	source: str,
) -> None:
	if u == v:
		return
	if sign not in (-1, 1):
		return

	if graph.has_edge(u, v):
		existing = graph[u][v].get("sign", sign)
		if existing != sign:
			# In case of conflicting evidence, keep the edge with larger weight.
			graph[u][v]["weight"] = graph[u][v].get("weight", 1.0) + 1.0
		else:
			graph[u][v]["weight"] = graph[u][v].get("weight", 1.0) + 1.0
		return

	graph.add_edge(u, v, sign=sign, source=source, weight=1.0)


def build_election_signed_graph(
	election_df: pd.DataFrame,
	rivalry_map: Dict[str, List[str]],
	max_negative_per_party_pair: Optional[int] = None,
) -> nx.Graph:
	graph = nx.Graph()

	members_by_party: Dict[str, List[str]] = {}
	for row in election_df.itertuples(index=False):
		member = f"member::{row.member}"
		party = normalize_party_name(row.party)
		region = str(row.region).strip()
		members_by_party.setdefault(party, []).append(member)

		graph.add_node(member, node_type="member", party=party, region=region)

	# Positive intra-party relations.
	for party, members in members_by_party.items():
		for u, v in itertools.combinations(sorted(set(members)), 2):
			add_signed_edge(graph, u, v, sign=1, source=f"same_party::{party}")

	# Negative inter-party relations according to rivalry assumptions.
	for party_a, rivals in rivalry_map.items():
		party_a = normalize_party_name(party_a)
		members_a = sorted(set(members_by_party.get(party_a, [])))
		if not members_a:
			continue

		for party_b in rivals:
			party_b = normalize_party_name(party_b)
			members_b = sorted(set(members_by_party.get(party_b, [])))
			if not members_b:
				continue

			candidate_pairs = [(a, b) for a in members_a for b in members_b]
			if not candidate_pairs:
				continue

			random.shuffle(candidate_pairs)
			if max_negative_per_party_pair is None or max_negative_per_party_pair <= 0:
				sampled = candidate_pairs
			else:
				sampled = candidate_pairs[:max_negative_per_party_pair]
			for u, v in sampled:
				add_signed_edge(
					graph,
					u,
					v,
					sign=-1,
					source=f"rival_party::{party_a}_vs_{party_b}",
				)

	return graph


def tweet_sentiment_sign(text: str) -> int:
	positive_words = {
		"support",
		"respect",
		"agree",
		"good",
		"great",
		"best",
		"important",
		"like",
		"love",
	}
	negative_words = {
		"hate",
		"worst",
		"corrupt",
		"fraud",
		"shame",
		"dirty",
		"disagree",
		"bad",
		"joker",
		"useless",
	}

	lowered = text.lower()
	pos_score = sum(1 for w in positive_words if w in lowered)
	neg_score = sum(1 for w in negative_words if w in lowered)
	if pos_score > neg_score:
		return 1
	if neg_score > pos_score:
		return -1
	return 0


def build_tweet_signed_graph(tweets_df: pd.DataFrame) -> nx.Graph:
	graph = nx.Graph()
	mention_pattern = re.compile(r"@([A-Za-z0-9_]+)")

	for row in tweets_df.itertuples(index=False):
		user = f"user::{str(row.User).strip()}"
		text = str(row.Tweet)

		graph.add_node(user, node_type="user")
		mentions = mention_pattern.findall(text)
		if not mentions:
			continue

		sign = tweet_sentiment_sign(text)
		if sign == 0:
			continue

		# Create denser supervision by linking co-mentioned accounts too.
		for m in mentions:
			target = f"user::{m.strip()}"
			graph.add_node(target, node_type="user")
			add_signed_edge(graph, user, target, sign=sign, source="tweet_mention")

		if len(mentions) > 1:
			mention_nodes = [f"user::{m.strip()}" for m in mentions]
			for left, right in itertools.combinations(sorted(set(mention_nodes)), 2):
				graph.add_node(left, node_type="user")
				graph.add_node(right, node_type="user")
				add_signed_edge(graph, left, right, sign=sign, source="tweet_mention_clique")

	return graph


def merge_graphs(*graphs: nx.Graph) -> nx.Graph:
	merged = nx.Graph()
	for g in graphs:
		for node, attrs in g.nodes(data=True):
			if node not in merged:
				merged.add_node(node, **attrs)
			else:
				merged.nodes[node].update(attrs)

		for u, v, attrs in g.edges(data=True):
			add_signed_edge(merged, u, v, sign=attrs.get("sign", 1), source=attrs.get("source", "merged"))
	return merged


def signed_benchmark_urls() -> Dict[str, str]:
	return {
		"slashdot": "https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz",
		"wikirfa": "https://snap.stanford.edu/data/wiki-RfA.txt.gz",
	}


def download_if_missing(url: str, destination: Path) -> Path:
	destination.parent.mkdir(parents=True, exist_ok=True)
	if destination.exists() and destination.stat().st_size > 0:
		return destination
	print(f"[INFO] Downloading benchmark dataset from {url}")
	urllib.request.urlretrieve(url, destination)
	return destination


def parse_signed_edgelist_gz(path: Path, prefix: str, max_edges: int = 0) -> nx.Graph:
	graph = nx.Graph()
	added = 0

	with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fp:
		for raw in fp:
			line = raw.strip()
			if not line or line.startswith("#"):
				continue

			parts = re.split(r"[\t\s]+", line)
			if len(parts) < 3:
				continue

			u = f"{prefix}::{parts[0]}"
			v = f"{prefix}::{parts[1]}"
			graph.add_node(u, node_type="benchmark")
			graph.add_node(v, node_type="benchmark")

			sign_token = None
			for tok in parts[2:7]:
				if tok in {"1", "+1", "-1"}:
					sign_token = tok
					break
			if sign_token is None:
				continue

			sign = 1 if sign_token in {"1", "+1"} else -1
			add_signed_edge(graph, u, v, sign=sign, source=f"benchmark::{prefix}")
			added += 1
			if max_edges > 0 and added >= max_edges:
				break

	return graph


def load_external_signed_benchmark(
	dataset_name: str,
	cache_dir: Path,
	max_edges: int,
) -> nx.Graph:
	name = dataset_name.lower().strip()
	url_map = signed_benchmark_urls()
	if name not in url_map:
		raise ValueError(
			f"Unsupported benchmark dataset '{dataset_name}'. Choose one of: {sorted(url_map)}"
		)

	file_name = f"{name}.txt.gz"
	gz_path = download_if_missing(url_map[name], cache_dir / file_name)
	graph = parse_signed_edgelist_gz(gz_path, prefix=f"benchmark_{name}", max_edges=max_edges)
	if graph.number_of_edges() == 0:
		raise ValueError(
			f"Parsed benchmark dataset '{dataset_name}' but found zero signed edges."
		)
	return graph


def summarize_signed_graph(graph: nx.Graph) -> Dict[str, int]:
	pos = 0
	neg = 0
	for _, _, attrs in graph.edges(data=True):
		if attrs.get("sign", 0) == 1:
			pos += 1
		elif attrs.get("sign", 0) == -1:
			neg += 1
	return {
		"nodes": graph.number_of_nodes(),
		"edges": graph.number_of_edges(),
		"positive_edges": pos,
		"negative_edges": neg,
	}


def sample_triangles(graph: nx.Graph, max_samples: int = 20000) -> List[Tuple[str, str, str]]:
	triangles: List[Tuple[str, str, str]] = []
	nodes = list(graph.nodes())
	random.shuffle(nodes)

	for u in nodes:
		neighbors = list(graph.neighbors(u))
		if len(neighbors) < 2:
			continue
		random.shuffle(neighbors)
		for v, w in itertools.combinations(neighbors, 2):
			if graph.has_edge(v, w):
				tri = tuple(sorted((u, v, w)))
				triangles.append(tri)
				if len(triangles) >= max_samples:
					return list(dict.fromkeys(triangles))

	return list(dict.fromkeys(triangles))


def compute_balance_metrics(graph: nx.Graph, max_samples: int = 20000) -> Dict[str, float]:
	triangles = sample_triangles(graph, max_samples=max_samples)
	if not triangles:
		return {
			"triangles": 0,
			"balanced_triangles": 0,
			"balance_ratio": 0.0,
		}

	balanced = 0
	for a, b, c in triangles:
		sab = graph[a][b].get("sign", 1)
		sbc = graph[b][c].get("sign", 1)
		sca = graph[c][a].get("sign", 1)
		if sab * sbc * sca > 0:
			balanced += 1

	ratio = balanced / len(triangles)
	return {
		"triangles": float(len(triangles)),
		"balanced_triangles": float(balanced),
		"balance_ratio": ratio,
	}


def edge_feature_vector(graph: nx.Graph, u: str, v: str) -> List[float]:
	nu = set(graph.neighbors(u))
	nv = set(graph.neighbors(v))
	intersection = len(nu & nv)
	union = len(nu | nv)
	jaccard = intersection / union if union else 0.0
	pref_attach = float(len(nu) * len(nv))

	party_u = graph.nodes[u].get("party")
	party_v = graph.nodes[v].get("party")
	same_party = 1.0 if party_u and party_v and party_u == party_v else 0.0

	return [
		float(graph.degree(u)),
		float(graph.degree(v)),
		float(intersection),
		jaccard,
		pref_attach,
		same_party,
	]


def build_edge_classification_dataset(
	graph: nx.Graph,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
	x = []
	y = []
	pairs = []

	for u, v, attrs in graph.edges(data=True):
		sign = attrs.get("sign")
		if sign not in (-1, 1):
			continue
		x.append(edge_feature_vector(graph, u, v))
		y.append(1 if sign == 1 else 0)
		pairs.append((u, v))

	if not x:
		raise ValueError("No signed edges available to build dataset.")

	return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), pairs


def train_test_split_edges(x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> SplitData:
	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=test_size,
		random_state=RANDOM_SEED,
		stratify=y,
	)
	return SplitData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def train_val_test_split_indices(y: np.ndarray) -> IndexSplitData:
	indices = np.arange(len(y))
	train_idx, temp_idx = train_test_split(
		indices,
		test_size=0.3,
		random_state=RANDOM_SEED,
		stratify=y,
	)
	temp_y = y[temp_idx]
	val_idx, test_idx = train_test_split(
		temp_idx,
		test_size=0.5,
		random_state=RANDOM_SEED,
		stratify=temp_y,
	)
	return IndexSplitData(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def run_baseline_logistic_regression(split: SplitData) -> Dict[str, float]:
	model = LogisticRegression(max_iter=2000)
	model.fit(split.x_train, split.y_train)

	probs = model.predict_proba(split.x_test)[:, 1]
	pred = (probs >= 0.5).astype(np.int64)

	return {
		"accuracy": float(accuracy_score(split.y_test, pred)),
		"f1": float(f1_score(split.y_test, pred)),
		"auc": float(roc_auc_score(split.y_test, probs)),
	}


def build_node_feature_matrix(graph: nx.Graph) -> Tuple[np.ndarray, Dict[str, int]]:
	nodes = sorted(graph.nodes())
	node_to_idx = {n: i for i, n in enumerate(nodes)}

	degrees = np.array([graph.degree(n) for n in nodes], dtype=np.float32)
	max_degree = max(float(degrees.max()), 1.0)

	rows: List[List[float]] = []
	for i, node in enumerate(nodes):
		attrs = graph.nodes[node]
		node_type = str(attrs.get("node_type", "unknown"))

		is_member = 1.0 if node_type == "member" else 0.0
		is_user = 1.0 if node_type == "user" else 0.0
		is_benchmark = 1.0 if node_type == "benchmark" else 0.0
		has_party = 1.0 if attrs.get("party") else 0.0

		deg = float(degrees[i])
		rows.append([
			deg / max_degree,
			math.log1p(deg),
			is_member,
			is_user,
			is_benchmark,
			has_party,
		])

	x = np.asarray(rows, dtype=np.float32)
	return x, node_to_idx


def build_pyg_tensors(
	graph: nx.Graph,
	pairs: Sequence[Tuple[str, str]],
	labels: np.ndarray,
) -> Dict[str, Any]:
	if torch is None:
		raise RuntimeError("PyTorch is required for GNN training but is not installed.")

	node_x, node_to_idx = build_node_feature_matrix(graph)

	edge_rows = []
	edge_cols = []
	edge_weights = []
	pos_rows = []
	pos_cols = []
	neg_rows = []
	neg_cols = []
	for u, v, attrs in graph.edges(data=True):
		ui = node_to_idx[u]
		vi = node_to_idx[v]
		sign = float(attrs.get("sign", 1))
		prop_weight = 1.0
		edge_rows.extend([ui, vi])
		edge_cols.extend([vi, ui])
		edge_weights.extend([prop_weight, prop_weight])
		if sign > 0:
			pos_rows.extend([ui, vi])
			pos_cols.extend([vi, ui])
		else:
			neg_rows.extend([ui, vi])
			neg_cols.extend([vi, ui])

	pair_index = np.array([[node_to_idx[u], node_to_idx[v]] for (u, v) in pairs], dtype=np.int64)

	return {
		"x": torch.tensor(node_x, dtype=torch.float32),
		"edge_index": torch.tensor([edge_rows, edge_cols], dtype=torch.long),
		"pos_edge_index": torch.tensor([pos_rows, pos_cols], dtype=torch.long),
		"neg_edge_index": torch.tensor([neg_rows, neg_cols], dtype=torch.long),
		"edge_weight": torch.tensor(edge_weights, dtype=torch.float32),
		"pair_index": torch.tensor(pair_index, dtype=torch.long),
		"y": torch.tensor(labels, dtype=torch.float32),
	}


class SignedLinkPredictor(nn.Module):
	def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.35):
		super().__init__()
		self.pos_conv1 = GCNConv(in_dim, hidden_dim)
		self.neg_conv1 = GCNConv(in_dim, hidden_dim)
		self.pos_conv2 = GCNConv(hidden_dim, hidden_dim)
		self.neg_conv2 = GCNConv(hidden_dim, hidden_dim)
		self.dropout = dropout
		self.cls = nn.Sequential(
			nn.Linear(hidden_dim * 4, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 1),
		)

	def encode(
		self,
		x: Any,
		pos_edge_index: Any,
		neg_edge_index: Any,
	) -> Any:
		h_pos = self.pos_conv1(x, pos_edge_index)
		h_neg = self.neg_conv1(x, neg_edge_index)
		h = F.relu(h_pos - h_neg)
		h = F.dropout(h, p=self.dropout, training=self.training)
		h_pos_2 = self.pos_conv2(h, pos_edge_index)
		h_neg_2 = self.neg_conv2(h, neg_edge_index)
		return F.relu(h_pos_2 - h_neg_2)

	def forward(
		self,
		x: Any,
		pos_edge_index: Any,
		neg_edge_index: Any,
		pair_index: Any,
	) -> Any:
		h = self.encode(x, pos_edge_index, neg_edge_index)
		u = h[pair_index[:, 0]]
		v = h[pair_index[:, 1]]
		edge_features = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
		logits = self.cls(edge_features).squeeze(1)
		return logits


def run_gnn(
	graph: nx.Graph,
	x: np.ndarray,
	y: np.ndarray,
	pairs: Sequence[Tuple[str, str]],
	model: Optional[nn.Module] = None,
	epochs: int = 120,
	hidden_dim: int = 128,
	lr: float = 0.005,
	dropout: float = 0.35,
	max_patience: int = 20,
) -> Tuple[Optional[Dict[str, float]], Optional[nn.Module], Optional[float]]:
	if not HAS_PYG or torch is None:
		print(
			"[WARN] Skipping GNN: torch-geometric and/or torch is not available in this interpreter."
		)
		print(f"[WARN] Active Python executable: {sys.executable}")
		return None, None, None

	split_idx = train_val_test_split_indices(y)

	tensors = build_pyg_tensors(graph, pairs, y)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tensors = {k: v.to(device) for k, v in tensors.items()}

	if model is None:
		model = SignedLinkPredictor(in_dim=tensors["x"].shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
	else:
		model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

	train_idx_t = torch.tensor(split_idx.train_idx, dtype=torch.long, device=device)
	val_idx_t = torch.tensor(split_idx.val_idx, dtype=torch.long, device=device)
	test_idx_t = torch.tensor(split_idx.test_idx, dtype=torch.long, device=device)
	best_state: Optional[Dict[str, Any]] = None
	best_val_auc = -float("inf")
	patience = 0

	for epoch in range(1, epochs + 1):
		model.train()
		optimizer.zero_grad()
		logits = model(
			tensors["x"],
			tensors["pos_edge_index"],
			tensors["neg_edge_index"],
			tensors["pair_index"],
		)
		loss = F.binary_cross_entropy_with_logits(logits[train_idx_t], tensors["y"][train_idx_t])
		loss.backward()
		optimizer.step()

		model.eval()
		with torch.no_grad():
			val_logits = model(
				tensors["x"],
				tensors["pos_edge_index"],
				tensors["neg_edge_index"],
				tensors["pair_index"],
			)
			val_probs = torch.sigmoid(val_logits[val_idx_t]).detach().cpu().numpy()
			val_true = tensors["y"][val_idx_t].detach().cpu().numpy().astype(np.int64)
			val_auc = roc_auc_score(val_true, val_probs)

		if val_auc > best_val_auc:
			best_val_auc = val_auc
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			patience = 0
		else:
			patience += 1

		if epoch % 20 == 0 or epoch == epochs:
			print(f"[GNN] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

		if patience >= max_patience:
			print(f"[GNN] Early stopping at epoch {epoch:03d} | Best Val AUC: {best_val_auc:.4f}")
			break

	if best_state is not None:
		model.load_state_dict(best_state)

	model.eval()
	with torch.no_grad():
		logits = model(
			tensors["x"],
			tensors["pos_edge_index"],
			tensors["neg_edge_index"],
			tensors["pair_index"],
		)
		probs = torch.sigmoid(logits[test_idx_t]).detach().cpu().numpy()
		y_true = tensors["y"][test_idx_t].detach().cpu().numpy().astype(np.int64)

	y_pred = (probs >= 0.5).astype(np.int64)
	metrics = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred)),
		"auc": float(roc_auc_score(y_true, probs)),
	}
	return metrics, model, best_val_auc


def run_gnn_sweep(
	graph: nx.Graph,
	benchmark_graph: Optional[nx.Graph],
	benchmark_x: Optional[np.ndarray],
	benchmark_pairs: Optional[Sequence[Tuple[str, str]]],
	benchmark_y: Optional[np.ndarray],
	x: np.ndarray,
	y: np.ndarray,
	pairs: Sequence[Tuple[str, str]],
	pretrain_epochs: int,
	finetune_epochs: int,
	sweep_grid: Sequence[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, float]], Optional[nn.Module], Optional[Dict[str, Any]]]:
	best_metrics: Optional[Dict[str, float]] = None
	best_model: Optional[nn.Module] = None
	best_config: Optional[Dict[str, Any]] = None
	best_score = -float("inf")

	for i, config in enumerate(sweep_grid, start=1):
		print(
			f"\n[SWEEP] Trial {i}/{len(sweep_grid)} | hidden_dim={config['hidden_dim']} | lr={config['lr']} | dropout={config['dropout']}"
		)
		trial_model: Optional[nn.Module] = None
		if benchmark_graph is not None and benchmark_pairs is not None and benchmark_y is not None:
			print("[SWEEP] Pretraining benchmark model for this trial...")
			_, trial_model, _ = run_gnn(
				graph=benchmark_graph,
				x=benchmark_x if benchmark_x is not None else np.zeros((benchmark_graph.number_of_edges(), 1), dtype=np.float32),
				y=benchmark_y,
				pairs=benchmark_pairs,
				epochs=pretrain_epochs,
				hidden_dim=config["hidden_dim"],
				lr=config["lr"],
				dropout=config["dropout"],
			)

		trial_metrics, trained_model, val_auc = run_gnn(
			graph=graph,
			x=x,
			y=y,
			pairs=pairs,
			model=trial_model,
			epochs=finetune_epochs,
			hidden_dim=config["hidden_dim"],
			lr=config["lr"],
			dropout=config["dropout"],
		)
		if trial_metrics is None or val_auc is None:
			continue
		print_metrics(f"Sweep Trial {i} Test", trial_metrics)
		print(f"[SWEEP] Trial {i} val_auc={val_auc:.4f}")
		if val_auc > best_score:
			best_score = val_auc
			best_metrics = trial_metrics
			best_model = trained_model
			best_config = config

	return best_metrics, best_model, best_config


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
	print(f"\n=== {title} ===")
	for k in ["accuracy", "f1", "auc"]:
		v = metrics.get(k, math.nan)
		print(f"{k:>9}: {v:.4f}")


def save_graph_plots(
	label: str,
	graph_summary: Dict[str, int],
	balance: Dict[str, float],
	output_dir: Path,
) -> Optional[Path]:
	if not HAS_MPL:
		print("[WARN] matplotlib is not installed; skipping plot generation.")
		return None

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_path = output_dir / f"{label}_graph_stats.png"

	fig, axes = plt.subplots(1, 2, figsize=(11, 4))
	fig.suptitle(f"{label} Graph Statistics")

	axes[0].bar(["Positive", "Negative"], [graph_summary["positive_edges"], graph_summary["negative_edges"]], color=["#2a9d8f", "#e76f51"])
	axes[0].set_title("Signed Edge Counts")
	axes[0].set_ylabel("Edges")

	axes[1].bar(["Balanced", "Unbalanced"], [balance["balanced_triangles"], max(balance["triangles"] - balance["balanced_triangles"], 0)], color=["#264653", "#f4a261"])
	axes[1].set_title("Triangle Balance")
	axes[1].set_ylabel("Triads")

	fig.tight_layout()
	fig.savefig(plot_path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	print(f"[PLOT] Saved {plot_path}")
	return plot_path


def save_metric_comparison_plot(
	label: str,
	baseline_metrics: Dict[str, float],
	gnn_metrics: Optional[Dict[str, float]],
	output_dir: Path,
) -> Optional[Path]:
	if not HAS_MPL:
		print("[WARN] matplotlib is not installed; skipping plot generation.")
		return None

	if gnn_metrics is None:
		return None

	output_dir.mkdir(parents=True, exist_ok=True)
	plot_path = output_dir / f"{label}_metric_comparison.png"

	metrics = ["accuracy", "f1", "auc"]
	baseline_vals = [baseline_metrics[m] for m in metrics]
	gnn_vals = [gnn_metrics[m] for m in metrics]

	x = np.arange(len(metrics))
	width = 0.36
	fig, ax = plt.subplots(figsize=(8, 4.5))
	ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#457b9d")
	ax.bar(x + width / 2, gnn_vals, width, label="GNN", color="#e63946")
	ax.set_xticks(x)
	ax.set_xticklabels([m.upper() for m in metrics])
	ax.set_ylim(0, 1)
	ax.set_title(f"{label} Model Comparison")
	ax.set_ylabel("Score")
	ax.legend()
	fig.tight_layout()
	fig.savefig(plot_path, dpi=180, bbox_inches="tight")
	plt.close(fig)
	print(f"[PLOT] Saved {plot_path}")
	return plot_path


def run_pipeline(
	election_path: Path,
	tweets_path: Path,
	rivalry_config_path: Path,
	use_gnn: bool,
	run_sweep: bool,
	use_external_benchmark: bool,
	benchmark_dataset: str,
	benchmark_cache_dir: Path,
	benchmark_max_edges: int,
	pretrain_epochs: int,
	finetune_epochs: int,
	max_negative_per_party_pair: int,
	default_hidden_dim: int,
	default_lr: float,
	default_dropout: float,
) -> None:
	set_seed(RANDOM_SEED)

	rivalry_map = load_party_rivalry_map(rivalry_config_path)
	election_df = load_election_data(election_path)
	tweets_df = load_tweets_data(tweets_path)

	pretrained_model: Optional[nn.Module] = None
	benchmark_graph: Optional[nx.Graph] = None
	bx: Optional[np.ndarray] = None
	by: Optional[np.ndarray] = None
	bpairs: Optional[Sequence[Tuple[str, str]]] = None

	if use_external_benchmark:
		benchmark_graph = load_external_signed_benchmark(
			dataset_name=benchmark_dataset,
			cache_dir=benchmark_cache_dir,
			max_edges=benchmark_max_edges,
		)
		benchmark_summary = summarize_signed_graph(benchmark_graph)
		print("=== External Signed Benchmark Summary ===")
		print(f"dataset: {benchmark_dataset}")
		for k, v in benchmark_summary.items():
			print(f"{k:>16}: {v}")

		bench_balance = compute_balance_metrics(benchmark_graph, max_samples=10000)
		print("\n=== Benchmark Structural Balance ===")
		print(f"{'triangles':>16}: {int(bench_balance['triangles'])}")
		print(f"{'balanced':>16}: {int(bench_balance['balanced_triangles'])}")
		print(f"{'balance_ratio':>16}: {bench_balance['balance_ratio']:.4f}")

		bx, by, bpairs = build_edge_classification_dataset(benchmark_graph)
		bsplit = train_test_split_edges(bx, by, test_size=0.2)
		bench_baseline_metrics = run_baseline_logistic_regression(bsplit)
		print_metrics("Benchmark Baseline (Logistic Regression)", bench_baseline_metrics)

		if use_gnn and not run_sweep:
			print("\n[INFO] Pretraining GNN on benchmark graph...")
			bench_gnn_metrics, pretrained_model, _ = run_gnn(
				graph=benchmark_graph,
				x=bx,
				y=by,
				pairs=bpairs,
				epochs=pretrain_epochs,
				hidden_dim=default_hidden_dim,
				lr=default_lr,
				dropout=default_dropout,
			)
			if bench_gnn_metrics is not None:
				print_metrics("Benchmark Pretrain GNN (PyG GCN)", bench_gnn_metrics)
				if benchmark_graph is not None:
					save_graph_plots(
						label=f"benchmark_{benchmark_dataset}",
						graph_summary=benchmark_summary,
						balance=bench_balance,
						output_dir=Path("plots"),
					)

		print("")

	election_graph = build_election_signed_graph(
		election_df,
		rivalry_map=rivalry_map,
		max_negative_per_party_pair=(None if max_negative_per_party_pair <= 0 else max_negative_per_party_pair),
	)
	tweet_graph = build_tweet_signed_graph(tweets_df)
	graph = merge_graphs(election_graph, tweet_graph)

	summary = summarize_signed_graph(graph)
	print("=== Signed Graph Summary ===")
	for k, v in summary.items():
		print(f"{k:>16}: {v}")

	balance = compute_balance_metrics(graph, max_samples=10000)
	print("\n=== Structural Balance Diagnostics ===")
	print(f"{'triangles':>16}: {int(balance['triangles'])}")
	print(f"{'balanced':>16}: {int(balance['balanced_triangles'])}")
	print(f"{'balance_ratio':>16}: {balance['balance_ratio']:.4f}")

	x, y, pairs = build_edge_classification_dataset(graph)
	split = train_test_split_edges(x, y, test_size=0.2)
	baseline_metrics = run_baseline_logistic_regression(split)
	print_metrics("Baseline (Logistic Regression)", baseline_metrics)

	saved_gnn_metrics: Optional[Dict[str, float]] = None

	if use_gnn:
		if run_sweep:
			sweep_grid = [
				{"hidden_dim": 128, "lr": 0.005, "dropout": 0.35},
				{"hidden_dim": 192, "lr": 0.003, "dropout": 0.40},
				{"hidden_dim": 256, "lr": 0.002, "dropout": 0.45},
			]
			best_metrics, _, best_config = run_gnn_sweep(
				graph=graph,
				benchmark_graph=benchmark_graph if use_external_benchmark else None,
				benchmark_x=bx if use_external_benchmark else None,
				benchmark_pairs=bpairs if use_external_benchmark else None,
				benchmark_y=by if use_external_benchmark else None,
				x=x,
				y=y,
				pairs=pairs,
				pretrain_epochs=pretrain_epochs,
				finetune_epochs=finetune_epochs,
				sweep_grid=sweep_grid,
			)
			if best_metrics is not None and best_config is not None:
				print_metrics(
					f"Best Sweep Result (hidden_dim={best_config['hidden_dim']}, lr={best_config['lr']}, dropout={best_config['dropout']})",
					best_metrics,
				)
				saved_gnn_metrics = best_metrics
			return

		if pretrained_model is not None:
			print("\n[INFO] Fine-tuning pretrained benchmark GNN on Pakistan graph...")
		else:
			print("\n[INFO] Training GNN from scratch on Pakistan graph...")

		gnn_metrics, _, _ = run_gnn(
			graph=graph,
			x=x,
			y=y,
			pairs=pairs,
			model=pretrained_model,
			epochs=finetune_epochs,
			hidden_dim=default_hidden_dim,
			lr=default_lr,
			dropout=default_dropout,
		)
		if gnn_metrics is not None:
			title = "Signed GNN (Fine-tuned from Benchmark)" if pretrained_model is not None else "Signed GNN (PyG GCN)"
			print_metrics(title, gnn_metrics)
			saved_gnn_metrics = gnn_metrics

	save_graph_plots(
		label="pakistan",
		graph_summary=summary,
		balance=balance,
		output_dir=Path("plots"),
	)
	save_metric_comparison_plot(
		label="pakistan",
		baseline_metrics=baseline_metrics,
		gnn_metrics=saved_gnn_metrics,
		output_dir=Path("plots"),
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Signed link polarity prediction + structural balance diagnostics (MVP)."
	)
	parser.add_argument(
		"--election",
		type=Path,
		default=Path("election2024.csv"),
		help="Path to Pakistan election CSV.",
	)
	parser.add_argument(
		"--tweets",
		type=Path,
		default=Path("tweets.csv"),
		help="Path to tweets CSV.",
	)
	parser.add_argument(
		"--rivalry-config",
		type=Path,
		default=Path("party_rivalries.json"),
		help="Path to JSON file mapping party to rival parties.",
	)
	parser.add_argument(
		"--use-gnn",
		action="store_true",
		help="Run the PyTorch Geometric signed GNN model in addition to baseline.",
	)
	parser.add_argument(
		"--sweep",
		action="store_true",
		help="Run a small hyperparameter sweep over hidden size, learning rate, and dropout.",
	)
	parser.add_argument(
		"--use-external-benchmark",
		action="store_true",
		help="Download and evaluate an external signed benchmark graph (Slashdot/Wiki-RfA).",
	)
	parser.add_argument(
		"--benchmark-dataset",
		type=str,
		default="slashdot",
		choices=["slashdot", "wikirfa"],
		help="Which external signed benchmark to ingest.",
	)
	parser.add_argument(
		"--benchmark-cache-dir",
		type=Path,
		default=Path("data_cache"),
		help="Directory to store downloaded benchmark files.",
	)
	parser.add_argument(
		"--benchmark-max-edges",
		type=int,
		default=0,
		help="Maximum benchmark edges to parse; 0 means use the full dataset.",
	)
	parser.add_argument(
		"--max-negative-per-party-pair",
		type=int,
		default=0,
		help="Maximum rival edges per party pair; 0 means use all cross-party pairs.",
	)
	parser.add_argument(
		"--pretrain-epochs",
		type=int,
		default=80,
		help="Epochs for benchmark pretraining when external benchmark is enabled.",
	)
	parser.add_argument(
		"--finetune-epochs",
		type=int,
		default=120,
		help="Epochs for Pakistan graph fine-tuning/training.",
	)
	parser.add_argument(
		"--hidden-dim",
		type=int,
		default=128,
		help="Hidden dimension for the signed GNN.",
	)
	parser.add_argument(
		"--lr",
		type=float,
		default=0.005,
		help="Learning rate for the signed GNN.",
	)
	parser.add_argument(
		"--dropout",
		type=float,
		default=0.35,
		help="Dropout for the signed GNN.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_pipeline(
		election_path=args.election,
		tweets_path=args.tweets,
		rivalry_config_path=args.rivalry_config,
		use_gnn=args.use_gnn,
		run_sweep=args.sweep,
		use_external_benchmark=args.use_external_benchmark,
		benchmark_dataset=args.benchmark_dataset,
		benchmark_cache_dir=args.benchmark_cache_dir,
		benchmark_max_edges=args.benchmark_max_edges,
		pretrain_epochs=args.pretrain_epochs,
		finetune_epochs=args.finetune_epochs,
		max_negative_per_party_pair=args.max_negative_per_party_pair,
		default_hidden_dim=args.hidden_dim,
		default_lr=args.lr,
		default_dropout=args.dropout,
	)


if __name__ == "__main__":
	main()
