"""
SL-GCN: a graph convolutional arm for the ablation.

Why this exists
---------------
The promoted model flattens each frame's 65 landmarks into a 195-vector and runs
Conv1D over time. That throws away the one thing we know for free: the skeleton
is a GRAPH. Nothing tells the model that wrist-elbow-shoulder are connected, so
it has to rediscover human kinematics from 4,284 clips.

SL-GCN (Jiang et al., SAM-SLR) instead convolves ALONG the skeleton. AI4Bharat
benchmarked it at 93.5% on INCLUDE, the dataset we train on, against 91.2% for
ST-GCN and 90.4% for a transformer, so on our exact data it is the best-evidenced
architecture available.

Three ideas, in the order they matter here:

1. DecoupleSCN. In an ordinary GCN every feature channel shares one adjacency
   matrix, which caps expressiveness and overfits, and ordinary dropout works
   poorly in GCNs. Decoupling gives each channel GROUP its own learnable
   adjacency, so one group can specialise in hand-internal structure while
   another tracks arm kinematics. (Cheng et al., ECCV 2020.)

2. STC attention. Spatial, temporal and channel attention cascaded, which
   joints matter, which frames matter, which features matter.

3. A smaller graph. SAM-SLR reduces to 27 keypoints and reports that this HELPS.
   We keep all 65 by default but expose CORE_27 to test the same idea, because
   "fewer nodes, more structure" runs against the instinct to add capacity and
   is worth measuring rather than assuming.

Everything here is einsum / matmul / conv2d, ops TF.js supports, so a winning
arm can actually ship to the browser rather than being a paper result.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------- the skeleton
# Index layout of the 65-point frame, matching features.select_points():
#   0..22   pose 0..22        (MediaPipe pose, legs 23..32 already dropped)
#   23..43  left hand 0..20
#   44..64  right hand 0..20
POSE, LH, RH = 0, 23, 44

# MediaPipe pose topology, restricted to the 23 points we keep.
_POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),      # nose -> left eye -> left ear
    (0, 4), (4, 5), (5, 6), (6, 8),      # nose -> right eye -> right ear
    (9, 10),                             # mouth corners
    (11, 12),                            # shoulders
    # Anatomical bridges MediaPipe's own topology omits. Without these the face
    # (0-8), the mouth (9-10) and the torso (11+) are three disconnected
    # components: verified: only 9 of 65 nodes were reachable. A GCN cannot
    # propagate across a disconnected graph, so head movement (which carries
    # negation in ISL) could never reach the hands that qualify it.
    (0, 11), (0, 12),                    # nose -> shoulders, joins head to body
    (0, 9), (0, 10),                     # nose -> mouth, joins the mouth cluster
    (11, 13), (13, 15),                  # left arm
    (12, 14), (14, 16),                  # right arm
    (15, 17), (15, 19), (15, 21),        # left hand stubs on the pose skeleton
    (16, 18), (16, 20), (16, 22),        # right hand stubs
]

# MediaPipe hand topology, 21 points, applied to each hand.
_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm closure
]

# The cross-links that make this ONE graph rather than three disconnected ones.
# Pose wrist 15/16 is the same physical joint as hand root 0, without these the
# model cannot relate handshape to where the hand is in body space, which is
# most of what distinguishes one sign from another.
_BRIDGE = [(15, LH + 0), (16, RH + 0)]

N_POINTS = 65

# SAM-SLR's reduction: drop the face detail and keep the joints that actually
# carry manual signing. Exposed so "fewer nodes" can be tested, not assumed.
CORE_27 = (
    [0, 11, 12, 13, 14, 15, 16]                       # nose, shoulders, arms
    + [LH + i for i in (0, 4, 8, 12, 16, 20, 5, 9, 13, 17)]   # left: root + tips + knuckles
    + [RH + i for i in (0, 4, 8, 12, 16, 20, 5, 9, 13, 17)]   # right
)


def build_adjacency(num_nodes: int = N_POINTS,
                    subset: list[int] | None = None) -> np.ndarray:
    """Normalised adjacency partitions, shape (3, V, V).

    Three partitions follow ST-GCN's spatial configuration: self-connections,
    inward edges, and outward edges. Splitting them lets the model treat "my own
    state", "what my parent is doing" and "what my children are doing" as
    different relationships rather than averaging all three together.

    Symmetric normalisation (D^-1/2 A D^-1/2) keeps activations from exploding
    at high-degree nodes: the wrists here, which carry six edges each.
    """
    edges = list(_POSE_EDGES)
    for a, b in _HAND_EDGES:
        edges.append((LH + a, LH + b))
        edges.append((RH + a, RH + b))
    edges += _BRIDGE

    if subset is not None:
        # CONTRACT, don't filter. Keeping only edges whose endpoints both survive
        # silently shatters the graph: CORE_27 keeps fingertips and knuckles but
        # drops the joints between them, so naive filtering left 17 of 27 nodes
        # unreachable: fingertips floating free of the hand they belong to.
        #
        # Instead, two kept nodes are joined when a path exists between them in
        # the full skeleton passing only through dropped nodes. That preserves
        # real anatomical adjacency at lower resolution, which is what SAM-SLR's
        # 27-node reduction is actually doing.
        import collections

        full_adj = collections.defaultdict(set)
        for a, b in edges:
            full_adj[a].add(b)
            full_adj[b].add(a)

        keep = {v: i for i, v in enumerate(subset)}
        keep_set = set(subset)
        contracted = set()
        for src in subset:
            # walk outward, passing through dropped nodes only
            seen = {src}
            queue = collections.deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in full_adj[cur]:
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    if nxt in keep_set:
                        contracted.add((keep[src], keep[nxt]))
                    else:
                        queue.append(nxt)   # keep walking through dropped nodes
        edges = sorted(contracted)
        num_nodes = len(subset)

    self_link = np.eye(num_nodes, dtype=np.float32)
    inward = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    outward = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for a, b in edges:
        inward[b, a] = 1.0    # child <- parent
        outward[a, b] = 1.0   # parent <- child

    def norm(A: np.ndarray) -> np.ndarray:
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        return A / deg

    return np.stack([self_link, norm(inward), norm(outward)]).astype(np.float32)


# ------------------------------------------------------------------- the layers
class DecoupleSCN(layers.Layer):
    """Decoupled spatial graph convolution.

    Channels are split into `groups`, and each group learns its own residual on
    top of the fixed skeleton adjacency. That is the decoupling: without it every
    channel is forced through one shared A, which is both a capacity ceiling and
    a strong overfitting pressure on a 1,441-clip dataset.

    The fixed skeleton is kept as a non-trainable base so the layer starts from
    real anatomy and learns a correction, rather than discovering the human body
    from scratch.
    """

    def __init__(self, out_ch: int, A: np.ndarray, groups: int = 4, **kw):
        super().__init__(**kw)
        self.out_ch = out_ch
        self.groups = groups
        self.A_init = A                      # (K, V, V)
        self.K, self.V = A.shape[0], A.shape[1]

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        if self.out_ch % self.groups:
            raise ValueError(f"out_ch {self.out_ch} not divisible by groups {self.groups}")

        self.A_fixed = self.add_weight(
            name="A_fixed", shape=(self.K, self.V, self.V),
            initializer=keras.initializers.Constant(self.A_init), trainable=False)
        # One learnable adjacency residual PER GROUP, this is the decoupling.
        self.A_res = self.add_weight(
            name="A_res", shape=(self.groups, self.K, self.V, self.V),
            initializer=keras.initializers.RandomNormal(stddev=1e-3), trainable=True)
        # Per-partition pointwise transform, produced in one kernel.
        self.W = self.add_weight(
            name="W", shape=(self.K, in_ch, self.out_ch),
            initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(
            name="bias", shape=(self.out_ch,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, V, Cin)
        g, cg = self.groups, self.out_ch // self.groups
        outs = []
        for k in range(self.K):
            # transform channels for this partition, then aggregate over the graph
            h = tf.einsum("btvc,co->btvo", x, self.W[k])          # (B,T,V,out)
            hg = tf.reshape(h, tf.concat([tf.shape(h)[:3], [g, cg]], axis=0))
            A_k = self.A_fixed[k] + self.A_res[:, k]              # (g,V,V)
            # each group aggregates with its own adjacency
            agg = tf.einsum("btvgc,gvw->btwgc", hg, A_k)
            outs.append(tf.reshape(agg, tf.shape(h)))
        y = tf.add_n(outs) + self.bias
        return y

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.out_ch)

    def get_config(self):
        return {**super().get_config(), "out_ch": self.out_ch, "groups": self.groups}


class STCAttention(layers.Layer):
    """Spatial, temporal and channel attention, cascaded.

    Cheap (three small dense layers) and it makes the model's focus inspectable , 
    the spatial weights say which joints a sign actually depends on, which is
    worth having when explaining a prediction to a clinician.
    """

    def __init__(self, reduction: int = 4, **kw):
        super().__init__(**kw)
        self.reduction = reduction

    def build(self, input_shape):
        _, T, V, C = input_shape
        r = max(1, C // self.reduction)
        self.s = keras.Sequential([layers.Dense(r, activation="relu"),
                                   layers.Dense(V, activation="sigmoid")], name="spatial")
        self.t = keras.Sequential([layers.Dense(r, activation="relu"),
                                   layers.Dense(T, activation="sigmoid")], name="temporal")
        self.c = keras.Sequential([layers.Dense(r, activation="relu"),
                                   layers.Dense(C, activation="sigmoid")], name="channel")
        super().build(input_shape)

    def call(self, x):
        s = self.s(tf.reduce_mean(x, axis=[1, 3]))[:, None, :, None]   # (B,1,V,1)
        x = x * s
        t = self.t(tf.reduce_mean(x, axis=[2, 3]))[:, :, None, None]   # (B,T,1,1)
        x = x * t
        c = self.c(tf.reduce_mean(x, axis=[1, 2]))[:, None, None, :]   # (B,1,1,C)
        return x * c

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction}


def sl_gcn_unit(x, out_ch, A, stride=1, groups=4, temporal_kernel=9, name=""):
    """One SL-GCN block: graph conv -> STC attention -> temporal conv, + residual."""
    res = x
    in_ch = int(x.shape[-1])

    y = DecoupleSCN(out_ch, A, groups=groups, name=f"{name}_scn")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = STCAttention(name=f"{name}_stc")(y)

    # Temporal convolution: Conv2D over (T, V) with a kernel that spans time only,
    # so joints stay independent here and mixing happens in the graph conv.
    y = layers.Conv2D(out_ch, (temporal_kernel, 1), strides=(stride, 1),
                      padding="same", use_bias=False, name=f"{name}_tcn")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    if in_ch != out_ch or stride != 1:
        res = layers.Conv2D(out_ch, (1, 1), strides=(stride, 1),
                            use_bias=False, name=f"{name}_down")(res)
        res = layers.BatchNormalization(name=f"{name}_downbn")(res)

    return layers.Activation("relu", name=f"{name}_out")(layers.Add(name=f"{name}_add")([y, res]))


def build_slgcn(seq_len: int, num_nodes: int, in_ch: int, n_classes: int,
                A: np.ndarray | None = None, width: int = 32,
                groups: int = 4) -> keras.Model:
    """SL-GCN over (T, V, C). Deliberately narrow, and that is the point.

    width=32 gives 461,830 parameters against the promoted Conv1D model's
    512,994: 0.90x. Matching capacity is not a detail: at the obvious width=64
    this arm carries 2.39x the parameters, and a win there would be
    uninterpretable, because "graph structure helps" and "more capacity helps"
    would be perfectly confounded. Held slightly UNDER the baseline so any gain
    is attributable to structure alone.
    """
    if A is None:
        A = build_adjacency(num_nodes)
    inp = keras.Input(shape=(seq_len, num_nodes, in_ch))

    x = layers.BatchNormalization(name="in_bn")(inp)
    x = sl_gcn_unit(x, width, A, groups=groups, name="b1")
    x = sl_gcn_unit(x, width * 2, A, stride=2, groups=groups, name="b2")
    x = layers.Dropout(0.2, name="drop1")(x)
    x = sl_gcn_unit(x, width * 4, A, stride=2, groups=groups, name="b3")

    x = layers.GlobalAveragePooling2D(name="gap")(x)   # over T and V
    x = layers.Dropout(0.4, name="drop2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="head")(x)

    m = keras.Model(inp, out, name="sl_gcn")
    # Same optimiser, loss and LR as train.build_model, another axis held fixed,
    # so the arms differ only in architecture.
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def standardise_graph(X4: np.ndarray) -> np.ndarray:
    """(N,T,V,C) -> (N,T,V,C), normalised EXACTLY as train.standardise does.

    The statistics must match the flat path bit for bit, same whole-clip mean
    and std over every value: or the two arms are not comparable and, worse,
    the parity contract with features.ts silently no longer describes this model.
    Only the final reshape differs: the graph is preserved instead of flattened.
    """
    flat = X4.reshape(len(X4), -1)
    m = flat.mean(axis=1, keepdims=True)
    s = np.maximum(flat.std(axis=1, keepdims=True), 1e-6)
    return ((flat - m) / s).reshape(X4.shape).astype(np.float32)
