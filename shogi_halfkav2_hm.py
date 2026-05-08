"""
将棋 HalfKAv2_hm Feature Set (#14)

Stockfish chess HalfKAv2_hm (model/modules/features/halfka_v2_hm.py) を将棋向けに移植。

設計詳細: docs/v1.2-research/2026-05-06-halfkav2-shogi-design.md
リファレンス実装: scripts/halfkav2_design.py (cshogi-based)

Phase 1 (本ファイル): Python feature_block 定義 + halfkp.py 互換 API
Phase 2 (将来): C++ data loader (training/nnue-pytorch/YaneuraOu/source/eval/nnue/features/) 拡張
Phase 3 (将来): Rust 推論側 (engine-core/src/nnue_halfkav2.rs) 新設

注意:
- 本ファイルは halfkp.py / halfka.py と同じく chess.Board API を使うが、
  実際の training は C++ data loader 経由なので Python の get_active_features は
  factorizer 用 metadata 提供のみ
- shogi 用定数を使用 (NUM_SQ=81, NUM_PT=14, mirror あり)
"""

import chess  # nnue-pytorch のしきたりに従う (将棋でも chess module を mock として使用)
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

# 将棋向け定数 (cshogi 由来)
NUM_SQ = 81  # 9 × 9
NUM_FILES = 9
NUM_RANKS = 9
NUM_PT = 14  # 駒種 (歩/香/桂/銀/金/角/飛/玉 + 成銀/成桂/成香/と/馬/竜)
NUM_PT_COLORED = NUM_PT * 2  # 28 (我方 + 敵方)

# Phase 2: HalfKAv2_hm 定数
NUM_BUCKETS_FILES = 5  # mirror 後の有効 file 数 (file 5-9 のみ、1-4 は mirror で 9-6 へ)
NUM_BUCKETS = NUM_BUCKETS_FILES * NUM_RANKS  # 45 (王バケット)
NUM_PLANES = NUM_SQ * NUM_PT_COLORED  # 81 * 28 = 2268
NUM_INPUTS = NUM_PLANES * NUM_BUCKETS  # 2268 * 45 = 102,060

# 持ち駒 features (別の独立 feature group)
HAND_MAX = [18, 4, 4, 4, 4, 2, 2]  # 歩, 香, 桂, 銀, 金, 角, 飛
NUM_HAND_FEATURES = sum(HAND_MAX) * 2  # 我方 + 敵方 = 76


def _build_king_buckets():
    """KingBuckets[81]: 各マスをバケット番号 (0-44) または -1 にマップ"""
    buckets = [-1] * NUM_SQ
    for file_idx in range(NUM_FILES):
        for rank_idx in range(NUM_RANKS):
            sq = file_idx * NUM_FILES + rank_idx
            if file_idx >= 4:  # file 5-9 (mirror 後の有効範囲)
                buckets[sq] = (file_idx - 4) * NUM_RANKS + rank_idx
    return buckets


KING_BUCKETS = _build_king_buckets()


def shogi_orient(is_sente_pov: bool, sq: int, ksq: int) -> int:
    """
    将棋向け orient_flip:
    - 王 file 1-4 (idx 0-3) → 水平 mirror (file_idx → 8 - file_idx)
    - 王 file 5 (idx 4) → mirror なし (中央)
    - 王 file 6-9 → mirror なし
    - 後手視点なら rank flip (rank_idx → 8 - rank_idx)

    注: 9 ファイルは奇数のため、file 5 (idx 4) は中央で mirror 不可、そのまま。
    """
    kfile = ksq // NUM_FILES
    file_idx = sq // NUM_FILES
    rank_idx = sq % NUM_FILES

    if kfile < 4:
        file_idx = 8 - file_idx

    if not is_sente_pov:
        rank_idx = 8 - rank_idx

    return file_idx * NUM_FILES + rank_idx


def halfkav2_hm_idx(
    is_sente_pov: bool, king_sq: int, sq: int, piece_type: int, piece_color: int
) -> int:
    """
    HalfKAv2_hm feature index 計算

    Args:
        is_sente_pov: 先手視点か
        king_sq: 自分の王のマス (0-80)
        sq: 駒のマス (0-80)
        piece_type: 1-14 (cshogi piece type)
        piece_color: 0=BLACK 駒, 1=WHITE 駒
    """
    is_own = piece_color == (0 if is_sente_pov else 1)
    p_idx = (piece_type - 1) * 2 + (0 if is_own else 1)

    o_sq = shogi_orient(is_sente_pov, sq, king_sq)
    o_ksq = shogi_orient(is_sente_pov, king_sq, king_sq)
    bucket = KING_BUCKETS[o_ksq]
    if bucket < 0:
        return -1

    return o_sq + p_idx * NUM_SQ + bucket * NUM_PLANES


class Features(FeatureBlock):
    """HalfKAv2_hm feature block (将棋向け)"""

    def __init__(self):
        super(Features, self).__init__(
            "ShogiHalfKAv2_hm",
            0x7AF32F16,  # version hash (Stockfish のと衝突しないように shogi prefix)
            OrderedDict([("ShogiHalfKAv2_hm", NUM_INPUTS)]),
        )

    def get_active_features(self, board):
        """
        Python 直接実装は使わない (C++ data loader 経由が原則)。
        factorizer 用 metadata 提供のみ。
        """
        raise Exception(
            "Python 直接抽出は未実装。training は C++ data loader 経由が必須。"
            " Phase 2 で yaneuraou-pytorch C++ 側の "
            " training/nnue-pytorch/YaneuraOu/source/eval/nnue/features/half_ka_v2_hm.cpp "
            "を実装する必要あり。"
        )


class FactorizedFeatures(FeatureBlock):
    """Factorized 版 (HalfKAv2_hm + A virtual feature)"""

    def __init__(self):
        super(FactorizedFeatures, self).__init__(
            "ShogiHalfKAv2_hm^",
            0x7AF32F16,
            OrderedDict(
                [
                    ("ShogiHalfKAv2_hm", NUM_INPUTS),
                    ("A", NUM_PLANES),  # virtual: 駒種 × マス (王位置非依存)
                ]
            ),
        )

    def get_active_features(self, board):
        raise Exception(
            "Python 直接抽出は未実装。C++ data loader 経由必須。"
        )

    def get_feature_factors(self, idx):
        """factorizer サポート: 各 real index に対応する virtual feature index を返す"""
        if idx >= self.num_real_features:
            raise Exception("Feature must be real")

        # idx = o_sq + p_idx * NUM_SQ + bucket * NUM_PLANES
        # virtual A: o_sq + p_idx * NUM_SQ (bucket non-dependent)
        a_idx = idx % NUM_PLANES

        return [idx, self.get_factor_base_feature("A") + a_idx]


def get_feature_block_clss():
    """features.py からの discovery 用"""
    return [Features, FactorizedFeatures]
