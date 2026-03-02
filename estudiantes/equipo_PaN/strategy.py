# estudiantes/equipo_PaN/strategy.py
"""
Estrategia híbrida Wordle (4/5/6, uniform/frequency).

Robustez:
- NO bitmasks (1<<idx). Usamos alfabeto dinámico por juego.
- 'ñ' cuenta como letra distinta sin romper nada.
- Precomputamos codes + uniq_count.

Diseño requerido:
1) Levenshtein con DP + memoria O(L) y cache dict.
2) Estructuras centrales en diccionarios: self.cache, self.word_info, self.stats.
3) Levenshtein SOLO para ranking dentro de palabras válidas (tras filter_candidates).
4) self.exploit + regla concreta (T[L], p_star[L]) + time guard fallback.

Mejoras para reducir “volado”:
- Parámetros por modo (uniform vs frequency).
- Non-words SOLO en uniform (frequency usa palabras del vocab).
- En frequency temprano, el pool de guesses se construye sobre un espacio más amplio
  (top-K del vocab completo) para incluir “preguntas” más informativas (tipo Entropy).
- El pool mezcla ranking por prob y ranking por info (cobertura) para no sesgarse.
"""

from __future__ import annotations

import math
import time
import numpy as np

from strategy import Strategy, GameConfig
from wordle_env import filter_candidates


# ------------------------- Levenshtein DP O(L) -------------------------
def levenshtein_dp_ol(a: str, b: str, cache: dict[tuple[str, str], int]) -> int:
    """Levenshtein con DP usando 2 filas (memoria O(len(b)))."""
    if a == b:
        return 0
    key = (a, b) if a <= b else (b, a)
    if key in cache:
        return cache[key]

    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cb = b[j - 1]
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            v = ins if ins < dele else dele
            if sub < v:
                v = sub
            cur[j] = v
        prev = cur

    d = prev[lb]
    cache[key] = d
    return d


def pattern_code(secret_codes: tuple[int, ...], guess_codes: tuple[int, ...]) -> int:
    """Feedback Wordle (0/1/2) -> entero base-3. Maneja duplicados."""
    L = len(secret_codes)
    pat = [0] * L

    remaining: dict[int, int] = {}
    for s in secret_codes:
        remaining[s] = remaining.get(s, 0) + 1

    # verdes
    for i in range(L):
        g = guess_codes[i]
        s = secret_codes[i]
        if g == s:
            pat[i] = 2
            remaining[g] -= 1

    # amarillos
    for i in range(L):
        if pat[i] == 2:
            continue
        g = guess_codes[i]
        cnt = remaining.get(g, 0)
        if cnt > 0:
            pat[i] = 1
            remaining[g] = cnt - 1

    code = 0
    for x in pat:
        code = code * 3 + x
    return code


def safe_argmax(x: np.ndarray) -> int:
    return int(np.argmax(x)) if x.size else 0


# ------------------------------ Estrategia ------------------------------
class MiEstrategia(Strategy):
    @property
    def name(self) -> str:
        return "MiEstrategia_equipo_PaN"

    def begin_game(self, config: GameConfig) -> None:
        # Dicts obligatorios
        self.cache: dict = {}
        self.word_info: dict[str, dict] = {}
        self.stats: dict = {}

        self.exploit = False
        self._L = int(config.word_length)
        self._mode = str(config.mode)  # "uniform" o "frequency"
        self._allow_non_words = bool(config.allow_non_words)

        # time guard (5s por juego incluye begin_game + guesses)
        self._t0 = time.perf_counter()
        self.stats["hard_time_limit"] = 4.85

        vocab = list(config.vocabulary)
        n = len(vocab)

        # Probabilidades (normalizadas)
        probs = np.empty(n, dtype=np.float64)
        for i, w in enumerate(vocab):
            probs[i] = float(config.probabilities.get(w, 0.0))
        s = float(probs.sum())
        probs[:] = probs / s if s > 0 else (1.0 / max(1, n))
        logp = np.log(probs + 1e-12)

        # Alfabeto dinámico (incluye 'ñ' y cualquier otro carácter presente)
        alphabet = sorted({c for w in vocab for c in w})
        A = len(alphabet)
        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        idx_to_char = alphabet  # lista indexable

        # Codes y uniq_count (sin bitmask)
        L = self._L
        codes = np.empty((n, L), dtype=np.uint16)  # uint16 por si A > 255
        uniq_count = np.empty(n, dtype=np.uint8)
        for i, w in enumerate(vocab):
            row = [char_to_idx[c] for c in w]
            codes[i, :] = row
            uniq_count[i] = len(set(row))

        # Precompute: tuples por palabra (para pattern_code sin reconvertir)
        code_tuples = [tuple(int(x) for x in codes[i]) for i in range(n)]

        all_idx = np.arange(n, dtype=np.int32)

        self.stats["data"] = {
            "words": vocab,
            "probs": probs,
            "logp": logp,
            "codes": codes,
            "code_tuples": code_tuples,
            "uniq": uniq_count,
            "word_to_idx": {w: i for i, w in enumerate(vocab)},
            "alphabet": alphabet,
            "A": A,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "all_idx": all_idx,
        }

        # Caches
        self.cache["lev"] = {}  # memo Levenshtein
        self.cache["cand_state"] = {
            "hist_len": 0,
            "candidates": vocab,
            "cand_idx": all_idx.copy(),
        }

        # ------------------ Parámetros por modo ------------------
        self.stats["T"] = {
            "uniform": {4: 4, 5: 6, 6: 8},
            "frequency": {4: 5, 5: 8, 6: 12},
        }

        # K: top-k candidatos por prob para “candidatos probables”
        self.stats["K"] = {
            "uniform": {4: 220, 5: 280, 6: 320},
            "frequency": {4: 260, 5: 340, 6: 480},
        }

        # K_global: en frequency temprano, usamos un “guess space” más amplio (vocab completo)
        # para meter “preguntas” informativas tipo Entropy sin O(n^2).
        self.stats["K_global"] = {4: 450, 5: 700, 6: 950}

        # N_eval[L]: máximo de candidatos (top por prob) para estimar entropía
        self.stats["N_eval"] = {4: 900, 5: 650, 6: 550}
        # K_eval[L]: tamaño del pool final (guesses) a evaluar con entropía
        self.stats["K_eval"] = {4: 60, 5: 55, 6: 55}  # un poquito más en 6 para variedad

        self.stats["p_star"] = {
            "uniform": {4: 0.60, 5: 0.60, 6: 0.60},
            "frequency": {4: 0.18, 5: 0.12, 6: 0.12},
        }

        # Si ya vamos tarde: forzar exploit SOLO cuando el set ya es manejable
        self.stats["force_exploit_after"] = {"uniform": 4, "frequency": 3}
        self.stats["force_exploit_cand_cap"] = {"uniform": 500, "frequency": 350}

        # Pesos scoring barato (solo para armar pool; la decisión final usa entropía)
        self.stats["w_pos"] = 1.00
        self.stats["w_uniq"] = 0.70
        self.stats["w_prob"] = 0.80 if self._mode == "frequency" else 0.15
        self.stats["w_rep_pen"] = 0.35
        self.stats["w_lev"] = 0.40

        # Starter cover guess (posible non-word) — SOLO lo usaremos en uniform
        self.stats["starter_guess"] = self._build_cover_guess(
            cand_idx=self.cache["cand_state"]["cand_idx"], history=[]
        )

    def guess(self, history: list[tuple[str, tuple[int, ...]]]) -> str:
        # Time guard
        if (time.perf_counter() - self._t0) > self.stats["hard_time_limit"]:
            return self._fallback_maxprob(history)

        candidates, cand_idx = self._get_candidates(history)
        if not candidates:
            data = self.stats["data"]
            cand_idx = data["all_idx"]

        if cand_idx.size == 1:
            return self.stats["data"]["words"][int(cand_idx[0])]

        if self.exploit:
            return self._pick_best_candidate_fast(cand_idx)

        L = self._L
        T = self.stats["T"][self._mode].get(L, 6)
        p_star = self.stats["p_star"][self._mode][L]

        data = self.stats["data"]
        probs = data["probs"][cand_idx]
        s = float(probs.sum())
        probs_norm = probs / s if s > 0 else probs
        max_p = float(probs_norm.max()) if probs_norm.size else 0.0

        # Exploit: tamaño pequeño, o max_p alto
        # (max_p solo dispara fuerte cuando ya no hay miles de candidatos)
        if cand_idx.size <= T or (max_p >= p_star and cand_idx.size <= 600):
            self.exploit = True
            return self._pick_best_candidate_fast(cand_idx)

        # Si ya vamos tarde, forzar exploit SOLO si candidatos ya son “manejables”
        force_after = self.stats["force_exploit_after"][self._mode]
        cap = self.stats["force_exploit_cand_cap"][self._mode]
        if len(history) >= force_after and cand_idx.size <= cap:
            self.exploit = True
            return self._pick_best_candidate_fast(cand_idx)

        # Primer guess: cover SOLO en uniform
        if len(history) == 0 and self._allow_non_words and self._mode == "uniform":
            g0 = self.stats["starter_guess"]
            if isinstance(g0, str) and len(g0) == L:
                return g0

        pos_freq, let_freq = self._letter_stats(cand_idx)
        proto = self._prototype_from_posfreq(pos_freq)

        # top_idx: candidatos “probables”
        top_idx = self._topk_indices_by_prob(cand_idx, self.stats["K"][self._mode][L])

        # guess_space (solo para construir pool):
        # en frequency temprano y con muchos candidatos, usa top del vocab completo
        # para permitir “preguntas” más informativas que no siempre son candidatas.
        guess_space = top_idx
        if (
            self._mode == "frequency"
            and L == 6
            and len(history) <= 1
            and cand_idx.size > 900
        ):
            all_idx = data["all_idx"]
            guess_space = self._topk_indices_by_prob(all_idx, self.stats["K_global"][L])

        # Entropía aprox si hay tiempo
        if (time.perf_counter() - self._t0) < 4.2:
            pool = self._build_guess_pool(guess_space, pos_freq, let_freq, history, proto)
            gbest = self._best_by_expected_information(cand_idx, pool)
            if gbest is not None:
                return gbest

        return self._best_by_cheap_score(cand_idx, top_idx, pos_freq, let_freq, history, proto)

    # ---------------- candidates (filter_candidates) ----------------
    def _get_candidates(self, history):
        state = self.cache["cand_state"]
        word_to_idx = self.stats["data"]["word_to_idx"]

        hlen = len(history)
        if hlen == 0:
            state["hist_len"] = 0
            state["candidates"] = self.stats["data"]["words"]
            state["cand_idx"] = self.stats["data"]["all_idx"]
            return state["candidates"], state["cand_idx"]

        if state.get("hist_len", 0) == hlen - 1:
            g, pat = history[-1]
            cand = filter_candidates(state["candidates"], g, pat)
        else:
            cand = self.stats["data"]["words"]
            for g, pat in history:
                cand = filter_candidates(cand, g, pat)

        if not cand:
            state["hist_len"] = hlen
            state["candidates"] = []
            state["cand_idx"] = np.empty(0, dtype=np.int32)
            return [], state["cand_idx"]

        idx = np.array([word_to_idx[w] for w in cand], dtype=np.int32)
        state["hist_len"] = hlen
        state["candidates"] = cand
        state["cand_idx"] = idx
        return cand, idx

    # ---------------- fast pick ----------------
    def _fallback_maxprob(self, history):
        _, cand_idx = self._get_candidates(history)
        if cand_idx.size == 0:
            return self.stats["data"]["words"][0]
        return self._pick_best_candidate_fast(cand_idx)

    def _pick_best_candidate_fast(self, cand_idx: np.ndarray) -> str:
        data = self.stats["data"]
        probs = data["probs"][cand_idx]
        best_local = safe_argmax(probs)
        return data["words"][int(cand_idx[best_local])]

    # ---------------- stats ----------------
    def _letter_stats(self, cand_idx: np.ndarray):
        data = self.stats["data"]
        codes = data["codes"]
        A = int(data["A"])

        probs = data["probs"][cand_idx]
        s = float(probs.sum())
        probs = probs / s if s > 0 else np.full_like(probs, 1.0 / max(1, probs.size), dtype=np.float64)

        sub = codes[cand_idx]  # (m, L)
        L = self._L

        pos_freq = np.zeros((L, A), dtype=np.float64)
        for i in range(L):
            pos_freq[i] = np.bincount(sub[:, i], weights=probs, minlength=A)

        wrep = np.repeat(probs, L)
        let_freq = np.bincount(sub.reshape(-1), weights=wrep, minlength=A)
        return pos_freq, let_freq

    def _prototype_from_posfreq(self, pos_freq: np.ndarray) -> str:
        idx_to_char = self.stats["data"]["idx_to_char"]
        return "".join(idx_to_char[int(np.argmax(pos_freq[i]))] for i in range(self._L))

    # ---------------- cover guess ----------------
    def _build_cover_guess(self, cand_idx: np.ndarray, history) -> str:
        """Guess de cobertura (letras frecuentes y distintas). Puede ser non-word."""
        data = self.stats["data"]
        idx_to_char = data["idx_to_char"]

        pos_freq, let_freq = self._letter_stats(cand_idx)
        L = self._L

        used_chars = set()
        for g, _ in history:
            used_chars.update(g)

        order = np.argsort(-let_freq)
        chosen = []
        for li in order:
            c = idx_to_char[int(li)]
            if c in used_chars:
                continue
            chosen.append(int(li))
            if len(chosen) >= L:
                break
        if len(chosen) < L:
            for li in order:
                chosen.append(int(li))
                if len(chosen) >= L:
                    break

        chosen_set = set(chosen)
        out = []
        for i in range(L):
            best = None
            best_val = -1.0
            for li in list(chosen_set):
                v = float(pos_freq[i, li])
                if v > best_val:
                    best_val = v
                    best = li
            if best is None:
                best = chosen[i]
            out.append(idx_to_char[int(best)])
            chosen_set.discard(best)

        guess = "".join(out)
        return guess[:L].ljust(L, idx_to_char[0])

    # ---------------- top-k ----------------
    def _topk_indices_by_prob(self, cand_idx: np.ndarray, k: int) -> np.ndarray:
        probs = self.stats["data"]["probs"][cand_idx]
        m = probs.size
        if m <= k:
            return cand_idx
        part = np.argpartition(-probs, kth=k - 1)[:k]
        top = cand_idx[part]
        order = np.argsort(-self.stats["data"]["probs"][top])
        return top[order]

    # ---------------- pool building ----------------
    def _build_guess_pool(self, guess_idx, pos_freq, let_freq, history, proto):
        """
        Arma pool para evaluar entropía:
        - mezcla top por "prob-score" y top por "info-score" para no sesgarse.
        - Levenshtein solo como desempate dentro de palabras del vocab (índices válidos).
        """
        data = self.stats["data"]
        words, codes, uniq = data["words"], data["codes"], data["uniq"]
        logp = data["logp"]

        seen = {g for g, _ in history}
        w_pos, w_uniq, w_prob, w_rep, w_lev = (
            self.stats["w_pos"],
            self.stats["w_uniq"],
            self.stats["w_prob"],
            self.stats["w_rep_pen"],
            self.stats["w_lev"],
        )
        lev_cache = self.cache["lev"]

        scored_prob = []
        scored_info = []

        for idx in guess_idx:
            i = int(idx)
            w = words[i]
            if w in seen:
                continue
            row = codes[i]

            # score por posiciones
            s_pos = 0.0
            for p in range(self._L):
                s_pos += float(pos_freq[p, int(row[p])])

            # score por letras únicas
            used = set()
            s_u = 0.0
            for p in range(self._L):
                li = int(row[p])
                if li in used:
                    continue
                used.add(li)
                s_u += float(let_freq[li])

            rep_pen = w_rep * (self._L - int(uniq[i]))

            base_info = w_pos * s_pos + w_uniq * s_u - rep_pen
            base_info -= w_lev * levenshtein_dp_ol(w, proto, lev_cache)

            base_prob = base_info + w_prob * float(logp[i])

            scored_info.append((base_info, w))
            scored_prob.append((base_prob, w))

        scored_prob.sort(key=lambda t: t[0], reverse=True)
        scored_info.sort(key=lambda t: t[0], reverse=True)

        k_eval = self.stats["K_eval"][self._L]
        # mezcla: mitad prob, mitad info (deduplicando)
        take_prob = max(10, k_eval // 2)
        take_info = k_eval

        pool = []
        used_w = set()

        for _, w in scored_prob[:take_prob]:
            if w not in used_w:
                used_w.add(w)
                pool.append(w)
            if len(pool) >= k_eval:
                break

        if len(pool) < k_eval:
            for _, w in scored_info[:take_info]:
                if w not in used_w:
                    used_w.add(w)
                    pool.append(w)
                if len(pool) >= k_eval:
                    break

        # Non-word cover SOLO en uniform (y solo al inicio)
        if self._mode == "uniform" and self._allow_non_words and len(history) <= 1:
            cover = self._build_cover_guess(self.stats["data"]["all_idx"], history)
            if cover not in used_w:
                pool.append(cover)

        if not pool:
            # fallback extremo
            pool = [data["words"][int(guess_idx[0])]]

        return pool

    def _best_by_cheap_score(self, cand_idx, top_idx, pos_freq, let_freq, history, proto):
        """Fallback barato: elige mejor score dentro de top_idx (candidatos probables)."""
        data = self.stats["data"]
        words, codes, uniq = data["words"], data["codes"], data["uniq"]
        logp = data["logp"]

        seen = {g for g, _ in history}

        # Non-word cover SOLO en uniform
        if (
            self._mode == "uniform"
            and self._allow_non_words
            and len(history) <= 1
            and cand_idx.size > 250
        ):
            cover = self._build_cover_guess(cand_idx, history)
            if cover not in seen:
                return cover

        w_pos, w_uniq, w_prob, w_rep, w_lev = (
            self.stats["w_pos"],
            self.stats["w_uniq"],
            self.stats["w_prob"],
            self.stats["w_rep_pen"],
            self.stats["w_lev"],
        )
        lev_cache = self.cache["lev"]

        best_w = words[int(top_idx[0])]
        best_s = -1e18

        for idx in top_idx:
            i = int(idx)
            w = words[i]
            if w in seen:
                continue
            row = codes[i]

            s_pos = 0.0
            for p in range(self._L):
                s_pos += float(pos_freq[p, int(row[p])])

            used = set()
            s_u = 0.0
            for p in range(self._L):
                li = int(row[p])
                if li in used:
                    continue
                used.add(li)
                s_u += float(let_freq[li])

            rep_pen = w_rep * (self._L - int(uniq[i]))
            score = w_pos * s_pos + w_uniq * s_u - rep_pen
            score += w_prob * float(logp[i])
            score -= w_lev * levenshtein_dp_ol(w, proto, lev_cache)

            if score > best_s:
                best_s = score
                best_w = w

        return best_w

    # ---------------- expected information (entropy approx) ----------------
    def _encode_word(self, w: str) -> tuple[int, ...]:
        m = self.stats["data"]["char_to_idx"]
        return tuple(m[c] for c in w)

    def _best_by_expected_information(self, cand_idx, guess_pool):
        """
        Elige g que maximiza entropía (normalizada) de los patrones inducidos,
        con bonus por p_hit (más fuerte en frequency).

        Evalúa patrones contra un subconjunto top-N_eval por probabilidad
        cuando hay muchos candidatos (anti-timeout).
        """
        if (time.perf_counter() - self._t0) > 4.35:
            return None

        data = self.stats["data"]
        probs_all = data["probs"][cand_idx]
        s = float(probs_all.sum())
        probs = (
            probs_all / s
            if s > 0
            else np.full_like(probs_all, 1.0 / max(1, probs_all.size), dtype=np.float64)
        )

        L = self._L
        N_eval = self.stats["N_eval"].get(L, probs.size)
        m = probs.size

        if m <= N_eval:
            eval_idx = cand_idx
            probs_eval = probs
        else:
            k = N_eval
            part = np.argpartition(-probs, k - 1)[:k]
            eval_idx = cand_idx[part]
            probs_eval = probs[part]

        code_tuples = data["code_tuples"]
        cand_codes = [code_tuples[int(idx)] for idx in eval_idx]

        n_patterns = 3 ** L
        w2i = data["word_to_idx"]

        # idx->prob para p_hit en O(1)
        idx_prob = {int(idx): float(probs[j]) for j, idx in enumerate(cand_idx)}

        # bonus p_hit
        if self._mode == "frequency":
            alpha_hit = {4: 0.30, 5: 0.40, 6: 0.48}.get(L, 0.40)
        else:
            alpha_hit = 0.10

        best_guess, best_obj = None, float("inf")

        for g in guess_pool:
            if len(g) != L:
                continue
            if (time.perf_counter() - self._t0) > 4.5:
                break

            gc = self._encode_word(g)
            masses = [0.0] * n_patterns

            for i, sc in enumerate(cand_codes):
                pc = pattern_code(sc, gc)
                masses[pc] += float(probs_eval[i])

            tot = sum(masses)
            if tot <= 0.0:
                continue

            ent = 0.0
            inv_tot = 1.0 / tot
            for mk in masses:
                if mk > 0.0:
                    pk = mk * inv_tot
                    ent -= pk * math.log(pk + 1e-12)

            gi = w2i.get(g)
            p_hit = float(idx_prob.get(int(gi), 0.0)) if gi is not None else 0.0

            obj = -ent - alpha_hit * p_hit
            if obj < best_obj:
                best_obj, best_guess = obj, g

        return best_guess
