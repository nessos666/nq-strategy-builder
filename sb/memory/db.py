from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from sb.models import BacktestResult

_DEFAULT_DB = Path(__file__).parent.parent.parent / "output" / "builder.db"
TIER_A_MIN_TRADES: int = 20
"""Minimale durchschnittliche OOS-Trades fuer Tier A (statistische Belastbarkeit)."""
_SQLITE_TIMEOUT_SECONDS = 30.0
_SQLITE_BUSY_TIMEOUT_MS = int(_SQLITE_TIMEOUT_SECONDS * 1000)
_SQLITE_LOCK_RETRIES = 5
_SQLITE_LOCK_RETRY_DELAY_SECONDS = 0.1


class BuilderDB:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to create database directory: {self.db_path.parent}"
            ) from exc

        try:
            self.conn = sqlite3.connect(
                str(self.db_path), timeout=_SQLITE_TIMEOUT_SECONDS
            )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Unable to open database: {self.db_path}") from exc
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()
        try:
            self._create_tables()
        except Exception:
            self.close()
            raise

    def _configure_connection(self) -> None:
        self._execute_with_retry(
            lambda: self.conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        )
        self._execute_with_retry(lambda: self.conn.execute("PRAGMA foreign_keys=ON"))
        self._execute_with_retry(lambda: self.conn.execute("PRAGMA journal_mode=WAL"))

    def _is_locked_error(self, exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def _execute_with_retry(self, operation: Any) -> Any:
        delay = _SQLITE_LOCK_RETRY_DELAY_SECONDS
        for attempt in range(_SQLITE_LOCK_RETRIES):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if (
                    not self._is_locked_error(exc)
                    or attempt == _SQLITE_LOCK_RETRIES - 1
                ):
                    raise
                try:
                    self.conn.rollback()
                except sqlite3.Error:
                    pass
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("SQLite retry loop exited unexpectedly")

    def _execute_write(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Cursor:
        def operation() -> sqlite3.Cursor:
            with self.conn:
                return self.conn.execute(sql, params)

        try:
            return self._execute_with_retry(operation)
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Database integrity error while executing write: {exc}"
            ) from exc
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Database write failed: {exc}") from exc

    def _execute_script(self, sql: str) -> None:
        def operation() -> None:
            with self.conn:
                self.conn.executescript(sql)

        try:
            self._execute_with_retry(operation)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Database schema update failed: {exc}") from exc

    def _normalize_limit(self, limit: int | str) -> int:
        try:
            normalized_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid limit: {limit}") from exc
        return max(normalized_limit, 0)

    def _create_tables(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS build_runs (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                idea                TEXT NOT NULL,
                trials              INTEGER NOT NULL,
                avg_oos_pf          REAL,
                tier                TEXT,
                session             TEXT,
                is_robust           INTEGER,
                holdout_pf          REAL,
                holdout_trades      INTEGER,
                holdout_validated   INTEGER DEFAULT 0,
                created_at          TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER REFERENCES build_runs(id),
                params      TEXT NOT NULL,
                pf          REAL,
                winrate     REAL,
                num_trades  INTEGER,
                score       REAL,
                rank        INTEGER,
                warnings    TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_results_score ON results(score DESC);
            CREATE TABLE IF NOT EXISTS suggestions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                idea          TEXT NOT NULL,
                model_runs    INTEGER,
                prob_ab       REAL,
                uncertainty   REAL,
                novelty       REAL,
                band          TEXT,
                status        TEXT DEFAULT 'pending',
                review_reason TEXT,
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status);
        """
        )
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Fügt fehlende Spalten zu bestehenden DBs hinzu (idempotent)."""
        cols = {
            row[1]
            for row in self._execute_with_retry(
                lambda: self.conn.execute("PRAGMA table_info(build_runs)")
            )
        }
        if "avg_oos_pf" not in cols:
            self._execute_write("ALTER TABLE build_runs ADD COLUMN avg_oos_pf REAL", ())
        if "tier" not in cols:
            self._execute_write("ALTER TABLE build_runs ADD COLUMN tier TEXT", ())
        if "session" not in cols:
            self._execute_write("ALTER TABLE build_runs ADD COLUMN session TEXT", ())
        if "is_robust" not in cols:
            self._execute_write(
                "ALTER TABLE build_runs ADD COLUMN is_robust INTEGER", ()
            )
        if "holdout_pf" not in cols:
            self._execute_write("ALTER TABLE build_runs ADD COLUMN holdout_pf REAL", ())
        if "holdout_trades" not in cols:
            self._execute_write(
                "ALTER TABLE build_runs ADD COLUMN holdout_trades INTEGER", ()
            )
        if "holdout_validated" not in cols:
            self._execute_write(
                "ALTER TABLE build_runs ADD COLUMN holdout_validated INTEGER DEFAULT 0",
                (),
            )
        if "pbo_score" not in cols:
            self._execute_write("ALTER TABLE build_runs ADD COLUMN pbo_score REAL", ())
        if "mc_pct_profitable" not in cols:
            self._execute_write(
                "ALTER TABLE build_runs ADD COLUMN mc_pct_profitable REAL", ()
            )
        self._execute_write(
            "CREATE INDEX IF NOT EXISTS idx_runs_tier ON build_runs(tier)", ()
        )

    def save_run(
        self,
        idea: str,
        trials: int,
        session: str | None = None,
        is_robust: bool | None = None,
    ) -> int:
        normalized_idea = idea.strip() if isinstance(idea, str) else ""
        if not normalized_idea:
            raise ValueError("idea must not be empty")
        try:
            normalized_trials = int(trials)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trials value: {trials}") from exc
        if normalized_trials <= 0:
            raise ValueError("trials must be greater than 0")

        normalized_session = session.strip() if isinstance(session, str) else None
        normalized_is_robust = int(is_robust) if is_robust is not None else None

        cur = self._execute_write(
            "INSERT INTO build_runs (idea, trials, session, is_robust) VALUES (?, ?, ?, ?)",
            (
                normalized_idea,
                normalized_trials,
                normalized_session,
                normalized_is_robust,
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def save_result(
        self,
        run_id: int,
        result: BacktestResult,
        score: float,
        rank: int,
        warnings: list[str],
    ) -> None:
        if run_id <= 0:
            raise ValueError("run_id must be greater than 0")
        try:
            params_json = json.dumps(result.params or {})
            warnings_json = json.dumps(warnings or [])
        except (TypeError, ValueError) as exc:
            raise ValueError("Result payload is not JSON serializable") from exc

        try:
            normalized_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid score value: {score}") from exc
        try:
            normalized_rank = int(rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rank value: {rank}") from exc

        try:
            self._execute_write(
                "INSERT INTO results (run_id, params, pf, winrate, num_trades, score, rank, warnings) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    params_json,
                    result.profit_factor,
                    result.winrate,
                    result.num_trades,
                    normalized_score,
                    normalized_rank,
                    warnings_json,
                ),
            )
        except ValueError as exc:
            if "FOREIGN KEY" in str(exc).upper():
                raise ValueError(f"Run {run_id} does not exist") from exc
            raise

    def compute_and_save_tier(
        self,
        run_id: int,
        pbo_score: float = float("nan"),
        mc_pct_profitable: float | None = None,
    ) -> str:
        """Berechnet avg OOS-PF über alle Fenster und setzt Tier A/B/C.

        Tier A erfordert avg_oos_pf >= 2.0 UND pbo_score < 0.5.
        NaN pbo_score (unbekannt) blockiert Tier A nicht.
        """
        try:
            normalized_run_id = int(run_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run_id: {run_id}") from exc
        if normalized_run_id <= 0:
            raise ValueError("run_id must be greater than 0")

        run_exists = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT 1 FROM build_runs WHERE id = ?", (normalized_run_id,)
            ).fetchone()
        )
        if run_exists is None:
            raise ValueError(f"Run {normalized_run_id} does not exist")

        rows = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT pf, num_trades FROM results WHERE run_id = ?",
                (normalized_run_id,),
            ).fetchall()
        )
        valid_pfs: list[float] = []
        total_trades: int = 0
        result_count: int = 0
        for row in rows:
            pf = row[0]
            nt = row[1]
            result_count += 1
            # NULL num_trades wird absichtlich als 0 gezaehlt (konservativer Ansatz:
            # fehlende Trade-Daten senken den Durchschnitt und blockieren Tier A)
            if nt is not None:
                try:
                    total_trades += int(nt)
                except (TypeError, ValueError):
                    pass
            if pf is None:
                continue
            try:
                pf_value = float(pf)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pf_value) or pf_value <= 0.0:
                continue
            valid_pfs.append(pf_value)

        avg_pf = sum(valid_pfs) / len(valid_pfs) if valid_pfs else None
        avg_trades: float = total_trades / result_count if result_count > 0 else 0
        try:
            normalized_pbo = float(pbo_score)
        except (TypeError, ValueError):
            normalized_pbo = float("nan")
        if not math.isfinite(normalized_pbo) or not 0.0 <= normalized_pbo <= 1.0:
            normalized_pbo = float("nan")

        trades_ok = avg_trades >= TIER_A_MIN_TRADES
        pbo_ok = not math.isfinite(normalized_pbo) or normalized_pbo < 0.5
        if avg_pf is not None and avg_pf >= 2.0 and pbo_ok and trades_ok:
            tier = "A"
        elif avg_pf is not None and avg_pf >= 1.5:
            tier = "B"
        else:
            tier = "C"
        pbo_db = round(normalized_pbo, 4) if math.isfinite(normalized_pbo) else None
        mc_db = round(mc_pct_profitable, 4) if mc_pct_profitable is not None else None
        self._execute_write(
            "UPDATE build_runs SET avg_oos_pf = ?, tier = ?, pbo_score = ?, mc_pct_profitable = ? WHERE id = ?",
            (
                round(avg_pf, 4) if avg_pf is not None else None,
                tier,
                pbo_db,
                mc_db,
                normalized_run_id,
            ),
        )
        return tier

    def save_holdout_result(
        self, run_id: int, holdout_pf: float, holdout_trades: int
    ) -> None:
        """Speichert das Holdout-Ergebnis für einen bestehenden Run."""
        try:
            normalized_run_id = int(run_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run_id: {run_id}") from exc
        if normalized_run_id <= 0:
            raise ValueError("run_id must be greater than 0")

        run_exists = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT 1 FROM build_runs WHERE id = ?", (normalized_run_id,)
            ).fetchone()
        )
        if run_exists is None:
            raise ValueError(f"Run {normalized_run_id} does not exist")

        try:
            pf_value = round(float(holdout_pf), 4)
            trades_value = int(holdout_trades)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid holdout values: {exc}") from exc

        self._execute_write(
            "UPDATE build_runs SET holdout_pf = ?, holdout_trades = ?, holdout_validated = 1 "
            "WHERE id = ?",
            (pf_value, trades_value, normalized_run_id),
        )

    def backfill_missing_metadata(self) -> int:
        """Füllt session + is_robust für Runs die NULL haben aus vorhandenen Daten."""
        _SESSION_KEYWORDS = {"london": "london", "ny": "ny", "asia": "asia"}
        rows = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT id, idea FROM build_runs WHERE session IS NULL OR is_robust IS NULL"
            ).fetchall()
        )
        updated = 0
        for row in rows:
            run_id = row["id"]
            idea = (row["idea"] or "").lower()

            # Session aus Idea-Namen ableiten
            session = None
            for keyword, value in _SESSION_KEYWORDS.items():
                if keyword in idea:
                    session = value
                    break

            # is_robust aus results-Fenstern berechnen (alle PF > 1.0 und > 0 Trades)
            window_pfs = self._execute_with_retry(
                lambda rid=run_id: self.conn.execute(
                    "SELECT pf, num_trades FROM results WHERE run_id = ?", (rid,)
                ).fetchall()
            )
            if window_pfs:
                robust = all(
                    r["pf"] is not None
                    and r["pf"] > 1.0
                    and r["num_trades"] is not None
                    and r["num_trades"] > 0
                    for r in window_pfs
                )
                is_robust = int(robust)
            else:
                is_robust = None

            self._execute_write(
                "UPDATE build_runs SET session = COALESCE(session, ?), "
                "is_robust = COALESCE(is_robust, ?) WHERE id = ?",
                (session, is_robust, run_id),
            )
            updated += 1
        return updated

    def get_registry(self, tier: str | None = None) -> list[dict[str, Any]]:
        """Gibt alle registrierten Strategien zurück, optional nach Tier gefiltert.

        Kein Limit – gibt alle Runs zurück. NULL avg_oos_pf (Tier C ohne Trades)
        wird an das Ende sortiert via COALESCE.
        """
        normalized_tier = tier.strip().upper() if isinstance(tier, str) else None
        if normalized_tier == "":
            normalized_tier = None
        if normalized_tier is not None and normalized_tier not in {"A", "B", "C"}:
            raise ValueError(f"Invalid tier filter: {tier}")

        base_sql = """
            SELECT b.*,
                   CAST(ROUND(AVG(r.num_trades)) AS INTEGER) AS avg_trades
            FROM build_runs b
            LEFT JOIN results r ON r.run_id = b.id
            {where}
            GROUP BY b.id
            ORDER BY COALESCE(b.avg_oos_pf, -1) DESC
        """
        if normalized_tier:
            sql = base_sql.format(where="WHERE b.tier = ?")
            rows = self._execute_with_retry(
                lambda: self.conn.execute(sql, (normalized_tier,)).fetchall()
            )
        else:
            sql = base_sql.format(where="")
            rows = self._execute_with_retry(lambda: self.conn.execute(sql).fetchall())
        return [dict(r) for r in rows]

    def get_registry_counts(self) -> dict[str, int]:
        """Gibt Anzahl der Runs pro Tier + Gesamt zurück."""
        rows = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT tier, COUNT(*) FROM build_runs GROUP BY tier"
            ).fetchall()
        )
        counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "total": 0}
        for row in rows:
            tier, count = row[0], row[1]
            if tier in counts:
                counts[tier] = count
            counts["total"] += count
        return counts

    def get_best_results(self, limit: int | str = 10) -> list[dict[str, Any]]:
        normalized_limit = self._normalize_limit(limit)
        rows = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT * FROM results ORDER BY score DESC, id ASC LIMIT ?",
                (normalized_limit,),
            ).fetchall()
        )
        return [self._decode_row(dict(r)) for r in rows]

    def get_best_result_for_idea(self, idea: str) -> dict[str, Any] | None:
        normalized_idea = idea.strip() if isinstance(idea, str) else ""
        if not normalized_idea:
            return None
        row = self._execute_with_retry(
            lambda: self.conn.execute(
                """
                SELECT r.*
                FROM results AS r
                JOIN build_runs AS b ON b.id = r.run_id
                WHERE LOWER(TRIM(b.idea)) = LOWER(?)
                ORDER BY r.score DESC, r.id ASC
                LIMIT 1
                """,
                (normalized_idea,),
            ).fetchone()
        )
        return self._decode_row(dict(row)) if row is not None else None

    def find_runs_by_idea(self, idea: str) -> list[dict[str, Any]]:
        normalized_idea = idea.strip() if isinstance(idea, str) else ""
        if not normalized_idea:
            return []
        rows = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT * FROM build_runs WHERE LOWER(TRIM(idea)) = LOWER(?) ORDER BY created_at DESC, id DESC",
                (normalized_idea,),
            ).fetchall()
        )
        return [dict(r) for r in rows]

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        for key in ("params", "warnings"):
            value = row.get(key)
            if not isinstance(value, str):
                continue
            try:
                row[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
        return row

    def save_suggestion(
        self,
        idea: str,
        model_runs: int,
        prob_ab: float,
        uncertainty: float,
        novelty: float,
        band: str,
    ) -> int:
        """Speichert einen Meta-Learner Vorschlag mit Status 'pending'."""
        cur = self._execute_write(
            """
            INSERT INTO suggestions (idea, model_runs, prob_ab, uncertainty, novelty, band)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                idea,
                model_runs,
                round(prob_ab, 4),
                round(uncertainty, 4),
                round(novelty, 4),
                band,
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def get_pending_suggestions(self) -> list[dict[str, Any]]:
        """Gibt alle Vorschlaege mit status='pending' zurueck."""
        rows = self._execute_with_retry(
            lambda: self.conn.execute(
                "SELECT * FROM suggestions WHERE status = 'pending' ORDER BY prob_ab DESC"
            ).fetchall()
        )
        return [dict(r) for r in rows]

    def update_suggestion_status(
        self, suggestion_id: int, status: str, reason: str = ""
    ) -> None:
        """Setzt den Status eines Vorschlags (approved / rejected)."""
        if status not in {"approved", "rejected"}:
            raise ValueError(
                f"Ungueltiger Status: {status}. Nur 'approved' oder 'rejected' erlaubt."
            )
        cur = self._execute_write(
            "UPDATE suggestions SET status = ?, review_reason = ? WHERE id = ?",
            (status, reason, suggestion_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Suggestion {suggestion_id} nicht gefunden.")

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
