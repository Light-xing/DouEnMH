"""
Local game runner — drives bots and built-in agents with the bundled Botzone
judge to play complete Doudizhu games for end-to-end testing.

Architecture:
    runner.py
      ├── judge/judge_botzone_ver.py (in-process) — official Botzone judge
      │     logic: dealing, bidding, trick validation, scoring (~0 ms overhead)
      ├── bot.exe (subprocess, one per seat) — DouEnMH C++ inference build,
      │     spoken to over stdin/stdout JSON (Botzone KEEP_RUNNING protocol)
      └── in-process agents — random / greedy / pass baselines for any seat

The judge is bundled in tools/runner/judge/, so no external Botzone checkout
is required. Relative bot paths passed to --p0/--p1/--p2/--bot/--vs are tried
against the current working directory first, then against this script's
directory (tools/runner/), so both `--bot cpp/build/bot.exe` (from the repo
root) and `--bot ../../cpp/build/bot.exe` work. With no seat specified the
runner uses cpp/build/bot.exe plus two random agents.

Usage (from the repository root, or from tools/runner/ with runner.py):
    python tools/runner/runner.py --help

    # One bot + two random opponents (backward-compatible form)
    python tools/runner/runner.py --bot ../../cpp/build/bot.exe --verbose

    # Three bots fighting each other
    python tools/runner/runner.py --p0 ../../cpp/build/bot.exe \\
        --p1 ../../cpp/build/bot.exe --p2 ../../cpp/build/bot.exe

    # Bot vs greedy vs pass
    python tools/runner/runner.py --p0 ../../cpp/build/bot.exe --p1 greedy --p2 pass

    # Batch test
    python tools/runner/runner.py --bot ../../cpp/build/bot.exe --games 100 --seed 42

    # Head-to-head VS verification (same deal, roles swapped)
    python tools/runner/runner.py --bot ../../cpp/build/bot_a.exe \\
        --vs ../../cpp/build/bot_b.exe --vs-games 100 --vs-seed 42
"""

import subprocess
import json
import sys
import os
import random
import time
import argparse
import io
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Judge is bundled inside this directory (see THIRD_PARTY.md for provenance).
JUDGE_DIR = os.path.join(SCRIPT_DIR, "judge")
# All runner logs go to tools/runner/log/ (git-ignored; created on demand).
JUDGE_LOG_DIR = os.path.join(SCRIPT_DIR, "log")

# Default bot build, relative to tools/runner/ (i.e. cpp/build/bot.exe).
DEFAULT_BOT_SPEC = os.path.join(os.pardir, os.pardir, "cpp", "build", "bot.exe")

# Import judge module in-process (fast path — no subprocess startup cost)
if JUDGE_DIR not in sys.path:
    sys.path.insert(0, JUDGE_DIR)

# Intercept exit() and sys.exit() before importing judge
import builtins as _builtins
_builtins.exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))

import sys as _sys
_orig_sys_exit = _sys.exit
_sys.exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))

from judge_botzone_ver import main as _judge_main, errored as _judge_errored, checkPokerType


# =============================================================================
# Card utilities
# =============================================================================

def card_value(card_id):
    """Convert Botzone card ID (0-53) to sortable rank value."""
    if card_id == 53:
        return 16  # Big joker
    if card_id == 52:
        return 15  # Small joker
    return card_id // 4  # 0=3, 1=4, ..., 12=2


RANK_NAMES = ['3','4','5','6','7','8','9','10','J','Q','K','A','2']
# Botzone card id % 4 order: 0=hearts, 1=diamonds, 2=spades, 3=clubs.
SUIT_SYMBOLS = ['♥','♦','♤','♣']

def card_to_str(card_id):
    """Convert a Botzone card ID to human-readable string like '3♠' or '小王'."""
    if card_id == 52:
        return '小王'
    if card_id == 53:
        return '大王'
    rank = card_id // 4
    suit = card_id % 4
    return RANK_NAMES[rank] + SUIT_SYMBOLS[suit]

def format_hand(cards):
    """Convert a list of Botzone card IDs to a sorted, readable hand string."""
    if not cards:
        return '[]'
    sorted_cards = sorted(cards, key=lambda c: (card_value(c), c % 4))
    return ' '.join(card_to_str(c) for c in sorted_cards)

def group_by_rank(cards):
    """Group cards by rank (level). Returns {rank: [card_ids]}."""
    groups = {}
    for c in cards:
        r = card_value(c)
        groups.setdefault(r, []).append(c)
    return groups


# =============================================================================
# In-process agents (used by ScriptedPlayer)
# =============================================================================

class RandomAgent:
    """Random legal-move agent — chooses any valid play."""

    def bid(self, hand, bid_history):
        max_bid = max(bid_history) if bid_history else 0
        if max_bid == 3:
            return 0
        choices = [0]
        if max_bid < 1: choices.append(1)
        if max_bid < 2: choices.append(2)
        if max_bid < 3: choices.append(3)
        return random.choice(choices)

    def play(self, hand, last_combo, _history=None):
        if not last_combo:
            if hand:
                return [min(hand, key=card_value)]
            return []
        if random.random() < 0.7:
            return []
        if hand:
            return [random.choice(hand)]
        return []


class GreedyAgent:
    """Simple heuristic: always play the smallest possible combo."""

    def bid(self, hand, bid_history):
        groups = group_by_rank(hand)
        num_jokers = (15 in groups) + (16 in groups)
        num_bombs = sum(1 for v in groups.values() if len(v) == 4)
        num_twos = len(groups.get(12, []))
        strength = num_jokers * 5 + num_bombs * 3 + num_twos * 2

        max_bid = max(bid_history) if bid_history else 0
        if max_bid == 3:
            return 0
        if strength >= 6 and max_bid < 3:
            return 3
        if strength >= 3 and max_bid < 2:
            return 2
        if strength >= 1 and max_bid < 1:
            return 1
        return 0

    def play(self, hand, last_combo, _history=None):
        if not hand:
            return []
        if not last_combo:
            return [min(hand, key=card_value)]
        if len(last_combo) == 1:
            last_val = card_value(last_combo[0])
            bigger = [c for c in hand if card_value(c) > last_val]
            if bigger:
                return [min(bigger, key=card_value)]
        return []


# =============================================================================
# Player abstraction — each seat can be a bot, an agent, or a pass player
# =============================================================================

class PassPlayer:
    """Player that always passes / bids 0."""
    def __init__(self, seat): self.seat = seat
    def start(self): pass
    def stop(self): pass
    def label(self): return "pass"
    def respond(self, _request, _hand):
        return 0 if "bid" in _request else []


class ScriptedPlayer:
    """Player driven by an in-process agent (random/greedy)."""
    def __init__(self, agent_type, seat):
        self.seat = seat
        self.agent_type = agent_type
        agent_cls = {"random": RandomAgent, "greedy": GreedyAgent}[agent_type]
        self.agent = agent_cls()
    def start(self): pass
    def stop(self): pass
    def label(self): return self.agent_type
    def respond(self, request, _hand):
        if "bid" in request:
            return self.agent.bid(request.get("own", []), request.get("bid", []))
        else:
            history = request.get("history", [])
            last_combo = []
            if history:
                if history[-1]:
                    last_combo = history[-1]
                elif len(history) >= 2:
                    last_combo = history[-2]
            return self.agent.play(request.get("own", []), last_combo)


class BotPlayer:
    """Player driven by an external bot.exe via KEEP_RUNNING protocol."""
    def __init__(self, exe_path, seat, verbose=False):
        self.exe = exe_path
        self.seat = seat
        self.verbose = verbose
        self.process = None
        self.stderr_f = None
        self.is_first_turn = True

    def start(self):
        bot_dir = os.path.dirname(os.path.abspath(self.exe))
        log_path = os.path.join(bot_dir, f"bot_stderr_p{self.seat}.log")
        self.stderr_f = open(log_path, "w")
        self.process = subprocess.Popen(
            [self.exe],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_f,
            text=True,
            bufsize=1,
            cwd=bot_dir,
        )
        self.is_first_turn = True

    def stop(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
        if self.stderr_f:
            self.stderr_f.close()
            self.stderr_f = None

    def label(self):
        return os.path.basename(self.exe)

    def respond(self, request, _hand):
        if self.is_first_turn:
            wrapper = {"requests": [request], "responses": []}
            payload = json.dumps(wrapper)
            self.is_first_turn = False
        else:
            payload = json.dumps(request)

        if self.verbose:
            print(f"  [BOT P{self.seat} IN] {payload[:120]}...")

        self.process.stdin.write(payload + "\n")
        self.process.stdin.flush()

        resp_line = None
        for _ in range(10):
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"Bot P{self.seat} terminated unexpectedly (EOF)")
            stripped = line.strip()
            if stripped.startswith(">>>BOTZONE_REQUEST_KEEP_RUNNING"):
                continue
            if stripped and stripped.startswith("{"):
                resp_line = line
                break

        if resp_line is None:
            raise RuntimeError(
                f"Bot P{self.seat} returned no JSON response")

        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Bot P{self.seat} returned invalid JSON: {repr(resp_line[:200])}")

        if self.verbose:
            print(f"  [BOT P{self.seat} OUT] {json.dumps(resp)[:120]}")

        return resp["response"]


# =============================================================================
# Forced-bid wrapper — pre-assigns landlord in VS verification mode
# =============================================================================

class ForcedBidWrapper:
    """Wraps any player and overrides their bid response with a fixed value.

    Used in VS batch mode to pre-assign the landlord: the single-seat bot is
    forced to bid 3 (landlord), the two-seat bot to bid 0 (farmer).

    The real bid request is *always* forwarded to the inner player so the bot
    can initialise game state, load models, and transition past first-turn
    mode.  The bot's natural bid value is then replaced with the forced value
    before the judge sees it.

    In VS mode only P0 receives a real bid request from the judge (after P0
    bids 3 the judge terminates bidding immediately).  For P1 and P2 the
    wrapper injects a synthetic bid request on their first non-bid call so
    they too initialise before seeing any play requests.
    """

    def __init__(self, inner_player, forced_bid):
        self._inner = inner_player
        self._forced_bid = forced_bid
        self._bid_sent = False

    def start(self):
        self._inner.start()

    def stop(self):
        self._inner.stop()

    def label(self):
        return self._inner.label()

    def respond(self, request, hand):
        if "bid" in request:
            # Real bid from judge — forward to bot so it initialises and
            # flips is_first_turn, then override the return value.
            self._inner.respond(request, hand)
            self._bid_sent = True
            return self._forced_bid

        # Not a bid.  If the bot hasn't seen any bid yet (P1/P2 farmer in VS
        # mode — the judge terminated bidding after P0's 3), inject a
        # synthetic bid request to let the bot initialise its game state and
        # transition past first-turn mode.
        if not self._bid_sent:
            synth = {"own": list(hand), "bid": [3]}
            self._inner.respond(synth, hand)
            self._bid_sent = True

        return self._inner.respond(request, hand)


# =============================================================================
# Player factory
# =============================================================================

def _resolve_bot_path(spec):
    """Resolve a (possibly relative) bot executable path.

    A relative path is first tried against the current working directory (so
    `--bot cpp/build/bot.exe` works from the repository root), then against
    this script's directory (tools/runner/). Absolute paths are returned as-is.
    """
    if os.path.isabs(spec):
        return os.path.normpath(spec)
    if os.path.exists(spec):
        return os.path.normpath(spec)
    return os.path.normpath(os.path.join(SCRIPT_DIR, spec))


def _create_player(spec, seat, verbose=False):
    """Create a Player from a spec string.

    spec is one of: "random", "greedy", "pass", or a path to a bot executable.
    """
    spec = spec.strip()
    if spec in ("random", "greedy"):
        return ScriptedPlayer(spec, seat)
    elif spec == "pass":
        return PassPlayer(seat)
    else:
        return BotPlayer(_resolve_bot_path(spec), seat, verbose=verbose)


# =============================================================================
# Game Logger — dual output: game state + JSON I/O
# =============================================================================

def _player_role(player, landlord):
    if landlord < 0:
        return "bidding"
    if player == landlord:
        return "landlord"
    return "farmer"


class GameLogger:
    """Writes two per-turn log files during a game."""

    def __init__(self, log_dir, timestamp):
        os.makedirs(log_dir, exist_ok=True)
        self.state_f = open(
            os.path.join(log_dir, f"game_state_{timestamp}.log"),
            "w", encoding="utf-8")
        self.json_f = open(
            os.path.join(log_dir, f"json_io_{timestamp}.log"),
            "w", encoding="utf-8")

    def log_turn(self, turn_num, player, role, hand, action,
                 request_json, response_json):
        self.state_f.write(f"--- Turn {turn_num}, Player {player} ({role}) ---\n")
        self.state_f.write(f"Hand: {hand}\n")
        self.state_f.write(f"Action: {action}\n")
        self.state_f.write("\n")
        self.state_f.flush()

        self.json_f.write(f"--- Turn {turn_num}, Player {player} ---\n")
        self.json_f.write(f"Request:  {request_json}\n")
        self.json_f.write(f"Response: {response_json}\n")
        self.json_f.write("\n")
        self.json_f.flush()

    def close(self):
        self.state_f.close()
        self.json_f.close()

    def log_game_summary(self, play_type_stats, player_labels, landlord):
        """Write per-player play type statistics at end of game."""
        self.state_f.write("--- 本局出牌统计 ---\n")
        for seat in range(3):
            label = player_labels[seat] if seat < len(player_labels) else f"P{seat}"
            role = "地主" if seat == landlord else "农民"
            self.state_f.write(f"Player {seat} ({label}, {role}):\n")
            stats = play_type_stats.get(seat, {})
            if stats:
                for ptype, count in sorted(stats.items(), key=lambda x: -x[1]):
                    self.state_f.write(f"  {ptype}: {count}\n")
            else:
                self.state_f.write(f"  (无出牌)\n")
        self.state_f.write("\n")
        self.state_f.flush()


# =============================================================================
# Game Runner
# =============================================================================

class GameRunner:
    """Runs one or more games with configurable players on each seat."""

    def __init__(self, player_specs, verbose=False, timeout=5.0,
                 enable_log=True, force_bids=None):
        self.player_specs = player_specs
        self.verbose = verbose
        self.timeout = timeout
        self.enable_log = enable_log
        self.force_bids = force_bids or {}
        self.players = []
        self.logger = None
        self._bot_seats = set()
        self._player_labels = []

    # ------------------------------------------------------------------
    # Player lifecycle
    # ------------------------------------------------------------------

    def _init_players(self):
        """Create player objects (called once at start of run_game)."""
        self.players = []
        self._bot_seats = set()
        self._player_labels = []
        for seat, spec in enumerate(self.player_specs):
            p = _create_player(spec, seat, verbose=self.verbose)
            if seat in self.force_bids:
                p = ForcedBidWrapper(p, self.force_bids[seat])
            self.players.append(p)
            self._player_labels.append(p.label())
            if isinstance(p, BotPlayer):
                self._bot_seats.add(seat)
            elif isinstance(p, ForcedBidWrapper) and isinstance(p._inner, BotPlayer):
                self._bot_seats.add(seat)

    def _start_players(self):
        for p in self.players:
            p.start()

    def _stop_players(self):
        for p in self.players:
            p.stop()
        if self.logger:
            self.logger.close()
            self.logger = None

    # ------------------------------------------------------------------
    # Judge communication (in-process, ~0ms overhead)
    # ------------------------------------------------------------------

    def call_judge(self, full_input):
        """Call judge_botzone_ver.main() in-process, capture printed output."""
        if self.verbose:
            log_len = len(full_input.get("log", []))
            print(f"  [JUDGE]  calling with log length={log_len}")

        # Reset global error state
        _judge_errored[:] = [None, None, None]

        # Capture print() output from main()
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _judge_main(full_input)
        except SystemExit:
            pass  # exit() in printFinish / setError(all errored)
        except (IndexError, KeyError, TypeError):
            pass  # Expected: judge code accessing invalid log indices on edge cases
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue().strip()
        if not output:
            # Judge returned silently — typically after setError() when not all
            # three players have errored yet (e.g. landlord passed on first turn).
            # Build a finish command penalising errored players.
            return self._build_error_finish(full_input)

        # main() may print multiple JSON lines (printBidding + printRequest);
        # take the last one — it's the most complete response for the current state
        lines = output.splitlines()
        last = lines[-1].strip()
        if not last:
            return self._build_error_finish(full_input)

        try:
            return json.loads(last)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Judge returned invalid JSON: {last[:200]}")

    def _build_error_finish(self, full_input):
        """When judge produces no output, synthesize a finish command from errored state."""
        # Determine landlord from bidding log (first 6 log entries)
        game_log = full_input.get("log", [])
        landlord = -1
        bid_history = []
        for i in range(1, min(len(game_log), 6), 2):
            entry = game_log[i]
            for k, v in entry.items():
                if k.isdigit():
                    try:
                        bid_history.append(int(v.get("response", 0)))
                    except (ValueError, TypeError):
                        bid_history.append(0)
        if bid_history:
            max_bid = max(bid_history)
            if max_bid > 0:
                landlord = bid_history.index(max_bid)
            elif len(bid_history) >= 3:
                landlord = 0

        scores = [0, 0, 0]
        if landlord >= 0:
            farmer_1 = (landlord + 1) % 3
            farmer_2 = (landlord + 2) % 3
            if _judge_errored[landlord]:
                scores[landlord] = -2
                scores[farmer_1] = 1
                scores[farmer_2] = 1
            elif _judge_errored[farmer_1] or _judge_errored[farmer_2]:
                scores[landlord] = 2
                scores[farmer_1] = -1
                scores[farmer_2] = -1

        return {
            "command": "finish",
            "content": {"0": scores[0], "1": scores[1], "2": scores[2]},
            "display": {"landlord": landlord, "errored": list(_judge_errored)},
        }

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def run_game(self, seed=None):
        """Run one complete game. Returns dict with result info."""
        if seed is None:
            seed = random.randint(0, 2 ** 31 - 1)

        random.seed(seed)

        self._init_players()
        self._start_players()

        full_input = {"log": [], "initdata": {"seed": str(seed)}}
        game_log = []
        stats = {"turns": 0, "bot_turns": 0, "errors": []}
        play_type_stats = {0: {}, 1: {}, 2: {}}
        landlord = -1
        hand_state = {0: [], 1: [], 2: []}
        publiccards_added = False

        # Create logger with timestamp for this game
        if self.logger:
            self.logger.close()
            self.logger = None
        if self.enable_log:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logger = GameLogger(JUDGE_LOG_DIR, ts)

        try:
            while True:
                stats["turns"] += 1
                full_input["log"] = game_log

                judge_output = self.call_judge(full_input)

                if judge_output["command"] == "finish":
                    result = self._handle_finish(judge_output, stats,
                                                 play_type_stats)
                    result["play_type_stats"] = play_type_stats
                    return result

                content = judge_output["content"]
                display = judge_output.get("display", {})

                if "landlord" in display:
                    landlord = display["landlord"]

                if not content:
                    continue

                player_str = list(content.keys())[0]
                player = int(player_str)
                request = content[player_str]

                # ── Track hand state ──
                if "own" in request:
                    hand_state[player] = list(request["own"])
                if "publiccard" in request and landlord >= 0 \
                        and not publiccards_added and player == landlord:
                    hand_state[landlord] = (
                        hand_state[landlord] + list(request["publiccard"]))
                    publiccards_added = True

                # ── Get response from the player ──
                if player in self._bot_seats:
                    stats["bot_turns"] += 1
                try:
                    resp_value = self.players[player].respond(
                        request, hand_state[player])
                except Exception:
                    # Bot crash: fall back to pass
                    stats["errors"].append(
                        f"Player {player} ({self.players[player].label()}) "
                        f"error, passing")
                    resp_value = (0 if "bid" in request else [])

                # ── Logging (hand before action) ──
                if self.logger:
                    role = _player_role(player, landlord)
                    hand_str = format_hand(hand_state[player])
                    if isinstance(resp_value, list):
                        action_str = format_hand(resp_value)
                    else:
                        action_str = resp_value
                    self.logger.log_turn(
                        stats["turns"], player, role,
                        hand_str, action_str,
                        json.dumps(request), json.dumps(resp_value))

                # ── Update hand state: remove played cards ──
                if isinstance(resp_value, list) and resp_value:
                    # Count play type for statistics
                    play_type, _, _ = checkPokerType(list(resp_value))
                    if play_type and play_type != "空" and play_type != "错误":
                        play_type_stats[player][play_type] = \
                            play_type_stats[player].get(play_type, 0) + 1

                    current = hand_state[player]
                    for c in resp_value:
                        try:
                            current.remove(c)
                        except ValueError:
                            pass

                # Build log entries for judge
                display_entry = ({"output": {"display": display}}
                                 if display else {})
                resp_entry = self._build_log_entry(
                    player, resp_value, judge_output)
                game_log.append(display_entry)
                game_log.append(resp_entry)

        except Exception as e:
            stats["errors"].append(str(e))
            self._stop_players()
            return {"winner": None, "scores": [0, 0, 0], "error": str(e),
                    "stats": stats, "seed": seed, "landlord": -1,
                    "bot_seats": self._bot_seats,
                    "player_labels": self._player_labels,
                    "play_type_stats": play_type_stats}

    def _build_log_entry(self, player, resp_value, judge_output):
        entry = {str(player): {"verdict": "OK", "response": resp_value}}
        if judge_output.get("display"):
            entry["output"] = {"display": judge_output["display"]}
        return entry

    def _handle_finish(self, judge_output, stats, play_type_stats=None):
        # Log game summary before stopping players
        if self.logger and play_type_stats:
            self.logger.log_game_summary(play_type_stats, self._player_labels,
                                         judge_output.get("display", {}).get("landlord", -1))

        self._stop_players()

        content = judge_output.get("content", {})
        display = judge_output.get("display", {})

        scores = [int(content.get("0", 0)), int(content.get("1", 0)),
                  int(content.get("2", 0))]

        winner = None
        for i, s in enumerate(scores):
            if s > 0:
                winner = i
                break

        landlord = display.get("landlord", -1)

        return {
            "winner": winner,
            "scores": scores,
            "landlord": landlord,
            "stats": stats,
            "seed": None,
            "player_labels": self._player_labels,
            "bot_seats": self._bot_seats,
        }


# =============================================================================
# Batch runner
# =============================================================================

def run_batch(player_specs, num_games, verbose=False, seed=None,
              enable_log=False):
    """Run multiple games and print summary statistics."""
    if seed is not None:
        random.seed(seed)

    runner = GameRunner(player_specs, verbose=verbose,
                        enable_log=enable_log)

    results = []
    wins = 0
    bot_wins = {}
    bot_games_as_landlord = {}
    bot_wins_as_landlord = {}

    bot_seats = set()
    for seat, spec in enumerate(player_specs):
        if spec not in ("random", "greedy", "pass"):
            bot_seats.add(seat)

    for seat in bot_seats:
        bot_wins[seat] = 0
        bot_games_as_landlord[seat] = 0
        bot_wins_as_landlord[seat] = 0

    labels = [_player_label(s) for s in player_specs]

    # Batch cumulative statistics
    batch_play_stats = {0: {}, 1: {}, 2: {}}
    batch_landlord_count = {0: 0, 1: 0, 2: 0}
    batch_win_count = {0: 0, 1: 0, 2: 0}

    print(f"Running {num_games} games...")
    for seat, label in enumerate(labels):
        print(f"  P{seat}: {label}")
    print(f"{'='*60}")

    start_time = time.time()

    for i in range(num_games):
        game_seed = random.randint(0, 2 ** 31 - 1)
        result = runner.run_game(seed=game_seed)
        result["seed"] = game_seed
        results.append(result)

        scores = result.get("scores", [0, 0, 0])
        if result.get("error"):
            status = "ERR"
        else:
            status = "WIN" if any(scores[s] > 0 for s in bot_seats) else "LOSE"

        for seat in bot_seats:
            if scores[seat] > 0:
                bot_wins[seat] += 1
            if result["landlord"] == seat:
                bot_games_as_landlord[seat] += 1
                if scores[seat] > 0:
                    bot_wins_as_landlord[seat] += 1

        # Accumulate batch statistics
        game_play_stats = result.get("play_type_stats", {})
        for seat in range(3):
            for ptype, count in game_play_stats.get(seat, {}).items():
                batch_play_stats[seat][ptype] = \
                    batch_play_stats[seat].get(ptype, 0) + count
        landlord = result.get("landlord", -1)
        if landlord >= 0:
            batch_landlord_count[landlord] += 1
        for seat in range(3):
            if scores[seat] > 0:
                batch_win_count[seat] += 1

        if verbose or (i + 1) % 10 == 0:
            err = f" ERROR: {result['error']}" if result.get("error") else ""
            ll = result.get("landlord", -1)
            print(f"  Game {i+1:4d}: {status:4s}  "
                  f"landlord=P{ll}  scores={result['scores']}{err}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Results ({num_games} games, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Players: P0={labels[0]}  P1={labels[1]}  P2={labels[2]}")

    for seat in bot_seats:
        w = bot_wins[seat]
        print(f"  P{seat} ({labels[seat]}):")
        print(f"    Win rate:    {w}/{num_games} ({100*w/num_games:.1f}%)")
        lg = bot_games_as_landlord[seat]
        wl = bot_wins_as_landlord[seat] if lg > 0 else 0
        if lg > 0:
            print(f"    As landlord: {wl}/{lg} ({100*wl/lg:.1f}%)")
        fg = num_games - lg
        fw = w - wl
        if fg > 0:
            print(f"    As farmer:   {fw}/{fg} ({100*fw/fg:.1f}%)")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"  Errors:      {len(errors)}")
        for e in errors[:3]:
            print(f"    seed={e['seed']}: {e['error'][:100]}")
    print(f"  Games/sec:   {num_games/elapsed:.2f}")

    # Write batch statistics log
    _write_batch_log(batch_play_stats, batch_landlord_count,
                     batch_win_count, num_games, labels, elapsed)

    return results


def _write_batch_log(batch_play_stats, batch_landlord_count,
                     batch_win_count, num_games, labels, elapsed):
    """Write cumulative batch statistics to a log file."""
    os.makedirs(JUDGE_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(JUDGE_LOG_DIR, f"batch_stats_{ts}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"批量对局统计 ({num_games} 局, {elapsed:.1f}s)\n")
        f.write(f"{'='*60}\n")
        f.write(f"玩家: P0={labels[0]}  P1={labels[1]}  P2={labels[2]}\n\n")

        all_types = set()
        for seat in range(3):
            all_types.update(batch_play_stats[seat].keys())

        # Ordered by frequency across all seats
        type_total = {}
        for pt in all_types:
            type_total[pt] = sum(batch_play_stats[s].get(pt, 0) for s in range(3))
        ordered_types = sorted(type_total, key=lambda t: -type_total[t])

        for seat in range(3):
            label = labels[seat]
            f.write(f"--- P{seat} ({label}) ---\n")

            # Landlord / win / loss stats
            lc = batch_landlord_count.get(seat, 0)
            wc = batch_win_count.get(seat, 0)
            lc_rate = f"{100*lc/num_games:.1f}%" if num_games > 0 else "N/A"
            wc_rate = f"{100*wc/num_games:.1f}%" if num_games > 0 else "N/A"
            f.write(f"  当庄次数: {lc}/{num_games} ({lc_rate})\n")
            f.write(f"  获胜次数: {wc}/{num_games} ({wc_rate})\n")

            # Play type stats
            f.write(f"  出牌种类统计:\n")
            stats = batch_play_stats.get(seat, {})
            if stats:
                for ptype in ordered_types:
                    cnt = stats.get(ptype, 0)
                    if cnt > 0:
                        f.write(f"    {ptype}: {cnt}\n")
            else:
                f.write(f"    (无出牌记录)\n")
            f.write("\n")

    print(f"  Batch log:   {log_path}")


# =============================================================================
# VS batch verification — head-to-head with role swapping
# =============================================================================

def run_vs_batch(bot_a, bot_b, num_matches, verbose=False, master_seed=None,
                 enable_log=True, unpaired=False):
    """Head-to-head batch verification between two bots with role swapping.

    Each match = 2 games (landlord pre-assigned, no bidding phase):
      Game 1: BotA = landlord (P0, forced bid 3), BotB = farmers (P1,P2, forced bid 0)
      Game 2: BotB = landlord (P0, forced bid 3), BotA = farmers (P1,P2, forced bid 0)

    ⚠ 配对(duplicate)语义: G1 与 G2 默认使用**同一副牌**(同一个种子), 只交换角色。
      unpaired=True 时退回旧行为(G1/G2 各用不同种子) —— 那种模式下两个 bot 永远不在
      同一副牌上比较, 牌组难度差会被误读成引擎强弱, 不建议使用。

    Bidding is bypassed via ForcedBidWrapper — the single-seat bot is
    forced to bid 3 while the two-seat bot is forced to bid 0 on both
    seats. This guarantees a clean landlord/farmer split for stats.

    Game seeds are deterministically generated from master_seed so that
    running n matches and n+1 matches with the same seed produces
    identical first 2n games.
    """
    if master_seed is None:
        master_seed = random.randint(0, 2 ** 31 - 1)

    label_a = _player_label(bot_a)
    label_b = _player_label(bot_b)

    # Deterministic seed generation: use a dedicated RNG so the global
    # random state is not affected.
    seed_rng = random.Random(master_seed)
    # 配对: 每个 match 一个种子(同一副牌跑两局, 只交换角色)
    # 旧行为(unpaired): 2*num_matches 个不同种子 → 两 bot 在不同牌组上比较
    game_seeds = [seed_rng.randint(0, 2 ** 31 - 1)
                  for _ in range(num_matches * 2 if unpaired else num_matches)]

    # Stats: per-bot, per-role
    stats = {
        'bot_a': {'label': label_a, 'path': bot_a,
                   'as_landlord': {'win': 0, 'loss': 0},
                   'as_farmer':  {'win': 0, 'loss': 0}},
        'bot_b': {'label': label_b, 'path': bot_b,
                   'as_landlord': {'win': 0, 'loss': 0},
                   'as_farmer':  {'win': 0, 'loss': 0}},
        # 配对统计（同一副牌上逐对比）
        'paired': {
            'a_both': 0,     # A 两局全胜（A 当地主赢 & B 当地主输）
            'b_both': 0,     # B 两局全胜
            'same': 0,       # 同向: 两局都是地主赢 / 都是农民赢（该副牌与引擎强弱无关）
            'error': 0,      # 该配对中有对局异常
            'net_ll': 0,     # 地主净分差累加: A 当地主得分 − B 当地主得分
            'net_list': [],  # 逐配对的净分差（供符号检验/均值）
        },
    }
    stats['bot_a']['points'] = 0
    stats['bot_b']['points'] = 0
    all_results = []

    # Specs for the two seat configurations
    spec_A = [bot_a, bot_b, bot_b]  # BotA at P0, BotB at P1,P2
    spec_B = [bot_b, bot_a, bot_a]  # BotB at P0, BotA at P1,P2

    print(f"VS Mode: {label_a}  vs  {label_b}")
    print(f"Matches: {num_matches}  ({num_matches * 2} games with swapped roles)")
    print(f"Master seed: {master_seed}")
    print(f"{'='*60}")

    start_time = time.time()

    for match_idx in range(num_matches):
        match_results = {}

        # ── Game 1: BotA = landlord (P0 forced bid 3), BotB = farmers ──
        seed_1 = game_seeds[match_idx * 2] if unpaired else game_seeds[match_idx]
        runner_1 = GameRunner(spec_A, verbose=verbose, enable_log=enable_log,
                              force_bids={0: 3, 1: 0, 2: 0})
        result_1 = runner_1.run_game(seed=seed_1)
        result_1['seed'] = seed_1
        result_1['match_idx'] = match_idx
        result_1['game_in_match'] = 1
        _record_vs_game(result_1, stats, spec='A')
        match_results['game1'] = result_1

        # ── Game 2: BotB = landlord (P0 forced bid 3), BotA = farmers ──
        seed_2 = (game_seeds[match_idx * 2 + 1] if unpaired
                  else game_seeds[match_idx])
        runner_2 = GameRunner(spec_B, verbose=verbose, enable_log=enable_log,
                              force_bids={0: 3, 1: 0, 2: 0})
        result_2 = runner_2.run_game(seed=seed_2)
        result_2['seed'] = seed_2
        result_2['match_idx'] = match_idx
        result_2['game_in_match'] = 2
        _record_vs_game(result_2, stats, spec='B')
        match_results['game2'] = result_2

        if not unpaired:
            _record_vs_pair(result_1, result_2, stats)

        all_results.append(match_results)

        if verbose or (match_idx + 1) % 10 == 0:
            _print_vs_progress(match_idx + 1, num_matches, result_1, result_2)

    elapsed = time.time() - start_time
    _print_vs_summary(stats, num_matches, elapsed, unpaired=unpaired)
    _write_vs_log(stats, num_matches, elapsed, master_seed, all_results,
                  label_a, label_b)

    return all_results, stats


def _record_vs_game(result, stats, spec):
    """Record one game's outcome into VS stats.

    spec='A': P0=BotA, P1=BotB, P2=BotB
    spec='B': P0=BotB, P1=BotA, P2=BotA
    """
    landlord = result.get('landlord', -1)
    scores = result.get('scores', [0, 0, 0])

    if spec == 'A':
        # P0=BotA (single seat), P1/P2=BotB (two seats)
        landlord_bot = 'bot_a' if landlord == 0 else 'bot_b'
    else:
        # P0=BotB (single seat), P1/P2=BotA (two seats)
        landlord_bot = 'bot_b' if landlord == 0 else 'bot_a'

    if landlord < 0:
        return  # Game errored before landlord was determined; skip

    landlord_wins = scores[landlord] > 0
    if landlord_wins:
        stats[landlord_bot]['as_landlord']['win'] += 1
        farmer_bot = 'bot_b' if landlord_bot == 'bot_a' else 'bot_a'
        stats[farmer_bot]['as_farmer']['loss'] += 1
    else:
        stats[landlord_bot]['as_landlord']['loss'] += 1
        farmer_bot = 'bot_b' if landlord_bot == 'bot_a' else 'bot_a'
        stats[farmer_bot]['as_farmer']['win'] += 1


def _record_vs_pair(r1, r2, stats):
    """同一副牌的两局（A 地主 / B 地主）配对记账。

    G1: A=地主(P0) vs B 双农民   → landlord_wins_1 = r1.scores[0] > 0
    G2: B=地主(P0) vs A 双农民   → landlord_wins_2 = r2.scores[0] > 0
      landlord_wins_1 与 2 都为真: 地主赢两局（同向, 牌力/角色决定）
      landlord_wins_1 为真, 2 为假: A 当地主赢 + A 当农民赢 → A 全胜
      landlord_wins_1 为假, 2 为真: B 全胜
      都为假: 农民赢两局（同向）
    """
    p = stats['paired']
    if r1.get('error') or r2.get('error'):
        p['error'] += 1
        return
    lw1 = r1.get('scores', [0, 0, 0])[0] > 0
    lw2 = r2.get('scores', [0, 0, 0])[0] > 0
    if lw1 and lw2:
        p['same'] += 1
    elif (not lw1) and (not lw2):
        p['same'] += 1
    elif lw1 and not lw2:
        p['a_both'] += 1
    else:
        p['b_both'] += 1
    net = r1.get('scores', [0, 0, 0])[0] - r2.get('scores', [0, 0, 0])[0]
    p['net_ll'] += net
    p['net_list'].append(net)


def _sign_test_p(k, n):
    """二项符号检验（双侧, p=0.5）: n 次决定性配对里 k 次偏向 A。"""
    if n <= 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _print_paired_summary(stats, num_matches):
    """打印配对结论（同一副牌上逐对比）。"""
    p = stats.get('paired')
    if not p or p.get('error', 0) == num_matches:
        return
    dec = p['a_both'] + p['b_both']
    print(f"\n  配对结果（同一副牌交换角色, {num_matches} 对）:")
    print(f"    A 两局全胜: {p['a_both']:3d}    "
          f"B 两局全胜: {p['b_both']:3d}    "
          f"同向(牌力/角色决定): {p['same']:3d}"
          + (f"    异常: {p['error']}" if p['error'] else ""))
    if dec > 0:
        pv = _sign_test_p(p['a_both'], dec)
        print(f"    决定性配对里 A 胜率: {p['a_both']}/{dec} "
              f"= {100*p['a_both']/dec:.1f}%   符号检验 p = {pv:.3f}")
    if p['net_list']:
        mean = p['net_ll'] / len(p['net_list'])
        print(f"    地主净分差(A−B): 合计 {p['net_ll']:+d}   均值 {mean:+.2f}/对")


def _print_vs_progress(match_num, total, r1, r2):
    """Print progress for a single match."""
    def _game_str(r):
        ll = r.get('landlord', -1)
        sc = r.get('scores', [0, 0, 0])
        err = r.get('error')
        if err:
            return f"ERR({str(err)[:40]})"
        return f"ll=P{ll} scores={sc}"
    print(f"  Match {match_num:4d}/{total}: "
          f"G1 [{_game_str(r1)}]  G2 [{_game_str(r2)}]")


def _print_vs_summary(stats, num_matches, elapsed, unpaired=False):
    """Print final VS summary to stdout."""
    total_games = num_matches * 2

    print(f"\n{'='*60}")
    print(f"VS Results ({num_matches} matches, {total_games} games, "
          f"{elapsed:.1f}s)")
    print(f"{'='*60}")

    for bot_key in ['bot_a', 'bot_b']:
        b = stats[bot_key]
        lw = b['as_landlord']['win']
        ll = b['as_landlord']['loss']
        lt = lw + ll
        fw = b['as_farmer']['win']
        fl = b['as_farmer']['loss']
        ft = fw + fl
        total_w = lw + fw
        total_g = lt + ft

        print(f"  {b['label']} ({b['path']}):")
        if lt > 0:
            print(f"    As landlord:  {lw}W / {ll}L  ({100*lw/lt:.1f}%)")
        else:
            print(f"    As landlord:  (never became landlord)")
        if ft > 0:
            print(f"    As farmer:    {fw}W / {fl}L  ({100*fw/ft:.1f}%)")
        else:
            print(f"    As farmer:    (never a farmer)")
        if total_g > 0:
            print(f"    Overall:      {total_w}W / {total_g - total_w}L  "
                  f"({100*total_w/total_g:.1f}%)")

    if unpaired:
        print("\n  --vs-unpaired: 未做同牌配对，配对统计与符号检验不适用。")
    else:
        print("\n  注意: 配对模式下 Overall 天然接近 50%（每对牌两 bot 各当一次地主），"
              "\n        判断强弱请看下面的配对结论，而不是 Overall。")
        _print_paired_summary(stats, num_matches)


def _write_vs_log(stats, num_matches, elapsed, master_seed, all_results,
                  label_a, label_b):
    """Write VS batch results to a persistent log file."""
    os.makedirs(JUDGE_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(JUDGE_LOG_DIR, f"vs_batch_{ts}.log")
    total_games = num_matches * 2

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"VS Batch Verification Log\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp:    {ts}\n")
        f.write(f"Bot A:        {label_a}  ({stats['bot_a']['path']})\n")
        f.write(f"Bot B:        {label_b}  ({stats['bot_b']['path']})\n")
        f.write(f"Matches:      {num_matches}  ({total_games} games)\n")
        f.write(f"Master seed:  {master_seed}\n")
        f.write(f"Duration:     {elapsed:.1f}s\n")
        f.write(f"\n")

        f.write(f"--- Summary ---\n")
        for bot_key in ['bot_a', 'bot_b']:
            b = stats[bot_key]
            lw, ll = b['as_landlord']['win'], b['as_landlord']['loss']
            lt = lw + ll
            fw, fl = b['as_farmer']['win'], b['as_farmer']['loss']
            ft = fw + fl

            f.write(f"\n{b['label']}:\n")
            f.write(f"  As landlord: {lw}W / {ll}L"
                    f"{' ({:.1f}%)'.format(100*lw/lt) if lt > 0 else ''}\n")
            f.write(f"  As farmer:   {fw}W / {fl}L"
                    f"{' ({:.1f}%)'.format(100*fw/ft) if ft > 0 else ''}\n")
            tw = lw + fw
            tg = lt + ft
            f.write(f"  Overall:     {tw}W / {tg - tw}L"
                    f"{' ({:.1f}%)'.format(100*tw/tg) if tg > 0 else ''}\n")

        p = stats.get('paired') or {}
        dec = p.get('a_both', 0) + p.get('b_both', 0)
        if dec or p.get('same'):
            f.write(f"\n--- Paired (same deal, roles swapped) ---\n")
            f.write(f"  A won both: {p.get('a_both', 0)}   "
                    f"B won both: {p.get('b_both', 0)}   "
                    f"same-direction: {p.get('same', 0)}")
            if p.get('error'):
                f.write(f"   errors: {p['error']}")
            f.write("\n")
            if dec:
                f.write(f"  Decisive pairs: A {p.get('a_both', 0)}/{dec} "
                        f"= {100*p['a_both']/dec:.1f}%   "
                        f"sign-test p = {_sign_test_p(p.get('a_both', 0), dec):.3f}\n")
            if p.get('net_list'):
                f.write(f"  Landlord net score (A-B): {p.get('net_ll', 0):+d} "
                        f"(mean {p['net_ll']/len(p['net_list']):+.2f}/pair)\n")

        # Per-match details
        f.write(f"\n--- Per-Match Details ---\n")
        for match_idx, match in enumerate(all_results):
            for game_tag in ['game1', 'game2']:
                r = match[game_tag]
                seed = r.get('seed', '?')
                ll = r.get('landlord', -1)
                sc = r.get('scores', [0, 0, 0])
                err = r.get('error')
                spec_side = 'A' if game_tag == 'game1' else 'B'
                f.write(f"  M{match_idx+1:04d} G{game_tag[-1]} "
                        f"seed={seed} spec={spec_side} "
                        f"landlord=P{ll} scores={sc}")
                if err:
                    f.write(f" ERROR: {err}")
                f.write("\n")

    print(f"  VS log:      {log_path}")


# =============================================================================
# CLI helpers
# =============================================================================

def _player_label(spec):
    """Human-readable label for a player spec."""
    spec = spec.strip()
    if spec in ("random", "greedy", "pass"):
        return spec
    return os.path.basename(spec)


# =============================================================================
# CLI
# =============================================================================

def _force_utf8_stdio():
    """Force UTF-8 on stdout/stderr so redirected output never crashes.

    On Windows, a pipe or file redirect makes Python use the ANSI code page
    (e.g. cp936/cp1252) instead of the console's UTF-8 writer.  The Chinese
    summary lines contain characters that code page cannot encode, which would
    raise UnicodeEncodeError halfway through a run.  Reconfiguring is a no-op
    on Python builds that already write UTF-8 to the console.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main():
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="DouEnMH Local Game Runner - play games against bots/agents "
                    "with the bundled Botzone judge")

    # Per-seat player specs
    parser.add_argument("--p0", default=None, metavar="SPEC",
                        help="Player 0: bot.exe path | random | greedy | pass")
    parser.add_argument("--p1", default=None, metavar="SPEC",
                        help="Player 1: bot.exe path | random | greedy | pass")
    parser.add_argument("--p2", default=None, metavar="SPEC",
                        help="Player 2: bot.exe path | random | greedy | pass")

    # Backward-compat: single bot + opponent type
    parser.add_argument("--bot", default=None, metavar="PATH",
                        help="Path to bot.exe (shorthand for one bot seat); "
                             "relative paths are tried against the current "
                             "directory, then tools/runner/. Default when no "
                             "seat is given: cpp/build/bot.exe")
    parser.add_argument("--opponent", choices=["random", "greedy", "pass"],
                        default="random",
                        help="Agent type for non-bot seats with --bot "
                             "(default: random)")

    parser.add_argument("--games", type=int, default=1,
                        help="Number of games to run (default: 1). "
                             "1 = single-game mode, >1 = batch mode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (single game: this game's seed; "
                             "batch: seed of the per-game seed sequence)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show bot I/O and judge calls)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-call timeout in seconds (default: 30; "
                             "parsed for compatibility, not enforced)")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable per-turn game logging (per-turn logs are "
                             "on by default in single-game and VS modes; "
                             "batch mode is always off)")

    # VS batch verification mode
    parser.add_argument("--vs", dest="vs_bot", default=None, metavar="PATH",
                        help="Path to the second bot (bot B); entering VS "
                             "head-to-head verification mode")
    parser.add_argument("--vs-games", type=int, default=100,
                        help="Number of matches in VS mode (default: 100; "
                             "each match = 2 games on the same deal)")
    parser.add_argument("--vs-seed", type=int, default=None,
                        help="Master seed for the deterministic VS game sequence")
    parser.add_argument("--vs-unpaired", action="store_true",
                        help="Legacy: G1/G2 use different seeds (not "
                             "recommended; deal variance would be read as "
                             "engine strength difference)")

    args = parser.parse_args()

    # ── VS batch verification mode ──
    if args.vs_bot is not None:
        if args.vs_games < 1:
            print("ERROR: --vs-games must be >= 1")
            sys.exit(1)

        # Resolve bot A path
        if args.bot is not None:
            bot_a = args.bot
        elif args.p0 is not None:
            bot_a = args.p0
        else:
            print("ERROR: --vs requires --bot (or --p0) to specify the first bot")
            sys.exit(1)

        # Resolve bot B path
        bot_b = args.vs_bot

        # Make paths absolute (relative to tools/runner/)
        if not os.path.isabs(bot_a):
            bot_a = os.path.join(SCRIPT_DIR, bot_a)
        if not os.path.isabs(bot_b):
            bot_b = os.path.join(SCRIPT_DIR, bot_b)
        bot_a = os.path.normpath(bot_a)
        bot_b = os.path.normpath(bot_b)

        # Validate both bots exist
        for path, name in [(bot_a, "Bot A"), (bot_b, "Bot B")]:
            if not os.path.isfile(path):
                print(f"ERROR: {name} executable not found: {path}")
                sys.exit(1)

        run_vs_batch(bot_a, bot_b, num_matches=args.vs_games,
                     verbose=args.verbose, master_seed=args.vs_seed,
                     enable_log=not args.no_log,
                     unpaired=args.vs_unpaired)
        return

    # ── Normal / backward-compat mode ──
    if args.games < 1:
        print("ERROR: --games must be >= 1")
        sys.exit(1)

    # Resolve player specs
    if args.p0 is not None or args.p1 is not None or args.p2 is not None:
        # Explicit per-seat config
        specs = [
            args.p0 if args.p0 is not None else "random",
            args.p1 if args.p1 is not None else "random",
            args.p2 if args.p2 is not None else "random",
        ]
    elif args.bot is not None:
        # Backward-compat: --bot <path> --opponent <type>
        specs = [args.bot, args.opponent, args.opponent]
    else:
        # Default: the repo's own bot build (../../cpp/build/bot.exe,
        # resolved against tools/runner/) + two random agents.
        specs = [DEFAULT_BOT_SPEC, "random", "random"]

    # Fail fast with a clear message instead of a Popen traceback.
    for seat, spec in enumerate(specs):
        if spec.strip() in ("random", "greedy", "pass"):
            continue
        exe = _resolve_bot_path(spec)
        if not os.path.isfile(exe):
            print(f"ERROR: P{seat} bot executable not found: "
                  f"{os.path.normpath(exe)}")
            print("       Build the bot first (see docs/cpp.md) or use "
                  "random/greedy/pass for that seat.")
            sys.exit(1)

    # Print player config
    labels = [_player_label(s) for s in specs]
    print(f"Players: P0={labels[0]}  P1={labels[1]}  P2={labels[2]}")
    print(f"Judge:   {JUDGE_DIR}/judge_botzone_ver.py (in-process)")
    print(f"Timeout: {args.timeout}s (not enforced)")
    print()

    if args.games == 1:
        enable_log = not args.no_log
        runner = GameRunner(specs, verbose=args.verbose,
                            timeout=args.timeout, enable_log=enable_log)
        result = runner.run_game(seed=args.seed)

        print(f"\n{'='*60}")
        print("Game Result")
        print(f"{'='*60}")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        else:
            status = "WIN" if result["winner"] in result.get("bot_seats", set()) else "LOSE"
            landlord = result.get("landlord", -1)
            print(f"  Result:    {status}")
            print(f"  Landlord:  P{landlord}")
            print(f"  Scores:    P0={result['scores'][0]:+d}  "
                  f"P1={result['scores'][1]:+d}  P2={result['scores'][2]:+d}")
            print(f"  Turns:     {result['stats']['turns']}")
            print(f"  Bot turns: {result['stats']['bot_turns']}")
    else:
        run_batch(specs, args.games, verbose=args.verbose,
                  seed=args.seed)


if __name__ == "__main__":
    main()
