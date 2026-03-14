"""
analysis/parser.py
Reads raw ABIDES .bz2 pickle logs and extracts Avellaneda-Stoikov quote data.
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
#  Regex for the AS_QUOTE event string produced by AvellanedaStoikovAgent
#  Format: "AVELLANEDA_STOIKOV_AGENT | inv={inv} mid={mid} bid={bid} ask={ask}"
# ---------------------------------------------------------------------------
_AS_PATTERN = re.compile(
    r"inv=(?P<inv>-?\d+)\s+"
    r"mid=(?P<mid>[\d.]+)\s+"
    r"bid=(?P<bid>[\d.]+)\s+"
    r"ask=(?P<ask>[\d.]+)"
)


def load_agent_log(log_dir: str,
                   agent_name: str = "AVELLANEDA_STOIKOV_AGENT") -> pd.DataFrame:
    """Load the raw ABIDES agent log from its .bz2 pickle file.

    Parameters
    ----------
    log_dir : str
        Path to the simulation log directory (e.g. ``log/TCC_AS_Experiment``).
    agent_name : str
        Agent filename stem (without the .bz2 extension).

    Returns
    -------
    pd.DataFrame
        Rows that contain AS quote data (``inv=`` **and** ``mid=``).
    """
    filepath = os.path.join(log_dir, f"{agent_name}.bz2")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Agent log not found: {filepath}\n"
            "Make sure the simulation was run with log_orders enabled "
            "and the agent name matches."
        )

    df = pd.read_pickle(filepath)

    # Identify the column that holds the event string.
    # In ABIDES the log DataFrame normally has EventType and Event columns,
    # with EventTime as the index.  Our custom rows use EventType == "AS_QUOTE"
    # and the formatted string lives in the Event column.
    text_col = "Event"
    if text_col not in df.columns:
        # Fallback: pick the first object-dtype column
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols) == 0:
            raise ValueError("No string column found in the agent log DataFrame.")
        text_col = obj_cols[0]

    # Keep only rows where the text column contains the key markers.
    mask = df[text_col].astype(str).str.contains("inv=") & \
           df[text_col].astype(str).str.contains("mid=")
    return df.loc[mask].copy()


def parse_as_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric inv / mid / bid / ask from filtered log rows.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_agent_log` (already filtered to AS_QUOTE rows).

    Returns
    -------
    pd.DataFrame
        Indexed by ``EventTime`` (datetime) with float columns
        ``inv``, ``mid``, ``bid``, ``ask``.
    """
    text_col = "Event" if "Event" in df.columns else df.select_dtypes(include="object").columns[0]

    records = []
    for idx, row in df.iterrows():
        m = _AS_PATTERN.search(str(row[text_col]))
        if m is None:
            continue
        records.append({
            "EventTime": idx if isinstance(idx, pd.Timestamp) else row.get("EventTime", idx),
            "inv": float(m.group("inv")),
            "mid": float(m.group("mid")),
            "bid": float(m.group("bid")),
            "ask": float(m.group("ask")),
        })

    if not records:
        raise ValueError("No AS quote rows matched the expected pattern.")

    result = pd.DataFrame(records)
    result["EventTime"] = pd.to_datetime(result["EventTime"])
    result.set_index("EventTime", inplace=True)
    return result
