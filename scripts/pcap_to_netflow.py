"""Convert a PCAP/PCAPNG file to NetFlow CSV compatible with the NIDS pipeline.

Uses NFStream to extract bidirectional flow features and maps them to the
NF-UNSW-NB15-v2 column schema expected by ``src/data/loader.py``.

Install:  pip install nfstream   (or:  uv pip install nfstream)

Usage:
    uv run python scripts/pcap_to_netflow.py capture.pcap
    uv run python scripts/pcap_to_netflow.py capture.pcap -o flows.csv
    uv run python scripts/pcap_to_netflow.py capture.pcap --label Benign
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

NFSTREAM_TO_NF = {
    "src_port": "L4_SRC_PORT",
    "dst_port": "L4_DST_PORT",
    "protocol": "PROTOCOL",
    "application_name": "L7_PROTO",
    "src2dst_bytes": "IN_BYTES",
    "dst2src_bytes": "OUT_BYTES",
    "src2dst_packets": "IN_PKTS",
    "dst2src_packets": "OUT_PKTS",
    "src2dst_duration_ms": "DURATION_IN",
    "dst2src_duration_ms": "DURATION_OUT",
    "src2dst_max_piat_ms": "LONGEST_FLOW_PKT",
    "src2dst_min_piat_ms": "SHORTEST_FLOW_PKT",
    "bidirectional_duration_ms": "FLOW_DURATION_MILLISECONDS",
    "bidirectional_min_ps": "MIN_IP_PKT_LEN",
    "bidirectional_max_ps": "MAX_IP_PKT_LEN",
    "src2dst_syn_packets": "_syn_src",
    "dst2src_syn_packets": "_syn_dst",
    "src2dst_fin_packets": "_fin_src",
    "dst2src_fin_packets": "_fin_dst",
    "src2dst_rst_packets": "_rst_src",
    "dst2src_rst_packets": "_rst_dst",
    "src2dst_psh_packets": "_psh_src",
    "dst2src_psh_packets": "_psh_dst",
    "src2dst_ack_packets": "_ack_src",
    "dst2src_ack_packets": "_ack_dst",
}

TCP_SYN = 0x02
TCP_FIN = 0x01
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10


def _build_tcp_flags(row: pd.Series, prefix: str) -> int:
    flags = 0
    if row.get(f"_syn_{prefix}", 0) > 0:
        flags |= TCP_SYN
    if row.get(f"_fin_{prefix}", 0) > 0:
        flags |= TCP_FIN
    if row.get(f"_rst_{prefix}", 0) > 0:
        flags |= TCP_RST
    if row.get(f"_psh_{prefix}", 0) > 0:
        flags |= TCP_PSH
    if row.get(f"_ack_{prefix}", 0) > 0:
        flags |= TCP_ACK
    return flags


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    dur_s = df["FLOW_DURATION_MILLISECONDS"].clip(lower=1) / 1000.0
    df["SRC_TO_DST_SECOND_BYTES"] = df["IN_BYTES"] / dur_s
    df["DST_TO_SRC_SECOND_BYTES"] = df["OUT_BYTES"] / dur_s
    df["SRC_TO_DST_AVG_THROUGHPUT"] = df["IN_BYTES"] * 8.0 / dur_s
    df["DST_TO_SRC_AVG_THROUGHPUT"] = df["OUT_BYTES"] * 8.0 / dur_s
    return df


def pcap_to_dataframe(
    pcap_path: str | Path,
    label: str = "Benign",
) -> pd.DataFrame:
    """Extract NetFlow-style features from a PCAP file.

    Parameters
    ----------
    pcap_path:
        Path to the .pcap or .pcapng file.
    label:
        Label string to assign to all flows (default: "Benign").

    Returns
    -------
    pd.DataFrame
        DataFrame with NF-UNSW-NB15-v2 compatible columns.
    """
    try:
        from nfstream import NFStreamer
    except ImportError:
        logger.error(
            "nfstream is required for PCAP conversion. "
            "Install it with: pip install nfstream"
        )
        sys.exit(1)

    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    logger.info("Reading %s ...", pcap_path)
    streamer = NFStreamer(source=str(pcap_path), statistical_analysis=True)
    raw_df = streamer.to_pandas()
    logger.info("Extracted %d flows from PCAP", len(raw_df))

    if raw_df.empty:
        logger.warning("No flows extracted — file may be empty or unreadable")
        return pd.DataFrame()

    renamed = raw_df.rename(columns={
        k: v for k, v in NFSTREAM_TO_NF.items() if k in raw_df.columns
    })

    df = pd.DataFrame()

    for col in [
        "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "OUT_BYTES",
        "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
        "DURATION_IN", "DURATION_OUT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    ]:
        df[col] = renamed[col] if col in renamed.columns else 0

    if "L7_PROTO" in renamed.columns:
        df["L7_PROTO"] = pd.Categorical(renamed["L7_PROTO"]).codes.astype(float)
    else:
        df["L7_PROTO"] = 0.0

    df["CLIENT_TCP_FLAGS"] = renamed.apply(
        lambda r: _build_tcp_flags(r, "src"), axis=1
    )
    df["SERVER_TCP_FLAGS"] = renamed.apply(
        lambda r: _build_tcp_flags(r, "dst"), axis=1
    )
    df["TCP_FLAGS"] = df["CLIENT_TCP_FLAGS"] | df["SERVER_TCP_FLAGS"]

    df["MIN_TTL"] = renamed.get("bidirectional_min_raw", pd.Series(64, index=df.index))
    df["MAX_TTL"] = renamed.get("bidirectional_max_raw", pd.Series(128, index=df.index))

    for col in [
        "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS",
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN",
    ]:
        df[col] = 0.0

    df = _compute_derived(df)

    if "bidirectional_first_seen_ms" in raw_df.columns:
        df["Timestamp"] = pd.to_datetime(
            raw_df["bidirectional_first_seen_ms"], unit="ms"
        )
    else:
        df["Timestamp"] = pd.Timestamp.now()

    if "src_ip" in raw_df.columns:
        df["IPV4_SRC_ADDR"] = raw_df["src_ip"]
        df["IPV4_DST_ADDR"] = raw_df["dst_ip"]

    df["Label"] = label

    df = df.astype({c: np.float64 for c in df.select_dtypes(include=[np.integer]).columns})

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PCAP to NF-UNSW-NB15-v2 CSV format"
    )
    parser.add_argument("pcap", type=Path, help="Path to .pcap/.pcapng file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV path (default: <pcap_stem>_flows.csv)"
    )
    parser.add_argument(
        "--label", default="Benign",
        help="Label to assign to all flows (default: Benign)"
    )
    args = parser.parse_args()

    out_path = args.output or args.pcap.with_name(f"{args.pcap.stem}_flows.csv")

    df = pcap_to_dataframe(args.pcap, label=args.label)
    if df.empty:
        logger.error("No flows to write")
        sys.exit(1)

    df.to_csv(out_path, index=False)
    logger.info("Wrote %d flows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
