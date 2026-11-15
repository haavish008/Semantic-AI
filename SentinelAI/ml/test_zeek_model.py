from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = ROOT / "data" / "zeek" / "zeek_features.json"
MODEL_PATH = ROOT / "models" / "zeek_ddos_model.pkl"


# Load model
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
features = bundle["features"]


# Load Zeek features
with open(FEATURE_PATH) as f:
    flows = json.load(f)


print(f"Zeek flows loaded: {len(flows)}")
print(f"Model features: {len(features)}")


results = []

for index, flow in enumerate(flows, start=1):

    row = {}

    # Map Zeek-native features to model features
    row["Destination Port"] = float(flow.get("destination_port", 0))
    row["Flow Duration"] = float(flow.get("duration_seconds", 0))
    row["Total Fwd Packets"] = float(flow.get("orig_packets", 0))
    row["Total Backward Packets"] = float(flow.get("resp_packets", 0))
    row["Total Length of Fwd Packets"] = float(flow.get("orig_bytes", 0))
    row["Total Length of Bwd Packets"] = float(flow.get("resp_bytes", 0))
    row["Flow Bytes/s"] = float(flow.get("total_bytes_per_second", 0))
    row["Flow Packets/s"] = float(flow.get("total_packets_per_second", 0))
    row["Fwd Packets/s"] = float(flow.get("orig_packets_per_second", 0))
    row["Bwd Packets/s"] = float(flow.get("resp_packets_per_second", 0))
    row["Average Packet Size"] = float(flow.get("average_packet_size", 0))

    orig_packets = float(flow.get("orig_packets", 0))
    resp_packets = float(flow.get("resp_packets", 0))

    row["Down/Up Ratio"] = (
        resp_packets / orig_packets
        if orig_packets > 0
        else 0
    )

    row["Subflow Fwd Packets"] = orig_packets
    row["Subflow Fwd Bytes"] = float(flow.get("orig_bytes", 0))
    row["Subflow Bwd Packets"] = resp_packets
    row["Subflow Bwd Bytes"] = float(flow.get("resp_bytes", 0))

    df = pd.DataFrame([row], columns=features)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    prediction = int(model.predict(df)[0])

    probabilities = model.predict_proba(df)[0]

    benign_probability = float(probabilities[0])
    ddos_probability = float(probabilities[1])

    result = {
        "flow": index,
        "source_ip": flow.get("source_ip"),
        "destination_ip": flow.get("destination_ip"),
        "destination_port": flow.get("destination_port"),
        "protocol": flow.get("protocol"),
        "prediction": prediction,
        "label": "DDoS" if prediction == 1 else "BENIGN",
        "benign_probability": round(benign_probability, 4),
        "ddos_probability": round(ddos_probability, 4),
    }

    results.append(result)

    print(
        f"Flow {index}: "
        f"{result['label']} "
        f"(DDoS probability={ddos_probability:.4f})"
    )


OUTPUT_PATH = ROOT / "data" / "zeek" / "ml_predictions.json"

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nInference complete.")
print(f"Predictions saved to: {OUTPUT_PATH}")
