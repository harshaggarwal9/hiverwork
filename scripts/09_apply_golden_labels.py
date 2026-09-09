import csv
import json
from pathlib import Path


INPUT_FILE = Path("evaluation/golden_set.csv")
OUTPUT_FILE = Path("evaluation/golden_set.csv")
INTENTS_FILE = Path("configs/intents.json")


# ---------------------------------------------------------
# Labels for the 200 golden examples
# format:
# golden_id: (intent_label, expected_action)
# ---------------------------------------------------------

labels = {
    1: ("device_troubleshooting", "auto_handle"),
    2: ("general_or_insufficient_context", "auto_handle"),
    3: ("battery_power", "auto_handle"),
    4: ("software_update", "auto_handle"),
    5: ("software_update", "auto_handle"),
    6: ("battery_power", "auto_handle"),
    7: ("software_update", "auto_handle"),
    8: ("apple_music_media", "escalate"),
    9: ("purchase_store_refund", "escalate"),
    10: ("software_update", "auto_handle"),

    11: ("device_troubleshooting", "auto_handle"),
    12: ("connectivity", "auto_handle"),
    13: ("device_troubleshooting", "auto_handle"),
    14: ("general_or_insufficient_context", "auto_handle"),
    15: ("general_or_insufficient_context", "auto_handle"),
    16: ("how_to_settings", "auto_handle"),
    17: ("connectivity", "auto_handle"),
    18: ("apple_music_media", "auto_handle"),
    19: ("device_troubleshooting", "auto_handle"),
    20: ("general_or_insufficient_context", "auto_handle"),

    21: ("software_update", "auto_handle"),
    22: ("device_troubleshooting", "auto_handle"),
    23: ("connectivity", "auto_handle"),
    24: ("battery_power", "auto_handle"),
    25: ("account_security", "escalate"),
    26: ("device_troubleshooting", "auto_handle"),
    27: ("purchase_store_refund", "escalate"),
    28: ("apple_music_media", "auto_handle"),
    29: ("app_service_issue", "auto_handle"),
    30: ("how_to_settings", "auto_handle"),

    31: ("device_troubleshooting", "auto_handle"),
    32: ("software_update", "auto_handle"),
    33: ("battery_power", "auto_handle"),
    34: ("connectivity", "auto_handle"),
    35: ("app_service_issue", "auto_handle"),
    36: ("general_or_insufficient_context", "auto_handle"),
    37: ("purchase_store_refund", "escalate"),
    38: ("general_or_insufficient_context", "auto_handle"),
    39: ("account_security", "escalate"),
    40: ("apple_music_media", "auto_handle"),

    41: ("device_troubleshooting", "auto_handle"),
    42: ("general_or_insufficient_context", "auto_handle"),
    43: ("software_update", "auto_handle"),
    44: ("connectivity", "auto_handle"),
    45: ("battery_power", "auto_handle"),
    46: ("app_service_issue", "auto_handle"),
    47: ("device_troubleshooting", "auto_handle"),
    48: ("how_to_settings", "auto_handle"),
    49: ("purchase_store_refund", "escalate"),
    50: ("general_or_insufficient_context", "auto_handle"),

    51: ("software_update", "auto_handle"),
    52: ("device_troubleshooting", "auto_handle"),
    53: ("battery_power", "auto_handle"),
    54: ("connectivity", "auto_handle"),
    55: ("apple_music_media", "auto_handle"),
    56: ("app_service_issue", "auto_handle"),
    57: ("account_security", "escalate"),
    58: ("purchase_store_refund", "escalate"),
    59: ("how_to_settings", "auto_handle"),
    60: ("device_troubleshooting", "auto_handle"),

    61: ("software_update", "auto_handle"),
    62: ("battery_power", "auto_handle"),
    63: ("connectivity", "auto_handle"),
    64: ("device_troubleshooting", "auto_handle"),
    65: ("general_or_insufficient_context", "auto_handle"),
    66: ("apple_music_media", "auto_handle"),
    67: ("app_service_issue", "auto_handle"),
    68: ("purchase_store_refund", "escalate"),
    69: ("account_security", "escalate"),
    70: ("how_to_settings", "auto_handle"),

    71: ("general_or_insufficient_context", "auto_handle"),
    72: ("software_update", "auto_handle"),
    73: ("device_troubleshooting", "auto_handle"),
    74: ("battery_power", "auto_handle"),
    75: ("connectivity", "auto_handle"),
    76: ("apple_music_media", "auto_handle"),
    77: ("app_service_issue", "auto_handle"),
    78: ("purchase_store_refund", "escalate"),
    79: ("account_security", "escalate"),
    80: ("how_to_settings", "auto_handle"),

    81: ("device_troubleshooting", "auto_handle"),
    82: ("general_or_insufficient_context", "auto_handle"),
    83: ("software_update", "auto_handle"),
    84: ("battery_power", "auto_handle"),
    85: ("connectivity", "auto_handle"),
    86: ("apple_music_media", "auto_handle"),
    87: ("app_service_issue", "auto_handle"),
    88: ("purchase_store_refund", "escalate"),
    89: ("account_security", "escalate"),
    90: ("how_to_settings", "auto_handle"),

    91: ("software_update", "auto_handle"),
    92: ("device_troubleshooting", "auto_handle"),
    93: ("battery_power", "auto_handle"),
    94: ("connectivity", "auto_handle"),
    95: ("apple_music_media", "auto_handle"),
    96: ("app_service_issue", "auto_handle"),
    97: ("purchase_store_refund", "escalate"),
    98: ("account_security", "escalate"),
    99: ("how_to_settings", "auto_handle"),
    100: ("general_or_insufficient_context", "auto_handle"),

    101: ("device_troubleshooting", "auto_handle"),
    102: ("software_update", "auto_handle"),
    103: ("battery_power", "auto_handle"),
    104: ("connectivity", "auto_handle"),
    105: ("general_or_insufficient_context", "auto_handle"),
    106: ("apple_music_media", "auto_handle"),
    107: ("app_service_issue", "auto_handle"),
    108: ("purchase_store_refund", "escalate"),
    109: ("account_security", "escalate"),
    110: ("general_or_insufficient_context", "auto_handle"),

    111: ("device_troubleshooting", "auto_handle"),
    112: ("software_update", "auto_handle"),
    113: ("battery_power", "auto_handle"),
    114: ("connectivity", "auto_handle"),
    115: ("apple_music_media", "auto_handle"),
    116: ("app_service_issue", "auto_handle"),
    117: ("purchase_store_refund", "escalate"),
    118: ("general_or_insufficient_context", "auto_handle"),
    119: ("account_security", "escalate"),
    120: ("how_to_settings", "auto_handle"),

    121: ("device_troubleshooting", "auto_handle"),
    122: ("software_update", "auto_handle"),
    123: ("battery_power", "auto_handle"),
    124: ("connectivity", "auto_handle"),
    125: ("apple_music_media", "auto_handle"),
    126: ("app_service_issue", "auto_handle"),
    127: ("purchase_store_refund", "escalate"),
    128: ("account_security", "escalate"),
    129: ("how_to_settings", "auto_handle"),
    130: ("general_or_insufficient_context", "auto_handle"),

    131: ("device_troubleshooting", "auto_handle"),
    132: ("software_update", "auto_handle"),
    133: ("battery_power", "auto_handle"),
    134: ("connectivity", "auto_handle"),
    135: ("apple_music_media", "auto_handle"),
    136: ("app_service_issue", "auto_handle"),
    137: ("general_or_insufficient_context", "auto_handle"),
    138: ("purchase_store_refund", "escalate"),
    139: ("account_security", "escalate"),
    140: ("how_to_settings", "auto_handle"),

    141: ("general_or_insufficient_context", "auto_handle"),
    142: ("device_troubleshooting", "auto_handle"),
    143: ("software_update", "auto_handle"),
    144: ("battery_power", "auto_handle"),
    145: ("connectivity", "auto_handle"),
    146: ("apple_music_media", "auto_handle"),
    147: ("app_service_issue", "auto_handle"),
    148: ("purchase_store_refund", "escalate"),
    149: ("account_security", "escalate"),
    150: ("how_to_settings", "auto_handle"),

    151: ("device_troubleshooting", "auto_handle"),
    152: ("software_update", "auto_handle"),
    153: ("battery_power", "auto_handle"),
    154: ("connectivity", "auto_handle"),
    155: ("general_or_insufficient_context", "auto_handle"),
    156: ("apple_music_media", "auto_handle"),
    157: ("app_service_issue", "auto_handle"),
    158: ("purchase_store_refund", "escalate"),
    159: ("account_security", "escalate"),
    160: ("how_to_settings", "auto_handle"),

    161: ("device_troubleshooting", "auto_handle"),
    162: ("software_update", "auto_handle"),
    163: ("battery_power", "auto_handle"),
    164: ("connectivity", "auto_handle"),
    165: ("general_or_insufficient_context", "auto_handle"),
    166: ("apple_music_media", "auto_handle"),
    167: ("app_service_issue", "auto_handle"),
    168: ("purchase_store_refund", "escalate"),
    169: ("account_security", "escalate"),
    170: ("how_to_settings", "auto_handle"),

    171: ("general_or_insufficient_context", "auto_handle"),
    172: ("device_troubleshooting", "auto_handle"),
    173: ("software_update", "auto_handle"),
    174: ("general_or_insufficient_context", "auto_handle"),
    175: ("battery_power", "auto_handle"),
    176: ("connectivity", "auto_handle"),
    177: ("apple_music_media", "auto_handle"),
    178: ("general_or_insufficient_context", "auto_handle"),
    179: ("purchase_store_refund", "escalate"),
    180: ("account_security", "escalate"),

    181: ("general_or_insufficient_context", "auto_handle"),
    182: ("device_troubleshooting", "auto_handle"),
    183: ("general_or_insufficient_context", "auto_handle"),
    184: ("software_update", "auto_handle"),
    185: ("battery_power", "auto_handle"),
    186: ("connectivity", "auto_handle"),
    187: ("apple_music_media", "auto_handle"),
    188: ("app_service_issue", "auto_handle"),
    189: ("purchase_store_refund", "escalate"),
    190: ("account_security", "escalate"),

    191: ("device_troubleshooting", "auto_handle"),
    192: ("software_update", "auto_handle"),
    193: ("battery_power", "auto_handle"),
    194: ("connectivity", "auto_handle"),
    195: ("apple_music_media", "auto_handle"),
    196: ("app_service_issue", "auto_handle"),
    197: ("purchase_store_refund", "escalate"),
    198: ("account_security", "escalate"),
    199: ("general_or_insufficient_context", "escalate"),
    200: ("general_or_insufficient_context", "escalate"),
}


resolutions = {
    "software_update":
        "Identify the affected OS/version and provide the relevant update, workaround, or rollback guidance supported by Apple's historical responses.",

    "device_troubleshooting":
        "Identify the device and symptoms, then provide appropriate troubleshooting steps or move to private support when device-specific investigation is required.",

    "battery_power":
        "Clarify device, iOS version, charging behavior, and battery symptoms, then provide battery-health or charging guidance supported by historical responses.",

    "connectivity":
        "Identify the connection type and relevant device/settings, then provide troubleshooting or direct the customer to private support when deeper investigation is needed.",

    "apple_music_media":
        "Identify the affected media feature and provide the relevant playback, library, sync, or media troubleshooting steps.",

    "account_security":
        "Avoid exposing or requesting sensitive credentials publicly; provide safe account/security guidance and escalate for account-specific investigation when necessary.",

    "purchase_store_refund":
        "Provide the appropriate store, sales, warranty, receipt, or refund path, escalating when transaction-specific investigation is required.",

    "app_service_issue":
        "Identify the affected app/service and symptoms, then provide relevant troubleshooting or route to the appropriate service/support team.",

    "how_to_settings":
        "Give concise step-by-step instructions for the relevant Apple setting or feature.",

    "general_or_insufficient_context":
        "Request the missing details or acknowledge the customer without inventing a solution; escalate when the issue cannot be safely determined from available context."
}


notes = {
    2: "Context-dependent acknowledgement; no substantive support issue.",
    4: "Software-update issue with enough context for an automated response.",
    14: "Very short acknowledgement; prior thread context carries the meaning.",
    15: "Very short conversational reply; context should be used.",
    20: "Insufficient standalone information, but safe to acknowledge.",
    36: "Short conversational message; intent is weak without thread context.",
    38: "Context-dependent acknowledgement.",
    42: "Short conversational response.",
    50: "Insufficient context for a more specific intent.",
    65: "Short acknowledgement; avoid inventing a support issue.",
    71: "Context-dependent conversational turn.",
    82: "Insufficient standalone detail.",
    100: "Insufficient context for a specific technical category.",
    105: "Context-dependent acknowledgement.",
    109: "Security/account-specific issue should not be handled with public instructions.",
    110: "Short acknowledgement.",
    118: "Very short conversational message.",
    130: "Insufficient information for a more specific issue.",
    137: "Context-dependent response.",
    141: "Conversational/insufficient context.",
    155: "Short acknowledgement.",
    165: "Context-dependent response.",
    171: "Conversational/insufficient context.",
    174: "Very short response requiring thread context.",
    178: "Context-dependent acknowledgement.",
    181: "Insufficient context.",
    183: "Short acknowledgement or conversational continuation.",
    199: "Insufficient information and unsafe to invent a resolution.",
    200: "Insufficient information and should be escalated rather than guessing."
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            f"Make sure your golden set is saved as evaluation/golden_set.csv"
        )

    rows = []

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("golden_set.csv has no header.")

        for row in reader:
            rows.append(row)

    print(f"Rows found: {len(rows)}")

    for row in rows:
        golden_id = int(row["golden_id"])

        if golden_id not in labels:
            raise ValueError(f"Missing label for golden_id={golden_id}")

        intent, action = labels[golden_id]

        row["intent_label"] = intent
        row["expected_action"] = action
        row["acceptable_resolution"] = resolutions[intent]

        row["label_notes"] = notes.get(
            golden_id,
            f"Labeled as {intent}; expected action = {action}."
        )

    fieldnames = list(rows[0].keys())

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Validate
    intent_counts = {}
    action_counts = {}

    for row in rows:
        intent = row["intent_label"]
        action = row["expected_action"]

        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\n" + "=" * 60)
    print("GOLDEN SET LABELED")
    print("=" * 60)
    print(f"Rows: {len(rows)}")
    print(f"Output: {OUTPUT_FILE}")

    print("\nIntent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")

    print("\nAction distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")


if __name__ == "__main__":
    main()