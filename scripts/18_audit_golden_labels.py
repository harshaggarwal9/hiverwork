import csv
import re
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

GOLDEN_FILE = Path(
    "evaluation/golden_set.csv"
)

OUTPUT_FILE = Path(
    "evaluation/golden_label_audit.csv"
)


# ============================================================
# INTENTS
# ============================================================

INTENTS = [
    "software_update",
    "device_troubleshooting",
    "battery_power",
    "connectivity",
    "apple_music_media",
    "account_security",
    "purchase_store_refund",
    "app_service_issue",
    "how_to_settings",
    "general_or_insufficient_context",
]


# ============================================================
# STRONG SIGNALS
# ============================================================

SIGNALS = {

    "software_update": [
        "ios update",
        "update ios",
        "software update",
        "update iphone",
        "update ipad",
        "update mac",
        "can't update",
        "cannot update",
        "wont update",
        "won't update",
        "update failed",
        "ios 10",
        "ios 11",
        "ios 12",
        "high sierra",
        "macos update",
    ],

    "battery_power": [
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "battery health",
        "battery dead",
        "battery power",
        "not charging",
        "won't charge",
        "wont charge",
        "charging",
        "overheating",
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "cellular",
        "mobile data",
        "network",
        "internet",
        "no service",
        "no signal",
        "can't connect",
        "cannot connect",
        "not connecting",
        "connection",
        "hotspot",
    ],

    "apple_music_media": [
        "apple music",
        "itunes",
        "music",
        "playlist",
        "song",
        "songs",
        "album",
        "podcast",
        "podcasts",
        "airplay",
        "movie",
        "video",
        "audio",
        "apple tv",
    ],

    "account_security": [
        "apple id",
        "appleid",
        "icloud",
        "password",
        "forgot password",
        "reset password",
        "account locked",
        "locked out",
        "hacked",
        "security",
        "verification code",
        "two factor",
        "2fa",
        "sign in",
        "login",
        "log in",
    ],

    "purchase_store_refund": [
        "refund",
        "money back",
        "charged",
        "billing",
        "receipt",
        "invoice",
        "purchase",
        "bought",
        "buy",
        "order",
        "payment",
        "apple store",
        "app store",
        "warranty",
        "applecare",
        "return",
        "replacement",
    ],

    "app_service_issue": [
        "app",
        "application",
        "app store",
        "not working",
        "doesn't work",
        "doesnt work",
        "won't open",
        "wont open",
        "crash",
        "crashing",
        "service down",
        "service not working",
    ],

    "how_to_settings": [
        "how do i",
        "how can i",
        "how to",
        "where do i",
        "where can i",
        "settings",
        "turn on",
        "turn off",
        "enable",
        "disable",
        "set up",
        "setup",
    ],

    "device_troubleshooting": [
        "iphone problem",
        "iphone issue",
        "ipad problem",
        "ipad issue",
        "mac problem",
        "mac issue",
        "device problem",
        "device issue",
        "screen",
        "display",
        "camera",
        "speaker",
        "microphone",
        "keyboard",
        "touchscreen",
        "restart",
        "reboot",
        "frozen",
        "freezing",
        "broken",
        "won't turn on",
        "wont turn on",
        "doesn't turn on",
        "doesnt turn on",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_signal_matches(text):
    """
    Count strong keyword signals for every intent.
    """

    text = normalize(text)

    scores = Counter()
    matches = {}

    for intent, keywords in SIGNALS.items():

        intent_matches = []

        for keyword in keywords:

            if keyword in text:
                scores[intent] += 1
                intent_matches.append(keyword)

        matches[intent] = intent_matches

    return scores, matches


def get_combined_text(row):
    """
    Audit using BOTH the conversation context and current
    customer message.
    """

    context = normalize(
        row.get("full_context", "")
    )

    message = normalize(
        row.get("customer_message", "")
    )

    if context:

        return (
            context
            + " "
            + message
        )

    return message


def calculate_risk(
    labeled_intent,
    scores
):
    """
    Produce an audit flag.

    This does NOT change the golden label.

    The goal is to identify cases that deserve human review.
    """

    if not scores:
        return (
            "REVIEW",
            "no_signal_detected"
        )

    ranked = scores.most_common()

    best_intent, best_score = ranked[0]

    labeled_score = scores.get(
        labeled_intent,
        0
    )

    # --------------------------------------------------------
    # No signal for the assigned label but strong signal for
    # another intent.
    # --------------------------------------------------------

    if (
        labeled_score == 0
        and best_score >= 2
        and best_intent != labeled_intent
    ):

        return (
            "HIGH_REVIEW",
            (
                f"assigned={labeled_intent}; "
                f"stronger_signal={best_intent}"
            )
        )

    # --------------------------------------------------------
    # Another intent has substantially stronger evidence.
    # --------------------------------------------------------

    if (
        best_intent != labeled_intent
        and best_score >= labeled_score + 2
    ):

        return (
            "HIGH_REVIEW",
            (
                f"assigned={labeled_intent}; "
                f"stronger_signal={best_intent}"
            )
        )

    # --------------------------------------------------------
    # Some evidence exists, but multiple intents overlap.
    # --------------------------------------------------------

    if len(ranked) >= 2:

        second_score = ranked[1][1]

        if (
            best_score >= 2
            and second_score >= 2
            and abs(best_score - second_score) <= 1
        ):

            return (
                "MEDIUM_REVIEW",
                (
                    f"ambiguous={ranked[0][0]}/"
                    f"{ranked[1][0]}"
                )
            )

    # --------------------------------------------------------
    # Assigned general class but a concrete intent exists.
    # --------------------------------------------------------

    if (
        labeled_intent
        == "general_or_insufficient_context"
        and best_score >= 2
    ):

        return (
            "HIGH_REVIEW",
            (
                f"general_label_but_signal="
                f"{best_intent}"
            )
        )

    # --------------------------------------------------------
    # Very short messages need contextual review.
    # --------------------------------------------------------

    message = normalize(
        " ".join(
            [
                "",
                ""
            ]
        )
    )

    return (
        "OK",
        "no_strong_conflict_detected"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GOLDEN SET LABEL AUDIT")
    print("=" * 70)

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {GOLDEN_FILE}"
        )

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    rows = []

    with GOLDEN_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    print(
        f"Golden examples loaded: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit_rows = []

    flag_counts = Counter()

    for row in rows:

        assigned_intent = normalize(
            row["intent_label"]
        )

        combined_text = get_combined_text(
            row
        )

        scores, matches = find_signal_matches(
            combined_text
        )

        flag, reason = calculate_risk(
            assigned_intent,
            scores
        )

        flag_counts[flag] += 1

        ranked = scores.most_common(3)

        strongest_intent = (
            ranked[0][0]
            if ranked
            else ""
        )

        strongest_score = (
            ranked[0][1]
            if ranked
            else 0
        )

        assigned_score = scores.get(
            assigned_intent,
            0
        )

        audit_rows.append(
            {
                "golden_id":
                    row["golden_id"],

                "customer_message":
                    row["customer_message"],

                "assigned_intent":
                    assigned_intent,

                "assigned_intent_signal_count":
                    assigned_score,

                "strongest_signal_intent":
                    strongest_intent,

                "strongest_signal_count":
                    strongest_score,

                "top_signal_2":
                    (
                        ranked[1][0]
                        if len(ranked) > 1
                        else ""
                    ),

                "top_signal_2_count":
                    (
                        ranked[1][1]
                        if len(ranked) > 1
                        else 0
                    ),

                "matched_keywords":
                    "; ".join(
                        matches.get(
                            strongest_intent,
                            []
                        )
                    ),

                "audit_flag":
                    flag,

                "audit_reason":
                    reason,
            }
        )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    fieldnames = [
        "golden_id",
        "customer_message",
        "assigned_intent",
        "assigned_intent_signal_count",
        "strongest_signal_intent",
        "strongest_signal_count",
        "top_signal_2",
        "top_signal_2_count",
        "matched_keywords",
        "audit_flag",
        "audit_reason",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(audit_rows)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    for flag in [
        "OK",
        "MEDIUM_REVIEW",
        "HIGH_REVIEW",
        "REVIEW",
    ]:

        print(
            f"{flag:18s}: "
            f"{flag_counts[flag]}"
        )

    # --------------------------------------------------------
    # Print high-priority cases
    # --------------------------------------------------------

    high_priority = [
        row
        for row in audit_rows
        if row["audit_flag"] == "HIGH_REVIEW"
    ]

    print("\n" + "=" * 70)
    print("HIGH-PRIORITY LABEL REVIEWS")
    print("=" * 70)

    for row in high_priority[:30]:

        print("\n" + "-" * 70)

        print(
            f"Golden ID: "
            f"{row['golden_id']}"
        )

        print(
            f"Message: "
            f"{row['customer_message'][:300]}"
        )

        print(
            f"Assigned intent: "
            f"{row['assigned_intent']}"
        )

        print(
            f"Strongest signal: "
            f"{row['strongest_signal_intent']}"
        )

        print(
            f"Matched keywords: "
            f"{row['matched_keywords']}"
        )

        print(
            f"Reason: "
            f"{row['audit_reason']}"
        )

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "This script ONLY flags suspicious labels."
    )

    print(
        "It does NOT automatically change the Golden Set."
    )

    print(
        "The Golden Set should remain evaluation-only."
    )

    print("\nSaved audit file:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()