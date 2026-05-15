import numpy as np


def compute_pure_levenshtein_cer(reference_str: str, hypothesis_str: str):
    """
    Computes Character Error Rate (CER) for speech recognition using 2D dynamic programming.
    reference_str: The ground-truth reference text sequence.
    hypothesis_str: The predicted hypothesis text sequence from the ASR model.
    """
    ref = list(reference_str)
    hyp = list(hypothesis_str)
    R_len = len(ref)
    H_len = len(hyp)

    # 1. Construct 2D dynamic programming matrix.
    # dp[i][j] represents the minimum edit distance between the prefix ref[:i] and hyp[:j].
    dp = np.zeros((R_len + 1, H_len + 1), dtype=np.int32)

    # 2. Initialize boundary conditions (base costs for deletion and insertion).
    for i in range(R_len + 1):
        dp[i, 0] = i
    for j in range(H_len + 1):
        dp[0, j] = j

    # 3. State transition logic for the 2D matrix.
    for i in range(1, R_len + 1):
        for j in range(1, H_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                # Characters match; inherit the previous state without adding cost.
                dp[i, j] = dp[i - 1, j - 1]
            else:
                # Character mismatch; find the minimum cost among Substitution, Deletion, and Insertion.
                sub_cost = dp[i - 1, j - 1] + 1  # Substitution error (S)
                del_cost = dp[i - 1, j] + 1  # Deletion error (D)
                ins_cost = dp[i, j - 1] + 1  # Insertion error (I)
                dp[i, j] = min(sub_cost, del_cost, ins_cost)

    # 4. The bottom-right corner of the matrix stores the total edit distance.
    total_edit_distance = dp[R_len, H_len]

    # 5. Calculate the final Character Error Rate standard: CER = (S + D + I) / N
    cer_score = (total_edit_distance / R_len) if R_len > 0 else 0.0
    return cer_score, total_edit_distance


if __name__ == "__main__":
    print("=" * 60)
    print(
        "[RUN] Activating Python Levenshtein dynamic programming CER evaluation test..."
    )

    # Mock ASR evaluation sample
    golden_reference = "原生多模态语音大模型"
    model_hypothesis = "原生多功能语音大模型的技术"

    cer, dist = compute_pure_levenshtein_cer(golden_reference, model_hypothesis)

    print("[SUCCESS] Text alignment calculation completed.")
    print(
        f" -> Reference length (N): {len(golden_reference)} | Hypothesis length: {len(model_hypothesis)}"
    )
    print(f" -> Total accumulated edit distance (S+D+I): {dist} steps")
    print(f" -> Final Character Error Rate (CER Score): {cer * 100:.2f}%")
    print("=" * 60)
